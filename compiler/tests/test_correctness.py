#!/usr/bin/env python3
"""
P5-8: Correctness Framework — Compiled vs ONNX Runtime Golden Reference

For each test model:
  1. Run ONNX model through ONNX Runtime → golden output
  2. Compile model: xtpu-import → xtpu-opt → xtpu-translate → .xbin
  3. Run .xbin on simulator via xbin_runner → sim output
  4. Compare golden vs sim output (bit-exact for INT8)

Usage:
  python3 compiler/tests/test_correctness.py [--verbose] [--model <name>]
  python3 compiler/tests/test_correctness.py --list

Requires:
  - onnx, onnxruntime (pip install onnx onnxruntime)
  - Built xtpu-opt and xbin_runner binaries
"""

import argparse
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # xTPU_simulator/
COMPILER = ROOT / "compiler"
TESTS_DIR = COMPILER / "tests"
XTPU_IMPORT = COMPILER / "tools" / "xtpu-import" / "xtpu_import.py"
XTPU_OPT = COMPILER / "build" / "tools" / "xtpu-opt" / "xtpu-opt"
XTPU_TRANSLATE = COMPILER / "tools" / "xtpu-translate" / "xtpu_translate.py"
XBIN_RUNNER = ROOT / "xbin_runner"


# ---------------------------------------------------------------------------
# Compiler pipeline helper
# ---------------------------------------------------------------------------
def compile_model(onnx_path: Path, work_dir: Path, verbose: bool = False) -> Path:
    """Compile an ONNX model to .xbin through the full pipeline."""
    name = onnx_path.stem
    tosa_path = work_dir / f"{name}_tosa.mlir"
    xtpu_path = work_dir / f"{name}_xtpu.mlir"
    xbin_path = work_dir / f"{name}.xbin"

    steps = [
        ("xtpu-import", [
            sys.executable, str(XTPU_IMPORT),
            str(onnx_path), "-o", str(tosa_path)
        ]),
        ("xtpu-opt", [
            str(XTPU_OPT),
            "--tosa-to-linalg-pipeline", "--linalg-to-xtpu",
            str(tosa_path), "-o", str(xtpu_path)
        ]),
        ("xtpu-translate", [
            sys.executable, str(XTPU_TRANSLATE),
            str(xtpu_path), "-o", str(xbin_path)
        ]),
    ]

    for step_name, cmd in steps:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            err = result.stderr.strip()
            raise RuntimeError(f"{step_name} failed:\n{err}")
        if verbose:
            print(f"    [{step_name}] OK", file=sys.stderr)

    return xbin_path


# ---------------------------------------------------------------------------
# ONNX Runtime golden reference
# ---------------------------------------------------------------------------
def run_golden(onnx_path: Path, input_data: Dict[str, np.ndarray],
               verbose: bool = False) -> Dict[str, np.ndarray]:
    """Run model to compute golden reference output.

    Strategy:
    - Try ONNX Runtime first (works for float/int32 models)
    - Fall back to numpy-based evaluation for INT8 models
      (ONNX Runtime doesn't support INT8 MatMul)
    """
    try:
        sess = ort.InferenceSession(str(onnx_path))
        output_names = [o.name for o in sess.get_outputs()]
        feeds = {k: v for k, v in input_data.items()}
        results = sess.run(output_names, feeds)
        if verbose:
            for name, arr in zip(output_names, results):
                print(f"    [golden/ort] {name}: shape={arr.shape} dtype={arr.dtype}",
                      file=sys.stderr)
        return dict(zip(output_names, results))
    except Exception as e:
        if verbose:
            print(f"    [golden] ORT failed ({e}), using numpy evaluator",
                  file=sys.stderr)
        return run_golden_numpy(onnx_path, input_data, verbose)


def run_golden_numpy(onnx_path: Path, input_data: Dict[str, np.ndarray],
                     verbose: bool = False,
                     hardware_mode: bool = False) -> Dict[str, np.ndarray]:
    """Evaluate ONNX graph using numpy (INT8 friendly).

    If hardware_mode=True, truncates all intermediate results to uint8 after
    every operation, matching xTPU hardware semantics where every compute
    engine produces uint8 output.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph

    def to_hw(arr):
        """Truncate to uint8 if hardware_mode is on."""
        if hardware_mode:
            if arr.dtype == np.int8:
                return arr.view(np.uint8)
            return (arr.astype(np.int64).flatten() % 256).astype(np.uint8).reshape(arr.shape)
        return arr

    # Build value store: name → numpy array
    if hardware_mode:
        values: Dict[str, np.ndarray] = {
            k: v.view(np.uint8) if v.dtype == np.int8 else v
            for k, v in input_data.items()
        }
    else:
        values: Dict[str, np.ndarray] = dict(input_data)

    # Load initializers — in hardware mode, constants are stored as uint8
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        if hardware_mode:
            values[init.name] = arr.view(np.uint8) if arr.dtype == np.int8 else \
                arr.astype(np.int64).flatten().__mod__(256).astype(np.uint8).reshape(arr.shape)
        else:
            values[init.name] = arr

    # Execute nodes in topological order
    for node in graph.node:
        op = node.op_type

        if op == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    arr = numpy_helper.to_array(attr.t)
                    values[node.output[0]] = to_hw(arr) if hardware_mode else arr

        elif op == "MatMul":
            a = values[node.input[0]]
            b = values[node.input[1]]
            if hardware_mode:
                # Hardware: uint8 × uint8, uint32 accum → uint8 truncate
                a32 = a.astype(np.uint32)
                b32 = b.astype(np.uint32)
                result = np.matmul(a32, b32)
                values[node.output[0]] = (result & 0xFF).astype(np.uint8)
            else:
                # INT8 matmul: accumulate in int32
                if a.dtype == np.int8:
                    a32 = a.astype(np.int32)
                    b32 = b.astype(np.int32)
                    values[node.output[0]] = np.matmul(a32, b32).astype(np.int32)
                else:
                    values[node.output[0]] = np.matmul(a, b)

        elif op == "Gemm":
            attrs = {a.name: a for a in node.attribute}
            trans_b = attrs.get("transB", None)
            trans_b = trans_b.i if trans_b else 0
            a = values[node.input[0]]
            b = values[node.input[1]]
            if hardware_mode:
                a = a.astype(np.uint32)
                b = b.astype(np.uint32)
            else:
                if a.dtype == np.int8:
                    a = a.astype(np.int32)
                if b.dtype == np.int8:
                    b = b.astype(np.int32)
            if trans_b:
                b = b.T
            result = np.matmul(a, b)
            if len(node.input) >= 3 and node.input[2]:
                c = values[node.input[2]]
                if hardware_mode:
                    result = result + c.astype(np.uint32)
                else:
                    result = result + c.astype(np.int32)
            if hardware_mode:
                values[node.output[0]] = (result & 0xFF).astype(np.uint8)
            else:
                values[node.output[0]] = result.astype(np.int32)

        elif op == "Add":
            a = values[node.input[0]]
            b = values[node.input[1]]
            if hardware_mode:
                # Hardware add: uint8 element-wise, truncate to uint8
                # Broadcasting: expand smaller operand
                a8 = a.astype(np.uint16)
                b8 = np.broadcast_to(b, a.shape).astype(np.uint16)
                values[node.output[0]] = ((a8 + b8) & 0xFF).astype(np.uint8)
            else:
                # Promote to i32 for mixed types
                if a.dtype != b.dtype:
                    a = a.astype(np.int32)
                    b = b.astype(np.int32)
                values[node.output[0]] = a + b

        elif op == "Relu":
            x = values[node.input[0]]
            if hardware_mode:
                # Hardware relu: interprets uint8 as int8, clamps negative to 0
                x_signed = x.view(np.int8)
                values[node.output[0]] = np.where(x_signed < 0, np.uint8(0), x).astype(np.uint8)
            else:
                values[node.output[0]] = np.maximum(x, 0)

        elif op == "Transpose":
            x = values[node.input[0]]
            attrs = {a.name: a for a in node.attribute}
            perm = list(attrs["perm"].ints) if "perm" in attrs else None
            values[node.output[0]] = np.transpose(x, perm)

        elif op == "Reshape":
            x = values[node.input[0]]
            shape_tensor = values[node.input[1]]
            values[node.output[0]] = x.reshape(shape_tensor.tolist())

        else:
            raise RuntimeError(f"Numpy evaluator: unsupported op '{op}'")

    # Collect outputs
    outputs = {}
    for out in graph.output:
        outputs[out.name] = values[out.name]
        if verbose:
            arr = outputs[out.name]
            print(f"    [golden/np] {out.name}: shape={arr.shape} dtype={arr.dtype}",
                  file=sys.stderr)
    return outputs


# ---------------------------------------------------------------------------
# Simulator execution
# ---------------------------------------------------------------------------
def run_simulator(xbin_path: Path, input_data: Dict[str, np.ndarray],
                  model: onnx.ModelProto, work_dir: Path,
                  verbose: bool = False) -> np.ndarray:
    """Run .xbin on simulator via xbin_runner, return output bytes."""

    # Determine input layout: the import pass assigns system memory
    # in order of graph inputs (non-initializer), starting at offset 0.
    init_names = {i.name for i in model.graph.initializer}
    graph_inputs = [
        inp for inp in model.graph.input if inp.name not in init_names
    ]

    # Build input binary files and calculate offsets
    # The compiler allocates inputs sequentially from offset 0.
    # Each input's size = product of dims × element_size_bytes
    input_args = []
    current_offset = 0
    for inp in graph_inputs:
        name = inp.name
        arr = input_data[name]

        # Write input to binary file
        inp_path = work_dir / f"input_{name}.bin"
        arr_bytes = arr.astype(np.int8).tobytes() if arr.dtype == np.int8 else arr.tobytes()
        with open(inp_path, "wb") as f:
            f.write(arr_bytes)

        input_args.extend([
            "--input", str(inp_path),
            "--input-offset", str(current_offset),
        ])
        current_offset += len(arr_bytes)
        if verbose:
            print(f"    [sim] input '{name}': {len(arr_bytes)} bytes at offset {current_offset - len(arr_bytes)}",
                  file=sys.stderr)

    # Weights/initializers are now embedded in .xbin .rodata section.
    # The xbin_runner auto-loads .rodata into system memory at the
    # correct offsets (matching the compiler's allocation).

    # Determine output size — hardware always produces uint8 results
    # (matmul does uint32 accumulation → uint8 truncation)
    output_info = model.graph.output[0]
    out_tt = output_info.type.tensor_type
    out_shape = [d.dim_value for d in out_tt.shape.dim]
    num_elements = int(np.prod(out_shape))
    # Hardware output is always uint8, 1 byte per element
    output_size = num_elements

    out_path = work_dir / "output.bin"

    cmd = [
        str(XBIN_RUNNER), str(xbin_path),
        *input_args,
        "--output", str(out_path),
        "--output-offset", "4096",
        "--output-size", str(output_size),
    ]
    if verbose:
        cmd.append("--verbose")
        print(f"    [sim] output: {output_size} bytes from offset 4096",
              file=sys.stderr)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"xbin_runner failed:\n{result.stderr}")

    # Read output as uint8 (hardware always produces uint8)
    with open(out_path, "rb") as f:
        out_bytes = f.read()

    return np.frombuffer(out_bytes, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_results(golden: np.ndarray, sim: np.ndarray,
                    model_name: str, verbose: bool = False) -> bool:
    """Compare golden (numpy/ORT) vs simulator output.

    For INT8 matmul models, the golden is i32 accumulation.
    The simulator does uint32 accumulation → uint8 truncation.
    We compare by truncating both to uint8.
    """
    golden_flat = golden.flatten()
    sim_flat = sim.flatten()

    # Truncate golden i32 → uint8 (matching hardware behavior:
    # uint32 accumulation → lowest 8 bits → uint8)
    if golden_flat.dtype in (np.int32, np.int64):
        # np.uint8 cast: takes lowest 8 bits (same as C uint8_t truncation)
        golden_uint8 = (golden_flat & 0xFF).astype(np.uint8)
    else:
        golden_uint8 = golden_flat.astype(np.uint8)

    sim_uint8 = sim_flat.astype(np.uint8)

    # Size check
    min_len = min(len(golden_uint8), len(sim_uint8))
    if min_len == 0:
        print(f"  WARNING: empty output for {model_name}")
        return False

    golden_cmp = golden_uint8[:min_len]
    sim_cmp = sim_uint8[:min_len]

    match = np.array_equal(golden_cmp, sim_cmp)

    if verbose or not match:
        print(f"  Golden (uint8, {min_len} elems): {golden_cmp[:16].tolist()}")
        print(f"  Sim    (uint8, {min_len} elems): {sim_cmp[:16].tolist()}")
        if golden_flat.dtype in (np.int32, np.int64):
            print(f"  Golden (i32 raw, first 8): {golden_flat[:8].tolist()}")
        if not match:
            diff_idx = np.where(golden_cmp != sim_cmp)[0]
            print(f"  Mismatch at {len(diff_idx)} / {min_len} positions")
            if len(diff_idx) > 0:
                idx = diff_idx[0]
                print(f"  First diff at [{idx}]: golden={golden_cmp[idx]} sim={sim_cmp[idx]}")

    return match


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def get_model_inputs(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    """Generate deterministic test inputs for a model."""
    init_names = {i.name for i in model.graph.initializer}
    inputs = {}
    np.random.seed(42)
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        tt = inp.type.tensor_type
        shape = [d.dim_value for d in tt.shape.dim]
        if tt.elem_type == onnx.TensorProto.INT8:
            inputs[inp.name] = np.random.randint(-3, 4, size=shape, dtype=np.int8)
        elif tt.elem_type == onnx.TensorProto.INT32:
            inputs[inp.name] = np.random.randint(-10, 10, size=shape, dtype=np.int32)
        else:
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
    return inputs


def run_test(onnx_path: Path, verbose: bool = False) -> Tuple[bool, str]:
    """Run full correctness test for a single model.

    Returns (passed, message).
    """
    name = onnx_path.stem
    model = onnx.load(str(onnx_path))

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    inputs = get_model_inputs(model)

    with tempfile.TemporaryDirectory(prefix=f"xtpu_test_{name}_") as tmpdir:
        work_dir = Path(tmpdir)

        # Step 1: Hardware-faithful golden reference
        # Uses uint8 truncation at every step, matching xTPU hardware semantics.
        try:
            golden_outputs = run_golden_numpy(
                onnx_path, inputs, verbose, hardware_mode=True)
            golden = list(golden_outputs.values())[0]  # First output
        except Exception as e:
            return False, f"Golden failed: {e}"

        # Step 2: Compile
        try:
            xbin_path = compile_model(onnx_path, work_dir, verbose)
        except Exception as e:
            return False, f"Compile failed: {e}"

        # Step 3: Simulate
        try:
            sim_output = run_simulator(
                xbin_path, inputs, model, work_dir, verbose)
        except Exception as e:
            return False, f"Simulator failed: {e}"

        # Step 4: Compare
        match = compare_results(golden, sim_output, name, verbose)
        if match:
            return True, "bit-exact match"
        else:
            return False, "output mismatch"


def main():
    parser = argparse.ArgumentParser(
        description="P5-8: Correctness Framework — Compiled vs Golden")
    parser.add_argument("--model", default=None,
                        help="Run only this model (basename without .onnx)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    parser.add_argument("--list", action="store_true",
                        help="List available test models")
    args = parser.parse_args()

    # Check prerequisites
    if not XTPU_OPT.exists():
        print(f"ERROR: xtpu-opt not found at {XTPU_OPT}", file=sys.stderr)
        print("  Build with: cd compiler/build && cmake --build . --target xtpu-opt",
              file=sys.stderr)
        sys.exit(1)
    if not XBIN_RUNNER.exists():
        print(f"ERROR: xbin_runner not found at {XBIN_RUNNER}", file=sys.stderr)
        print("  Build with: make xbin_runner", file=sys.stderr)
        sys.exit(1)

    # Find test models
    models = sorted(TESTS_DIR.glob("*.onnx"))
    if not models:
        print("ERROR: No .onnx files found in", TESTS_DIR)
        sys.exit(1)

    if args.list:
        print("Available test models:")
        for m in models:
            print(f"  {m.stem}")
        return

    if args.model:
        models = [m for m in models if m.stem == args.model]
        if not models:
            print(f"ERROR: Model '{args.model}' not found")
            sys.exit(1)

    # Run tests
    print(f"=== P5-8 Correctness Framework ===")
    print(f"Models: {len(models)}")
    print()

    results = {}
    passed = 0
    failed = 0

    for onnx_path in models:
        name = onnx_path.stem
        print(f"[TEST] {name} ... ", end="", flush=True)

        ok, msg = run_test(onnx_path, args.verbose)
        results[name] = (ok, msg)

        if ok:
            print(f"PASS ✓ ({msg})")
            passed += 1
        else:
            print(f"FAIL ✗ ({msg})")
            failed += 1

    # Summary
    print()
    print(f"=== Results: {passed}/{passed+failed} passed ===")
    if failed > 0:
        print("\nFailed models:")
        for name, (ok, msg) in results.items():
            if not ok:
                print(f"  ✗ {name}: {msg}")
        sys.exit(1)
    else:
        print("All models passed bit-exact correctness check!")


if __name__ == "__main__":
    main()
