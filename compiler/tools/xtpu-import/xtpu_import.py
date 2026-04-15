#!/usr/bin/env python3
"""
xtpu-import — ONNX model → TOSA MLIR frontend (P5-2)

Parses an ONNX model and emits equivalent TOSA dialect MLIR.
The output can be further lowered via:
  xtpu-opt --tosa-to-linalg-pipeline -o model.linalg.mlir

MVP supported ops (INT8 quantized flow):
  - MatMul      → tosa.matmul (3D batch matmul)
  - Add         → tosa.add (i32 bias add)
  - Relu        → tosa.maximum with zero
  - Reshape     → tosa.reshape
  - Transpose   → tosa.transpose
  - Constant    → tosa.const

Usage:
  python3 xtpu_import.py model.onnx -o model.tosa.mlir
  python3 xtpu_import.py model.onnx --emit-linalg -o model.linalg.mlir
"""

import argparse
import sys
import os
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import onnx
    from onnx import numpy_helper, TensorProto
except ImportError:
    print("ERROR: 'onnx' package not found. Install with: pip install onnx",
          file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# ONNX dtype → MLIR type mapping
# ---------------------------------------------------------------------------
_ONNX_DTYPE_TO_MLIR = {
    TensorProto.FLOAT:    "f32",
    TensorProto.FLOAT16:  "f16",
    TensorProto.INT8:     "i8",
    TensorProto.UINT8:    "ui8",
    TensorProto.INT32:    "i32",
    TensorProto.INT64:    "i64",
    TensorProto.BOOL:     "i1",
}

# MVP supported ONNX ops
_SUPPORTED_OPS = {"MatMul", "Add", "Relu", "Reshape", "Transpose", "Constant",
                  "Gemm"}


def mlir_type_str(dtype: int) -> str:
    """Convert ONNX TensorProto dtype to MLIR type string."""
    if dtype in _ONNX_DTYPE_TO_MLIR:
        return _ONNX_DTYPE_TO_MLIR[dtype]
    raise ValueError(f"Unsupported ONNX dtype: {dtype}")


def tensor_type_str(shape: List[int], dtype_str: str) -> str:
    """Format MLIR tensor type, e.g. 'tensor<1x4x4xi8>'."""
    dims = "x".join(str(d) for d in shape)
    return f"tensor<{dims}x{dtype_str}>"


def _format_nested(arr: np.ndarray) -> str:
    """Format numpy array as nested MLIR dense literal (e.g. [[1,2],[3,4]])."""
    if arr.ndim == 0:
        return str(arr.item())
    if arr.ndim == 1:
        return "[" + ", ".join(str(v) for v in arr.tolist()) + "]"
    return "[" + ", ".join(_format_nested(arr[i]) for i in range(arr.shape[0])) + "]"


def dense_attr(arr: np.ndarray, dtype_str: str) -> str:
    """Format a small numpy array as MLIR dense attribute."""
    if arr.size == 1:
        val_str = str(arr.flat[0])
    else:
        val_str = _format_nested(arr)
    shape_str = "x".join(str(d) for d in arr.shape)
    return f"dense<{val_str}> : tensor<{shape_str}x{dtype_str}>"


class ONNXToTOSA:
    """Converts an ONNX model graph to TOSA MLIR text."""

    def __init__(self, model: onnx.ModelProto, func_name: str = "main"):
        self.model = model
        self.graph = model.graph
        self.func_name = func_name

        # SSA value map: onnx_name → (mlir_ssa_name, shape, dtype_str)
        self.values: Dict[str, Tuple[str, List[int], str]] = {}
        self.ssa_counter = 0
        self.lines: List[str] = []

        # Initializers (weights) by name
        self.initializers: Dict[str, onnx.TensorProto] = {}
        for init in self.graph.initializer:
            self.initializers[init.name] = init

        # Infer shapes
        try:
            self.model = onnx.shape_inference.infer_shapes(self.model)
            self.graph = self.model.graph
        except Exception as e:
            print(f"WARNING: shape inference failed: {e}", file=sys.stderr)

        # Build value_info map for shapes
        self.value_info: Dict[str, onnx.ValueInfoProto] = {}
        for vi in list(self.graph.value_info) + list(self.graph.input) + \
                  list(self.graph.output):
            self.value_info[vi.name] = vi

    def fresh_ssa(self) -> str:
        name = f"%v{self.ssa_counter}"
        self.ssa_counter += 1
        return name

    def get_shape_and_dtype(self, name: str) -> Tuple[List[int], str]:
        """Get shape and MLIR dtype for a named tensor."""
        if name in self.value_info:
            vi = self.value_info[name]
            tt = vi.type.tensor_type
            shape = [d.dim_value for d in tt.shape.dim]
            dtype_str = mlir_type_str(tt.elem_type)
            return shape, dtype_str

        if name in self.initializers:
            init = self.initializers[name]
            shape = list(init.dims)
            dtype_str = mlir_type_str(init.data_type)
            return shape, dtype_str

        raise ValueError(f"Cannot determine shape for '{name}'")

    def emit_initializer(self, name: str) -> str:
        """Emit an initializer (weight) as tosa.const and return SSA name."""
        init = self.initializers[name]
        arr = numpy_helper.to_array(init)
        shape = list(arr.shape)
        dtype_str = mlir_type_str(init.data_type)
        ttype = tensor_type_str(shape, dtype_str)

        ssa = self.fresh_ssa()
        attr = dense_attr(arr, dtype_str)
        self.lines.append(
            f'    {ssa} = "tosa.const"() {{values = {attr}}} : () -> {ttype}')
        self.values[name] = (ssa, shape, dtype_str)
        return ssa

    def ensure_value(self, name: str) -> str:
        """Get or emit the SSA name for an ONNX tensor."""
        if name in self.values:
            return self.values[name][0]
        if name in self.initializers:
            return self.emit_initializer(name)
        raise ValueError(f"Value '{name}' not found")

    def emit_reshape(self, ssa_in: str, old_shape: List[int],
                     new_shape: List[int], dtype_str: str) -> str:
        """Emit tosa.reshape with const_shape."""
        ndim = len(new_shape)
        shape_vals = "[" + ", ".join(str(d) for d in new_shape) + "]"
        shape_ssa = self.fresh_ssa()
        self.lines.append(
            f'    {shape_ssa} = tosa.const_shape '
            f'{{values = dense<{shape_vals}> : tensor<{ndim}xindex>}} '
            f': () -> !tosa.shape<{ndim}>')

        out_type = tensor_type_str(new_shape, dtype_str)
        in_type = tensor_type_str(old_shape, dtype_str)
        out_ssa = self.fresh_ssa()
        self.lines.append(
            f'    {out_ssa} = tosa.reshape {ssa_in}, {shape_ssa} '
            f': ({in_type}, !tosa.shape<{ndim}>) -> {out_type}')
        return out_ssa

    def emit_cast(self, ssa: str, shape: List[int],
                  from_dtype: str, to_dtype: str) -> Tuple[str, str]:
        """Emit tosa.cast and return (new_ssa, to_dtype)."""
        if from_dtype == to_dtype:
            return ssa, from_dtype
        in_type = tensor_type_str(shape, from_dtype)
        out_type = tensor_type_str(shape, to_dtype)
        out_ssa = self.fresh_ssa()
        self.lines.append(
            f'    {out_ssa} = tosa.cast {ssa} '
            f': ({in_type}) -> {out_type}')
        return out_ssa, to_dtype

    def convert_matmul(self, node: onnx.NodeProto):
        """MatMul → tosa.matmul (3D batch matmul, i8→i32)."""
        a_name, b_name = node.input[0], node.input[1]
        a_ssa = self.ensure_value(a_name)
        b_ssa = self.ensure_value(b_name)
        a_shape, a_dtype = self.values[a_name][1], self.values[a_name][2]
        b_shape, b_dtype = self.values[b_name][1], self.values[b_name][2]

        # Cast i32 inputs back to i8 for quantized matmul
        if a_dtype == "i32":
            a_ssa, a_dtype = self.emit_cast(a_ssa, a_shape, "i32", "i8")
        if b_dtype == "i32":
            b_ssa, b_dtype = self.emit_cast(b_ssa, b_shape, "i32", "i8")

        # Ensure 3D for tosa.matmul
        if len(a_shape) == 2:
            new_a_shape = [1] + a_shape
            a_ssa = self.emit_reshape(a_ssa, a_shape, new_a_shape, a_dtype)
            a_shape = new_a_shape
        if len(b_shape) == 2:
            new_b_shape = [1] + b_shape
            b_ssa = self.emit_reshape(b_ssa, b_shape, new_b_shape, b_dtype)
            b_shape = new_b_shape

        # Determine output dtype (matmul i8×i8 → i32)
        out_dtype = "i32" if a_dtype in ("i8", "ui8") else a_dtype
        out_shape = [a_shape[0], a_shape[1], b_shape[2]]
        out_type = tensor_type_str(out_shape, out_dtype)
        a_type = tensor_type_str(a_shape, a_dtype)
        b_type = tensor_type_str(b_shape, b_dtype)

        # Zero points for quantized matmul
        zp_type = tensor_type_str([1], a_dtype)
        azp = self.fresh_ssa()
        bzp = self.fresh_ssa()
        self.lines.append(
            f'    {azp} = "tosa.const"() '
            f'{{values = dense<0> : {zp_type}}} : () -> {zp_type}')
        self.lines.append(
            f'    {bzp} = "tosa.const"() '
            f'{{values = dense<0> : {zp_type}}} : () -> {zp_type}')

        out_ssa = self.fresh_ssa()
        self.lines.append(
            f'    {out_ssa} = tosa.matmul {a_ssa}, {b_ssa}, {azp}, {bzp} '
            f': ({a_type}, {b_type}, {zp_type}, {zp_type}) -> {out_type}')

        out_name = node.output[0]
        self.values[out_name] = (out_ssa, out_shape, out_dtype)

    def convert_gemm(self, node: onnx.NodeProto):
        """Gemm → tosa.matmul + optional bias add.

        Handles: Y = alpha * A @ B + beta * C (simplified: alpha=1, beta=1)
        transB attribute is common in ONNX Gemm.
        """
        attrs = {a.name: a for a in node.attribute}
        trans_b = attrs.get("transB", None)
        trans_b = trans_b.i if trans_b else 0

        a_name = node.input[0]
        b_name = node.input[1]
        a_ssa = self.ensure_value(a_name)
        b_ssa = self.ensure_value(b_name)
        a_shape, a_dtype = self.values[a_name][1], self.values[a_name][2]
        b_shape, b_dtype = self.values[b_name][1], self.values[b_name][2]

        # Track original dimensionality for reshaping back
        orig_a_ndim = len(a_shape)

        # Cast i32 inputs back to i8 for quantized matmul
        if a_dtype == "i32":
            a_ssa, a_dtype = self.emit_cast(a_ssa, a_shape, "i32", "i8")
        if b_dtype == "i32":
            b_ssa, b_dtype = self.emit_cast(b_ssa, b_shape, "i32", "i8")

        # Handle transB
        if trans_b and len(b_shape) == 2:
            b_type_in = tensor_type_str(b_shape, b_dtype)
            b_shape = [b_shape[1], b_shape[0]]
            b_type_out = tensor_type_str(b_shape, b_dtype)
            new_b = self.fresh_ssa()
            self.lines.append(
                f'    {new_b} = tosa.transpose {b_ssa} '
                f'{{perms = array<i32: 1, 0>}} '
                f': ({b_type_in}) -> {b_type_out}')
            b_ssa = new_b

        # Reshape to 3D for tosa.matmul
        if len(a_shape) == 2:
            new_a = [1] + a_shape
            a_ssa = self.emit_reshape(a_ssa, a_shape, new_a, a_dtype)
            a_shape = new_a
        if len(b_shape) == 2:
            new_b = [1] + b_shape
            b_ssa = self.emit_reshape(b_ssa, b_shape, new_b, b_dtype)
            b_shape = new_b

        out_dtype = "i32" if a_dtype in ("i8", "ui8") else a_dtype
        out_shape = [a_shape[0], a_shape[1], b_shape[2]]
        out_type = tensor_type_str(out_shape, out_dtype)
        a_type = tensor_type_str(a_shape, a_dtype)
        b_type = tensor_type_str(b_shape, b_dtype)

        zp_type = tensor_type_str([1], a_dtype)
        azp = self.fresh_ssa()
        bzp = self.fresh_ssa()
        self.lines.append(
            f'    {azp} = "tosa.const"() '
            f'{{values = dense<0> : {zp_type}}} : () -> {zp_type}')
        self.lines.append(
            f'    {bzp} = "tosa.const"() '
            f'{{values = dense<0> : {zp_type}}} : () -> {zp_type}')

        mm_ssa = self.fresh_ssa()
        self.lines.append(
            f'    {mm_ssa} = tosa.matmul {a_ssa}, {b_ssa}, {azp}, {bzp} '
            f': ({a_type}, {b_type}, {zp_type}, {zp_type}) -> {out_type}')

        result_ssa = mm_ssa
        result_shape = out_shape

        # Optional bias (third input)
        if len(node.input) >= 3 and node.input[2]:
            c_name = node.input[2]
            c_ssa = self.ensure_value(c_name)
            c_shape, c_dtype = self.values[c_name][1], self.values[c_name][2]

            # Broadcast bias to match matmul output shape
            if c_shape != out_shape:
                # Typical: bias is [N], need [1, 1, N]
                new_c_shape = [1] * (len(out_shape) - len(c_shape)) + c_shape
                if new_c_shape != c_shape:
                    c_ssa = self.emit_reshape(c_ssa, c_shape, new_c_shape,
                                              c_dtype)
                    c_shape = new_c_shape

            c_type = tensor_type_str(c_shape, c_dtype)
            add_ssa = self.fresh_ssa()
            self.lines.append(
                f'    {add_ssa} = tosa.add {mm_ssa}, {c_ssa} '
                f': ({out_type}, {c_type}) -> {out_type}')
            result_ssa = add_ssa

        # Reshape back to 2D if original input was 2D (Gemm is a 2D op)
        if orig_a_ndim == 2 and len(result_shape) == 3:
            flat_shape = result_shape[1:]  # drop batch dim [1,M,N] → [M,N]
            result_ssa = self.emit_reshape(result_ssa, result_shape,
                                           flat_shape, out_dtype)
            result_shape = flat_shape

        out_name = node.output[0]
        self.values[out_name] = (result_ssa, result_shape, out_dtype)

    def convert_add(self, node: onnx.NodeProto):
        """Add → tosa.add."""
        a_name, b_name = node.input[0], node.input[1]
        a_ssa = self.ensure_value(a_name)
        b_ssa = self.ensure_value(b_name)
        a_shape, a_dtype = self.values[a_name][1], self.values[a_name][2]
        b_shape, b_dtype = self.values[b_name][1], self.values[b_name][2]

        # Broadcast b if needed
        if b_shape != a_shape and len(b_shape) < len(a_shape):
            new_b_shape = [1] * (len(a_shape) - len(b_shape)) + b_shape
            if new_b_shape != b_shape:
                b_ssa = self.emit_reshape(b_ssa, b_shape, new_b_shape, b_dtype)
                b_shape = new_b_shape

        # Cast to match types if needed (tosa.add requires same element type)
        if a_dtype != b_dtype:
            # Promote i8 to i32 (or the wider type)
            wider = a_dtype if a_dtype == "i32" else b_dtype
            if a_dtype != wider:
                a_ssa, a_dtype = self.emit_cast(a_ssa, a_shape, a_dtype, wider)
            if b_dtype != wider:
                b_ssa, b_dtype = self.emit_cast(b_ssa, b_shape, b_dtype, wider)

        a_type = tensor_type_str(a_shape, a_dtype)
        b_type = tensor_type_str(b_shape, b_dtype)

        out_shape = a_shape  # simplified broadcast
        out_type = tensor_type_str(out_shape, a_dtype)

        out_ssa = self.fresh_ssa()
        self.lines.append(
            f'    {out_ssa} = tosa.add {a_ssa}, {b_ssa} '
            f': ({a_type}, {b_type}) -> {out_type}')

        self.values[node.output[0]] = (out_ssa, out_shape, a_dtype)

    def convert_relu(self, node: onnx.NodeProto):
        """Relu → tosa.maximum(x, 0)."""
        x_name = node.input[0]
        x_ssa = self.ensure_value(x_name)
        x_shape, x_dtype = self.values[x_name][1], self.values[x_name][2]
        x_type = tensor_type_str(x_shape, x_dtype)

        # Zero constant with same shape
        zero_ssa = self.fresh_ssa()
        self.lines.append(
            f'    {zero_ssa} = "tosa.const"() '
            f'{{values = dense<0> : {x_type}}} : () -> {x_type}')

        out_ssa = self.fresh_ssa()
        self.lines.append(
            f'    {out_ssa} = tosa.maximum {x_ssa}, {zero_ssa} '
            f': ({x_type}, {x_type}) -> {x_type}')

        self.values[node.output[0]] = (out_ssa, x_shape, x_dtype)

    def convert_reshape(self, node: onnx.NodeProto):
        """Reshape → tosa.reshape."""
        x_name = node.input[0]
        x_ssa = self.ensure_value(x_name)
        x_shape, x_dtype = self.values[x_name][1], self.values[x_name][2]

        out_shape, _ = self.get_shape_and_dtype(node.output[0])
        out_ssa = self.emit_reshape(x_ssa, x_shape, out_shape, x_dtype)
        self.values[node.output[0]] = (out_ssa, out_shape, x_dtype)

    def convert_transpose(self, node: onnx.NodeProto):
        """Transpose → tosa.transpose."""
        x_name = node.input[0]
        x_ssa = self.ensure_value(x_name)
        x_shape, x_dtype = self.values[x_name][1], self.values[x_name][2]

        attrs = {a.name: a for a in node.attribute}
        perm = list(attrs["perm"].ints) if "perm" in attrs else list(
            range(len(x_shape) - 1, -1, -1))

        perm_vals = ", ".join(str(p) for p in perm)

        out_shape = [x_shape[p] for p in perm]
        x_type = tensor_type_str(x_shape, x_dtype)
        out_type = tensor_type_str(out_shape, x_dtype)

        out_ssa = self.fresh_ssa()
        self.lines.append(
            f'    {out_ssa} = tosa.transpose {x_ssa} '
            f'{{perms = array<i32: {perm_vals}>}} '
            f': ({x_type}) -> {out_type}')

        self.values[node.output[0]] = (out_ssa, out_shape, x_dtype)

    def convert(self) -> str:
        """Convert the full ONNX graph and return MLIR text."""
        # Check for unsupported ops first
        unsupported = []
        for node in self.graph.node:
            if node.op_type not in _SUPPORTED_OPS:
                unsupported.append((node.name or "<unnamed>", node.op_type))

        if unsupported:
            print("ERROR: Unsupported ONNX ops found:", file=sys.stderr)
            for name, op in unsupported:
                print(f"  - node '{name}': op_type='{op}'", file=sys.stderr)
            print(f"\nSupported ops: {sorted(_SUPPORTED_OPS)}",
                  file=sys.stderr)
            sys.exit(1)

        # Register graph inputs (non-initializer)
        init_names = set(self.initializers.keys())
        func_inputs: List[Tuple[str, str, List[int], str]] = []
        arg_idx = 0

        for inp in self.graph.input:
            if inp.name in init_names:
                continue  # Will be emitted as tosa.const
            shape, dtype_str = self.get_shape_and_dtype(inp.name)
            arg_name = f"%arg{arg_idx}"
            func_inputs.append((inp.name, arg_name, shape, dtype_str))
            self.values[inp.name] = (arg_name, shape, dtype_str)
            arg_idx += 1

        # Convert each node
        for node in self.graph.node:
            op = node.op_type
            if op == "Constant":
                # Extract constant value
                attrs = {a.name: a for a in node.attribute}
                if "value" in attrs:
                    t = attrs["value"].t
                    arr = numpy_helper.to_array(t)
                    shape = list(arr.shape)
                    dtype_str = mlir_type_str(t.data_type)
                    ttype = tensor_type_str(shape, dtype_str)
                    ssa = self.fresh_ssa()
                    attr = dense_attr(arr, dtype_str)
                    self.lines.append(
                        f'    {ssa} = "tosa.const"() '
                        f'{{values = {attr}}} : () -> {ttype}')
                    self.values[node.output[0]] = (ssa, shape, dtype_str)
            elif op == "MatMul":
                self.convert_matmul(node)
            elif op == "Gemm":
                self.convert_gemm(node)
            elif op == "Add":
                self.convert_add(node)
            elif op == "Relu":
                self.convert_relu(node)
            elif op == "Reshape":
                self.convert_reshape(node)
            elif op == "Transpose":
                self.convert_transpose(node)

        # Get outputs
        func_outputs = []
        for out in self.graph.output:
            ssa, shape, dtype_str = self.values[out.name]
            func_outputs.append((ssa, shape, dtype_str))

        # Assemble MLIR text
        # Function signature
        args_str = ", ".join(
            f"{arg}: {tensor_type_str(shape, dt)}"
            for _, arg, shape, dt in func_inputs)
        rets_str = ", ".join(
            tensor_type_str(shape, dt) for _, shape, dt in func_outputs)

        header = f'func.func @{self.func_name}({args_str}) -> ({rets_str}) {{'
        ret_vals = ", ".join(ssa for ssa, _, _ in func_outputs)
        ret_types = ", ".join(
            tensor_type_str(shape, dt) for _, shape, dt in func_outputs)
        footer = f'    return {ret_vals} : {ret_types}\n  }}'

        body = "\n".join(self.lines)
        return f"module {{\n  {header}\n{body}\n{footer}\n}}\n"


def main():
    parser = argparse.ArgumentParser(
        description="xtpu-import: ONNX model → TOSA MLIR frontend (P5-2)")
    parser.add_argument("input", help="Input ONNX model (.onnx)")
    parser.add_argument("-o", "--output", default="-",
                        help="Output MLIR file (default: stdout)")
    parser.add_argument("--func-name", default="main",
                        help="MLIR function name (default: main)")
    parser.add_argument("--emit-linalg", action="store_true",
                        help="Also lower TOSA→Linalg via mlir-opt "
                             "(requires mlir-opt in PATH or --mlir-opt)")
    parser.add_argument("--mlir-opt", default=None,
                        help="Path to mlir-opt binary (for --emit-linalg)")
    parser.add_argument("--list-ops", action="store_true",
                        help="List all ops in the ONNX model and exit")
    args = parser.parse_args()

    # Load ONNX model
    if not os.path.isfile(args.input):
        print(f"ERROR: File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    model = onnx.load(args.input)
    onnx.checker.check_model(model)

    # --list-ops mode
    if args.list_ops:
        ops = {}
        for node in model.graph.node:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1
        print("ONNX ops in model:")
        for op, count in sorted(ops.items()):
            supported = "✓" if op in _SUPPORTED_OPS else "✗"
            print(f"  [{supported}] {op} × {count}")
        unsupported = [op for op in ops if op not in _SUPPORTED_OPS]
        if unsupported:
            print(f"\nUnsupported: {unsupported}")
            sys.exit(1)
        else:
            print("\nAll ops supported!")
        return

    # Convert
    converter = ONNXToTOSA(model, func_name=args.func_name)
    mlir_text = converter.convert()

    # Optional: pipe through mlir-opt for TOSA→Linalg
    if args.emit_linalg:
        import subprocess
        mlir_opt = args.mlir_opt or "mlir-opt"
        try:
            result = subprocess.run(
                [mlir_opt, "--tosa-to-linalg-pipeline"],
                input=mlir_text, capture_output=True, text=True, check=True)
            mlir_text = result.stdout
        except FileNotFoundError:
            print(f"ERROR: mlir-opt not found at '{mlir_opt}'. "
                  f"Set --mlir-opt path.", file=sys.stderr)
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: mlir-opt --tosa-to-linalg-pipeline failed:\n"
                  f"{e.stderr}", file=sys.stderr)
            sys.exit(1)

    # Output
    if args.output == "-":
        sys.stdout.write(mlir_text)
    else:
        with open(args.output, "w") as f:
            f.write(mlir_text)
        print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
