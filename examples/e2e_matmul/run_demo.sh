#!/bin/bash
set -euo pipefail
#===----------------------------------------------------------------------===//
# P5-10: End-to-End Demo — ONNX → .xbin → Simulator → Verify
#
# This demo takes a 4×4 INT8 matmul ONNX model through the complete
# xTPU compilation and simulation pipeline, then verifies bit-exact
# correctness against a numpy golden reference.
#
# Pipeline:
#   1. Generate ONNX model (single_matmul_i8)
#   2. xtpu-import: ONNX → TOSA MLIR
#   3. xtpu-opt:    TOSA → Linalg → xTPU dialect
#   4. xtpu-translate: xTPU MLIR → .xbin binary
#   5. xtpu-dump:   Analyze compilation output
#   6. xbin_runner:  Execute on xTPU simulator
#   7. Verify output against golden reference
#===----------------------------------------------------------------------===//

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKDIR="$(mktemp -d)"
trap "rm -rf $WORKDIR" EXIT

# Tool paths
XTPU_IMPORT="$ROOT/compiler/tools/xtpu-import/xtpu_import.py"
XTPU_OPT="$ROOT/compiler/build/tools/xtpu-opt/xtpu-opt"
XTPU_TRANSLATE="$ROOT/compiler/tools/xtpu-translate/xtpu_translate.py"
XTPU_DUMP="$ROOT/compiler/tools/xtpu-dump/xtpu_dump.py"
XBIN_RUNNER="$ROOT/xbin_runner"
GEN_MODELS="$ROOT/compiler/tests/gen_test_models.py"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          xTPU End-to-End Demo (P5-10)                       ║"
echo "║  ONNX Model → TOSA → Linalg → xTPU → .xbin → Simulator    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Check prerequisites
for tool in "$XTPU_OPT" "$XBIN_RUNNER"; do
    if [ ! -f "$tool" ]; then
        echo "ERROR: $tool not found. Build first."
        exit 1
    fi
done

# Step 0: Generate model
echo "Step 0: Generate ONNX test model"
cd "$ROOT" && python3 "$GEN_MODELS" 2>&1 | grep single_matmul
MODEL="$ROOT/compiler/tests/single_matmul_i8.onnx"
echo "  Model: $MODEL"
echo

# Step 1: ONNX → TOSA MLIR
echo "Step 1: xtpu-import (ONNX → TOSA MLIR)"
python3 "$XTPU_IMPORT" "$MODEL" -o "$WORKDIR/model.tosa.mlir" 2>&1
echo "  Output: $WORKDIR/model.tosa.mlir"
echo "  --- TOSA IR ---"
cat "$WORKDIR/model.tosa.mlir"
echo

# Step 2: TOSA → Linalg → xTPU
echo "Step 2: xtpu-opt (TOSA → Linalg → xTPU dialect)"
"$XTPU_OPT" --tosa-to-linalg-pipeline --linalg-to-xtpu \
    "$WORKDIR/model.tosa.mlir" -o "$WORKDIR/model.xtpu.mlir" 2>&1
echo "  Output: $WORKDIR/model.xtpu.mlir"
echo "  --- xTPU IR ---"
cat "$WORKDIR/model.xtpu.mlir"
echo

# Step 3: xTPU MLIR → .xbin
echo "Step 3: xtpu-translate (xTPU MLIR → .xbin)"
python3 "$XTPU_TRANSLATE" "$WORKDIR/model.xtpu.mlir" -o "$WORKDIR/model.xbin" 2>&1
echo "  Output: $WORKDIR/model.xbin ($(wc -c < "$WORKDIR/model.xbin") bytes)"
echo

# Step 4: Analyze compilation
echo "Step 4: xtpu-dump (compilation analysis)"
python3 "$XTPU_DUMP" "$WORKDIR/model.xbin" --stats --schedule 2>&1
echo

# Step 5: Prepare input and run simulator
echo "Step 5: Execute on xTPU simulator"

# Generate input: random 4x4 int8 matrix
python3 -c "
import numpy as np
np.random.seed(42)
x = np.random.randint(-3, 4, size=(1,4,4), dtype=np.int8)
x.tofile('$WORKDIR/input.bin')
print('  Input X (4x4 int8):')
print(x.reshape(4,4))
"

"$XBIN_RUNNER" "$WORKDIR/model.xbin" \
    --input "$WORKDIR/input.bin" --input-offset 0 \
    --output "$WORKDIR/output.bin" --output-offset 4096 --output-size 16 \
    --verbose 2>&1
echo

# Step 6: Verify against golden
echo "Step 6: Verify bit-exact correctness"
python3 -c "
import numpy as np
import onnx
from onnx import numpy_helper

# Load model weights
model = onnx.load('$MODEL')
W = numpy_helper.to_array(model.graph.initializer[0])

# Load input
X = np.fromfile('$WORKDIR/input.bin', dtype=np.int8).reshape(1, 4, 4)

# Golden: int8 matmul with int32 accumulation, truncated to uint8
golden_i32 = np.matmul(X.astype(np.int32), W.astype(np.int32))
golden_u8 = (golden_i32 & 0xFF).astype(np.uint8).flatten()

# Simulator output
sim_u8 = np.fromfile('$WORKDIR/output.bin', dtype=np.uint8)

print(f'  Golden (uint8): {golden_u8.tolist()}')
print(f'  Sim    (uint8): {sim_u8.tolist()}')

if np.array_equal(golden_u8, sim_u8):
    print()
    print('  ==========================================')
    print('  =   BIT-EXACT MATCH  ✓                   =')
    print('  =   End-to-End Pipeline Verified!         =')
    print('  ==========================================')
else:
    diff = np.where(golden_u8 != sim_u8)[0]
    print(f'  MISMATCH at {len(diff)} positions')
    exit(1)
"

echo
echo "Demo complete. Pipeline: ONNX → TOSA → Linalg → xTPU → .xbin → Simulator"
