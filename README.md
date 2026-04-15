# xTPU Simulator

A behavioral-accurate C++ simulator for a VLIW-based TPU (Tensor Processing Unit),
with a full ONNX-to-binary compiler toolchain built on MLIR.

The simulator models asynchronous multi-engine execution (sDMA, iDMA, two PUs),
LPDDR5 memory timing, and a bitmask-based scoreboard for hardware–software
synchronization.  The companion compiler lowers ONNX models through
TOSA → Linalg → xTPU MLIR → `.xbin` binary, which the simulator executes.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Repository Layout](#2-repository-layout)
3. [Requirements](#3-requirements)
4. [Quick Start](#4-quick-start)
5. [Building the Simulator](#5-building-the-simulator)
6. [Building the Compiler](#6-building-the-compiler)
7. [End-to-End Demo](#7-end-to-end-demo)
8. [Running Tests](#8-running-tests)
9. [Compiler Pipeline](#9-compiler-pipeline)
10. [ISA Reference](#10-isa-reference)

---

## 1. Architecture

```
  ONNX model
      │
      ▼
 xtpu-import  ──►  TOSA MLIR
      │
      ▼
  xtpu-opt   ──►  xTPU dialect MLIR  (TOSA→Linalg→xTPU passes)
      │
      ▼
 xtpu-translate ──►  .xbin binary
      │
      ▼
  xbin_runner  ──►  xTPU Simulator
      │
      ▼
   Output (uint8 tensor)
```

### Hardware Engines

| Engine | Function |
|--------|----------|
| **sDMA** | System Memory (LPDDR5) ↔ Scratchpad (1 MB on-chip) |
| **iDMA** | Scratchpad ↔ PU Local Memory (64 KB, double-buffered) |
| **PU0 / PU1** | Compute: MATMUL, ADD, MUL, SUB, RELU, MAX, REDUCE\_SUM, REDUCE\_MAX |

All engines execute in parallel; a bitmask scoreboard (`sync_mask`) enforces ordering.

---

## 2. Repository Layout

```
xTPU_simulator/
├── include/            # Simulator C++ headers
│   ├── commands.hpp    # ComputeType enum, VLIWPacket struct
│   ├── engines.hpp     # SDMAEngine, IDMAEngine, ComputeEngine
│   ├── simulator.hpp   # Top-level Simulator class
│   ├── xbin_loader.hpp # .xbin binary format loader
│   └── ...
├── src/                # Simulator implementation
├── tests/              # 27 simulator unit tests
├── tools/
│   └── xbin_runner.cpp # CLI: execute .xbin on simulator
├── compiler/
│   ├── include/xtpu/IR/     # MLIR dialect (XTPUOps.td, XTPUEnums.td)
│   ├── lib/Transforms/      # LinalgToXTPU lowering pass
│   ├── tools/
│   │   ├── xtpu-import/     # ONNX → TOSA frontend (Python)
│   │   ├── xtpu-translate/  # xTPU MLIR → .xbin (Python)
│   │   └── xtpu-dump/       # .xbin visualizer (Python)
│   └── tests/               # ONNX correctness framework
├── examples/
│   └── e2e_matmul/    # End-to-end demo script
├── submodule/
│   └── lpddr5-sim/    # LPDDR5 functional simulator (git submodule)
├── Makefile            # Simulator build
└── revision_note.md   # Change log
```

---

## 3. Requirements

### Simulator

| Dependency | Version |
|-----------|---------|
| C++ compiler | C++17 (`g++` or `clang++`) |
| `cmake` | ≥ 3.16 |
| `make` | any |
| `pthread` | system library |

### Compiler Toolchain

| Dependency | Version / Notes |
|-----------|----------------|
| LLVM / MLIR | 18 or 19 (with `mlir-opt`, `FileCheck`) |
| Python | ≥ 3.10 |
| `onnx` | `pip install onnx` |
| `numpy` | `pip install numpy` |
| `onnxruntime` | `pip install onnxruntime` (optional, for ORT golden) |

---

## 4. Quick Start

```bash
# 1. Clone with submodules
git clone --recurse-submodules https://github.com/egan-wu/xTPU_simulator.git
cd xTPU_simulator

# 2. Build the simulator and xbin_runner
make xbin_runner

# 3. Build the MLIR compiler
cd compiler && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target xtpu-opt --parallel
cd ../..

# 4. Run the end-to-end demo
bash examples/e2e_matmul/run_demo.sh
```

---

## 5. Building the Simulator

The `Makefile` in the project root handles everything, including building the
`lpddr5-sim` submodule automatically.

```bash
# Build test_simulator (27 unit tests)
make

# Build xbin_runner (execute .xbin files)
make xbin_runner

# Run all 27 unit tests
./test_simulator

# Debug build (no optimisation, AddressSanitizer)
make asan
```

> **If you cloned without `--recurse-submodules`**, initialise the submodule first:
> ```bash
> git submodule update --init --recursive
> ```

---

## 6. Building the Compiler

The compiler uses CMake + MLIR/LLVM.

```bash
cd compiler
mkdir build && cd build

# Point CMake at your LLVM installation if not in PATH
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir

# Build xtpu-opt (the main MLIR tool)
cmake --build . --target xtpu-opt --parallel

# Verify
./tools/xtpu-opt/xtpu-opt --version
```

The Python tools (`xtpu-import`, `xtpu-translate`, `xtpu-dump`) require no
separate build step — they run directly with Python 3.

---

## 7. End-to-End Demo

```bash
bash examples/e2e_matmul/run_demo.sh
```

This script:
1. Generates a 4×4 INT8 MatMul ONNX model
2. Compiles: ONNX → TOSA → Linalg → xTPU MLIR → `.xbin`
3. Runs on the xTPU simulator via `xbin_runner`
4. Verifies bit-exact output against a numpy golden reference

Expected final output:
```
✅  PASS — bit-exact match with numpy golden reference
```

---

## 8. Running Tests

### Simulator unit tests (27 tests)

```bash
make
./test_simulator
# Expected: ALL TESTS SUCCESSFUL
```

### ONNX correctness framework (12 models)

```bash
# Generate test models first (one-time)
python3 compiler/tests/gen_test_models.py

# Build compiler + xbin_runner
cd compiler/build && cmake --build . --target xtpu-opt && cd ../..
make xbin_runner

# Run correctness tests
python3 compiler/tests/test_correctness.py

# Run a single model verbosely
python3 compiler/tests/test_correctness.py --model single_matmul_i8 --verbose
```

Current results (6/12 bit-exact pass):

| Model | Status | Notes |
|-------|--------|-------|
| single_matmul_i8 | ✅ PASS | |
| two_layer_mlp_i8 | ✅ PASS | |
| gemm_mlp_i8 | ✅ PASS | |
| deep_mlp_i8 | ✅ PASS | |
| residual_block_i8 | ✅ PASS | |
| bottleneck_block_i8 | ✅ PASS | |
| transformer_attention_i8 | ❌ | Transpose lowered as identity (MVP) |
| multi_head_attention_i8 | ❌ | Transpose lowered as identity (MVP) |
| gpt_block_i8 | ❌ | Transpose + multi-path (MVP) |
| encoder_decoder_i8 | ❌ | Transpose across blocks (MVP) |
| dual_path_network_i8 | ❌ | Dual-path data flow (MVP) |
| mlp_mixer_block_i8 | ❌ | Transpose for mixer pattern (MVP) |

---

## 9. Compiler Pipeline

```
ONNX model
    │  python3 compiler/tools/xtpu-import/xtpu_import.py model.onnx
    ▼
TOSA MLIR
    │  compiler/build/tools/xtpu-opt/xtpu-opt \
    │      --tosa-to-linalg-pipeline --linalg-to-xtpu
    ▼
xTPU dialect MLIR  (+ xtpu.rodata module attribute for constants)
    │  python3 compiler/tools/xtpu-translate/xtpu_translate.py -o out.xbin
    ▼
.xbin binary
    │  ./xbin_runner out.xbin --input in.bin --output result.bin ...
    ▼
uint8 output tensor
```

### Analysing a .xbin

```bash
# Gantt chart of VLIW engine utilisation
python3 compiler/tools/xtpu-dump/xtpu_dump.py out.xbin --schedule

# Memory access map
python3 compiler/tools/xtpu-dump/xtpu_dump.py out.xbin --memory

# Hazard / sync analysis
python3 compiler/tools/xtpu-dump/xtpu_dump.py out.xbin --hazards

# All stats
python3 compiler/tools/xtpu-dump/xtpu_dump.py out.xbin --stats
```

---

## 10. ISA Reference

### ComputeType (xTPU compute operations)

| Value | Name | Operands | Semantics |
|-------|------|----------|-----------|
| 0 | NOP | — | No operation |
| 1 | MATMUL | src | 4×4 uint8 matrix multiply, uint32 accum → uint8 |
| 2 | VECTOR | src | `dst[i] = src[i] × src[i]` (square) |
| 3 | SCALAR | src | `dst[i] = src[i] + 1` |
| 4 | ADD | src, src2 | `dst[i] = (src[i] + src2[i]) & 0xFF` |
| 5 | MUL | src, src2 | `dst[i] = (src[i] × src2[i]) & 0xFF` |
| 6 | SUB | src, src2 | `dst[i] = (src[i] − src2[i]) & 0xFF` |
| 7 | RELU | src | `dst[i] = max(0, (int8_t)src[i])` |
| 8 | MAX | src, src2 | `dst[i] = max(src[i], src2[i])` |
| 9 | REDUCE\_SUM | src | `dst[0] = Σ src[i]` |
| 10 | REDUCE\_MAX | src | `dst[0] = max(src[i])` |

### .xbin Binary Layout

```
Header (32 bytes):  magic "XTPU", version, num_sections, entry_offset, flags
Section Table:      8 bytes per entry (type, flags, offset)
.text section:      num_packets (uint32) + array of 140-byte VLIWPackets
.rodata section:    constant weight tensors (auto-loaded by xbin_runner)
.meta section:      JSON tensor metadata
```

Each **VLIWPacket** (140 bytes):

| Field | Size | Description |
|-------|------|-------------|
| sDMA command | 40 B | System DMA |
| iDMA command | 40 B | Internal DMA |
| PU0 compute command | 28 B | Processing Unit 0 (includes src2\_offset) |
| PU1 compute command | 28 B | Processing Unit 1 (includes src2\_offset) |
| sync\_mask | 4 B | Bitmask of engines to wait on before dispatch |

---

## Submodule

`submodule/lpddr5-sim` is a standalone LPDDR5 functional simulator:

- **Repository**: https://github.com/egan-wu/lpddr5-sim
- **Description**: Cycle-accurate bank/rank FSM, open/closed-page scheduling,
  per-bank refresh, multi-channel, power states
- **Integration**: `LPDDR5Adapter` bridges it to xTPU's `IMemoryPort` interface
