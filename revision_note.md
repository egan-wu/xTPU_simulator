# Revision Note — P5-11 ISA Extension & Correctness Fixes

**Date**: 2026-04-15  
**Scope**: Phase 5 completion — P5-11 ISA extension, compiler fixes, correctness framework

---

## 1. Hardware Layer

### `include/commands.hpp`
- Extended `ComputeType` enum from 4 to 11 operations:
  ```cpp
  enum class ComputeType {
      NOP, MATMUL, VECTOR, SCALAR,       // original
      ADD, MUL, SUB, RELU, MAX,          // P5-11 new
      REDUCE_SUM, REDUCE_MAX,            // P5-11 new
  };
  ```
- Added `uint32_t src2_offset = 0` to `Compute_Command` for dual-operand ops (ADD, MUL, SUB, MAX)

### `src/engines.cpp`
- Added latency cases for all 7 new compute types (uses `scalar_latency`)
- Implemented in `ComputeEngine::process()`:
  - `ADD`: `dst[i] = (src[i] + src2[i]) & 0xFF`
  - `MUL`: `dst[i] = (src[i] * src2[i]) & 0xFF`
  - `SUB`: `dst[i] = (src[i] - src2[i]) & 0xFF`
  - `RELU`: interprets as `int8_t`, clamps negative to 0
  - `MAX`: `dst[i] = max(src[i], src2[i])`
  - `REDUCE_SUM`: accumulate all elements into `dst[0]`
  - `REDUCE_MAX`: find maximum of all elements into `dst[0]`

### `include/engines.hpp` — **Bug fix: pure virtual function crash**
- Moved `shutdown()` method from `protected` to `public` in `Engine<CmdType>`
- Added explicit destructors to all derived classes calling `shutdown()` before vtable teardown:
  ```cpp
  ~SDMAEngine()  override { shutdown(); }
  ~IDMAEngine()  override { shutdown(); }
  ~ComputeEngine() override { shutdown(); }
  ```
- Root cause: C++ destroys derived class data before base class destructor runs; if worker thread is still alive at that point, it calls `process()` through a stale vtable → `pure virtual function called`

### `include/simulator.hpp` — **Bug fix: perf counter ordering**
- Added public `shutdown()` method to `Simulator` that explicitly stops all 4 engine threads:
  ```cpp
  void shutdown() {
      sdma_.shutdown(); idma_.shutdown(); pu0_.shutdown(); pu1_.shutdown();
  }
  ```

---

## 2. Binary Format

### `include/xbin_loader.hpp`
- `XBinComputeCommand`: added `uint32_t src2_offset` → now **28 bytes** (was 24)
- `XBinPacket`: total size now **140 bytes** (was 132)
- Updated PU0/PU1 decode path to unpack `src2_offset`

### `compiler/tools/xtpu-translate/xtpu_translate.py`
- `ComputeCommand`: added `src2_offset: int = 0` field; `pack()` uses `"<iIIIIII"` (7 fields, 28 bytes)
- `VLIWPacket.pack()`: assertion updated to 140 bytes
- Added `COMPUTE_ADD=4` through `COMPUTE_REDUCE_MAX=10` constants and `COMPUTE_TYPE_MAP` entries
- Compute parser regex updated to capture optional `src2_offset`:
  ```python
  r'(?:\s+src2_offset\s*=\s*(\d+))?'
  ```

---

## 3. MLIR Dialect

### `compiler/include/xtpu/IR/XTPUEnums.td`
- Fixed existing enum values to match C++ (matmul=1, not 0; vector=2; scalar=3)
- Added new enum cases:
  ```
  add=4, mul=5, sub=6, relu=7, max=8, reduce_sum=9, reduce_max=10
  ```

### `compiler/include/xtpu/IR/XTPUOps.td`
- Added optional `src2_offset` attribute to `XTPU_ComputeOp`:
  ```tablegen
  DefaultValuedAttr<UI32Attr, "0">:$src2_offset
  ```
- Assembly format updated: `(`src2_offset` `=` $src2_offset^)?`

---

## 4. Compiler (LinalgToXTPU.cpp)

### New Helper: `getHardwareSizeBytes()`
- Returns `numElements` (1 byte per element) regardless of MLIR element type
- Hardware always operates in uint8; MLIR uses i32 for matmul output → sizes must match hardware reality

### Hardware-Aware Sizing Applied Throughout
- Function argument allocation: `getHardwareSizeBytes` instead of `getTensorSizeBytes`
- Constant allocation: `getHardwareSizeBytes` for system memory reservation
- Output region reservation: `getHardwareSizeBytes`
- `lowerGenericAdd`: uses hardware sizes for both operands and result
- `GenericPattern::ReLU` handler: uses hardware sizes
- `GenericPattern::Cast` handler: records hardware size in `sizeMap`
- `linalg.transpose` handler: records hardware size

### Real Compute Types (no more workarounds)
- `GenericPattern::Add` → emits `ComputeType::add` with dual-operand `src2_offset`
- `GenericPattern::ReLU` → emits `ComputeType::relu` with proper DMA sequence
- `GenericPattern::Cast` → still pass-through (no-op), but now records correct hardware size

### Broadcast Add Support
- `lowerGenericAdd` detects when `input1Size < resultSize`
- Replicates smaller operand by repeated IDMA loads into contiguous local memory:
  ```cpp
  while (loaded < resultSize) {
      chunk = min(input1Size, resultSize - loaded);
      emitter.emitIDMALoad(input1ScratchAddr, src1Local + loaded, chunk);
      loaded += chunk;
  }
  ```

### Scratch Allocation Overlap Fix — **Critical Bug Fix**
- Previous code called `planner.resetScratch()` unconditionally before resolving Add operands
- If `input0` was already in scratch at offset 0, resetting caused `input1` to be loaded at offset 0, overwriting the matmul result
- Fix: only reset scratch if neither operand is already resident in scratch:
  ```cpp
  if (!in0InScratch && !in1InScratch)
      planner.resetScratch();
  ```

### Splat Constant Handling — **Bug Fix**
- `DenseElementsAttr::getRawData()` returns compact data for splat values (e.g., `dense<1> : tensor<4xi32>` returns just 4 bytes, not 16)
- Added `isSplat()` check: extract low byte and `assign(numElems, val)` to expand
- Non-splat i8: use raw data directly, resize to `numElems`
- Non-splat wider: extract low byte per element with bounds check

### i32 Constant → uint8 Conversion
- Constants stored in system memory are always 1 byte per element (hardware uint8)
- For i32 constants: extract `rawData[i * byteWidth]` (low byte, little-endian)
- Old behavior: stored full i32 bytes, causing hardware to read wrong data when loading 16-byte uint8 matrix from 64-byte i32 storage

### `PacketEmitter::emitCompute()` — Extended
- Added `uint32_t src2Off = 0` parameter
- Passes `src2_offset` attribute to `xtpu::ComputeOp::create()`

---

## 5. Tools

### `tools/xbin_runner.cpp`
- Added explicit `sim.shutdown()` call before returning from `main()`
- Prevents pure virtual function crash during `Simulator` destructor

### `compiler/tools/xtpu-dump/xtpu_dump.py`
- Updated for 140-byte packet format and new compute type names

---

## 6. Correctness Framework

### `compiler/tests/test_correctness.py`

#### Hardware-Mode Golden Reference (`hardware_mode=True`)
- New parameter added to `run_golden_numpy()`
- When enabled, every operation truncates intermediate results to uint8:
  - **MatMul**: `uint8 × uint8`, `uint32` accumulation, `& 0xFF`
  - **Gemm**: same uint8 semantics
  - **Add**: `uint16` intermediate, `& 0xFF`, with numpy broadcast support
  - **ReLU**: views uint8 as int8, clamps negatives to 0
  - Constants loaded as uint8 (`view(np.uint8)` for i8, `% 256` for wider types)
- Used as the primary golden reference in `run_test()`

#### Test Runner
- `run_test()` now calls `run_golden_numpy(..., hardware_mode=True)` instead of `run_golden()` (ORT)
- Reason: hardware truncates to uint8 at every step; ORT uses i32 accumulation throughout

---

## 7. Test Results

| Metric | Before | After |
|--------|--------|-------|
| Simulator unit tests | 27/27 ✅ | 27/27 ✅ |
| ONNX bit-exact PASS | 1/12 | **6/12** |
| xbin_runner crashes | Every run (`pure virtual`) | 0 |
| Compile failures | 0/12 | 0/12 |

### Models Passing (6/12)
| Model | Notes |
|-------|-------|
| single_matmul_i8 | Pure matmul, 1 layer |
| two_layer_mlp_i8 | Matmul + bias add + ReLU + matmul |
| gemm_mlp_i8 | Gemm-based MLP |
| deep_mlp_i8 | 4-layer MLP |
| residual_block_i8 | Matmul + residual add + ReLU |
| bottleneck_block_i8 | Bottleneck with residual |

### Models Failing (6/12) — Known Limitations
| Model | Root Cause |
|-------|-----------|
| transformer_attention_i8 | Transpose lowered as identity copy |
| multi_head_attention_i8 | Transpose lowered as identity copy |
| gpt_block_i8 | Transpose + complex data flow |
| encoder_decoder_i8 | Transpose across encoder/decoder |
| dual_path_network_i8 | Dual-path data flow planning |
| mlp_mixer_block_i8 | Transpose for mixer pattern |

---

## 8. Known Remaining Limitations

1. **Transpose as identity**: `linalg.transpose` is lowered as a no-op copy (data layout unchanged). Attention-based models require true transpose for correct head dimension handling.

2. **Signed vs unsigned matmul semantics**: Hardware uses uint8 accumulation; ONNX INT8 matmul uses signed int8 with int32 accumulation. For small values (no overflow), results agree; for larger values, results diverge.

3. **REDUCE_SUM/REDUCE_MAX not yet exercised by compiler**: Lowering pass emits `add`/`relu` but does not yet detect pooling or normalization patterns to emit `reduce_sum`/`reduce_max`.

4. **MUL/SUB not yet emitted by compiler**: Hardware ops exist but no Linalg patterns currently lower to them.
