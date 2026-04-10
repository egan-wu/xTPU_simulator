# xTPU MLIR Dialect Specification

> **Version**: 0.1.0 (Draft)
> **Date**: 2026-04-10
> **Status**: P5-1 Initial Definition
> **Author**: xTPU Simulator Team

---

## 1. Overview

The `xtpu` dialect is the **lowest-level MLIR abstraction** in the xTPU compiler
stack, sitting directly above the binary encoding layer (`.xbin`). Every op in
this dialect has a 1:1 correspondence to a field in the simulator's
`VLIWPacket` / `DMA_Command` / `Compute_Command` C++ structs.

**Design philosophy** (LPU-inspired):
- All data movement is explicit — no implicit caches, no runtime prefetch.
- All memory addresses are static — resolved at compile time.
- All latencies are deterministic — the compiler and simulator share the same `TimingConfig`.
- The execution model is a linear stream of `VLIWPacket`s — no branches, no OoO.

**Dialect namespace**: `xtpu`

---

## 2. Hardware Resource Model

The dialect's type system and op semantics are derived from the simulator's
physical resources:

| Resource | Size | Dialect Symbol | Notes |
|----------|------|----------------|-------|
| System Memory (LPDDR5) | 16 MB (configurable) | `@sys_mem` | Accessed via SDMA only |
| Scratchpad (on-chip SRAM) | 1 MB | `@scratchpad` | Shared by SDMA and IDMA |
| LocalMemory PU0 | 64 KB × 2 buffers | `@local_mem0` | Private to PU0, double-buffered |
| LocalMemory PU1 | 64 KB × 2 buffers | `@local_mem1` | Private to PU1, double-buffered |

**Engine slots** (one per VLIWPacket, at most one active op each):

| Slot | Engine | Command Type | Status Bit |
|------|--------|-------------|------------|
| 0 | SDMA | `DMA_Command` | `STATUS_SDMA_BUSY` (bit 0) |
| 1 | IDMA | `DMA_Command` | `STATUS_PU0_DMA_BUSY` (bit 1) / `STATUS_PU1_DMA_BUSY` (bit 3) |
| 2 | PU0 | `Compute_Command` | `STATUS_PU0_CMD_BUSY` (bit 2) |
| 3 | PU1 | `Compute_Command` | `STATUS_PU1_CMD_BUSY` (bit 4) |

**Sync mask bits** (bitmask, OR-combined):

| Bit | Hex | Constant | Meaning |
|-----|-----|----------|---------|
| 0 | 0x01 | `sdma` | Wait for SDMA completion |
| 1 | 0x02 | `pu0_dma` | Wait for IDMA→PU0 completion |
| 2 | 0x04 | `pu0_cmd` | Wait for PU0 compute completion |
| 3 | 0x08 | `pu1_dma` | Wait for IDMA→PU1 completion |
| 4 | 0x10 | `pu1_cmd` | Wait for PU1 compute completion |

---

## 3. Type System

The `xtpu` dialect operates at the **physical address level** — it does not use
MLIR's `memref` or `tensor` types. All operands are integer attributes
representing byte addresses and sizes.

### 3.1 Attribute Types

| Attribute | MLIR Type | Description |
|-----------|-----------|-------------|
| `addr` | `ui64` | Byte address within a memory space |
| `size` | `ui64` | Transfer or operation size in bytes |
| `offset` | `ui32` | Byte offset within a LocalMemory buffer |
| `length` | `ui32` | Operation length in bytes |
| `buffer_idx` | `I32Attr` | LocalMemory buffer index: 0 or 1 |
| `target` | `TargetAttr` | IDMA target: `pu0`, `pu1`, or `pu01` (broadcast) |
| `sync_mask` | `SyncMaskAttr` | Bitmask of engine status bits to wait on |

### 3.2 Enum Attributes

```tablegen
def XTPU_DMADirection : I32EnumAttr<"DMADirection",
    "DMA transfer direction", [
      I32EnumAttrCase<"to_device",   0>,   // Load:  outer → inner
      I32EnumAttrCase<"from_device", 1>,   // Store: inner → outer
    ]>;

def XTPU_ComputeType : I32EnumAttr<"ComputeType",
    "Compute operation type", [
      I32EnumAttrCase<"matmul", 0>,
      I32EnumAttrCase<"vector", 1>,
      I32EnumAttrCase<"scalar", 2>,
    ]>;

def XTPU_Target : I32EnumAttr<"Target",
    "IDMA target PU mask", [
      I32EnumAttrCase<"pu0",  1>,   // TARGET_PU0
      I32EnumAttrCase<"pu1",  2>,   // TARGET_PU1
      I32EnumAttrCase<"pu01", 3>,   // Broadcast: TARGET_PU0 | TARGET_PU1
    ]>;
```

---

## 4. Operations

### 4.1 `xtpu.program` — Top-Level Container

The root operation. Contains an ordered sequence of `xtpu.packet` ops.
Represents a complete compiled program that can be serialized to `.xbin`.

**Syntax**:
```mlir
xtpu.program @name {
  // ... sequence of xtpu.packet ops ...
}
```

**Semantics**:
- The body region contains a single block with an ordered sequence of packets.
- Packets are dispatched sequentially by the simulator's `dispatch_packet()`.
- The program name becomes the symbol in the `.xbin` binary.

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `sym_name` | `StrAttr` | Yes | Program symbol name |

**Constraints**:
- Body must contain only `xtpu.packet` ops.
- At least one `xtpu.packet` must be present.

---

### 4.2 `xtpu.packet` — VLIW Packet Container

Maps 1:1 to `VLIWPacket`. Contains up to 4 engine ops plus an optional sync
barrier. All ops within a packet are dispatched simultaneously.

**Syntax**:
```mlir
xtpu.packet sync_mask = ["sdma", "pu0_cmd"] {
  xtpu.sdma ...
  xtpu.idma ...
  xtpu.compute ...
  xtpu.compute ...
}
```

**Semantics**:
1. If `sync_mask` is non-empty, the simulator calls `status_reg_.wait_on_mask(mask)`
   **before** dispatching any op in this packet.
2. All contained ops are dispatched to their respective engines concurrently.
3. Each engine slot may appear **at most once**:
   - At most 1 `xtpu.sdma`
   - At most 1 `xtpu.idma`
   - At most 1 `xtpu.compute` with `pu = 0`
   - At most 1 `xtpu.compute` with `pu = 1`

**Attributes**:

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `sync_mask` | `ArrayAttr` of `StringAttr` | `[]` | Engine status bits to wait on before dispatch |

`sync_mask` string values map to C++ constants:

| String | C++ Constant | Hex |
|--------|-------------|-----|
| `"sdma"` | `STATUS_SDMA_BUSY` | `0x01` |
| `"pu0_dma"` | `STATUS_PU0_DMA_BUSY` | `0x02` |
| `"pu0_cmd"` | `STATUS_PU0_CMD_BUSY` | `0x04` |
| `"pu1_dma"` | `STATUS_PU1_DMA_BUSY` | `0x08` |
| `"pu1_cmd"` | `STATUS_PU1_CMD_BUSY` | `0x10` |

**Verification rules**:
- V1: No duplicate engine slots (e.g., two `xtpu.sdma` in one packet is illegal).
- V2: `sync_mask` entries must be valid enum strings.
- V3: `sync_mask` should only reference engines that have been dispatched in prior packets and not yet synced (warning, not error — conservative sync is safe).

---

### 4.3 `xtpu.sdma` — System DMA Operation

Maps 1:1 to `VLIWPacket::sDMA_op` (`DMA_Command`).

Transfers data between System Memory and Scratchpad.

**Syntax**:
```mlir
// Load: System Memory → Scratchpad
xtpu.sdma load src_addr = 0x0 dst_addr = 0x1000 size = 4096

// Store: Scratchpad → System Memory
xtpu.sdma store src_addr = 0x1000 dst_addr = 0x0 size = 4096
```

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `direction` | `DMADirection` | Yes | `load` = TO_DEVICE, `store` = FROM_DEVICE |
| `src_addr` | `UI64Attr` | Yes | Source byte address |
| `dst_addr` | `UI64Attr` | Yes | Destination byte address |
| `size` | `UI64Attr` | Yes | Transfer size in bytes |

**Direction semantics**:

| Direction | Keyword | src_addr space | dst_addr space |
|-----------|---------|---------------|---------------|
| `to_device` | `load` | System Memory | Scratchpad |
| `from_device` | `store` | Scratchpad | System Memory |

**Verification rules**:
- V4: `size` > 0.
- V5: `load` path: `src_addr + size` ≤ `SYSTEM_MEMORY_SIZE` (16 MB), `dst_addr + size` ≤ `SCRATCHPAD_SIZE` (1 MB).
- V6: `store` path: `src_addr + size` ≤ `SCRATCHPAD_SIZE`, `dst_addr + size` ≤ `SYSTEM_MEMORY_SIZE`.

---

### 4.4 `xtpu.idma` — Internal DMA Operation

Maps 1:1 to `VLIWPacket::iDMA_op` (`DMA_Command`).

Transfers data between Scratchpad and LocalMemory. Supports broadcast to both PUs.

**Syntax**:
```mlir
// Load: Scratchpad → PU0 LocalMem buffer 0
xtpu.idma load src_addr = 0x0 dst_addr = 0x0 size = 64 target = pu0 buffer = 0

// Broadcast: Scratchpad → Both PUs, buffer 1
xtpu.idma load src_addr = 0x0 dst_addr = 0x0 size = 64 target = pu01 buffer = 1

// Writeback: PU0 LocalMem buffer 0 → Scratchpad
xtpu.idma store src_addr = 0x0 dst_addr = 0x400 size = 64 target = pu0 buffer = 0
```

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `direction` | `DMADirection` | Yes | `load` = TO_DEVICE, `store` = FROM_DEVICE |
| `src_addr` | `UI64Attr` | Yes | Source byte address |
| `dst_addr` | `UI64Attr` | Yes | Destination byte address |
| `size` | `UI64Attr` | Yes | Transfer size in bytes |
| `target` | `Target` | Yes | Target PU(s): `pu0`, `pu1`, or `pu01` |
| `buffer` | `I32Attr` | Yes | LocalMemory buffer index: 0 or 1 |

**Direction semantics**:

| Direction | Keyword | src_addr space | dst_addr space |
|-----------|---------|---------------|---------------|
| `to_device` | `load` | Scratchpad | LocalMemory[target][buffer] |
| `from_device` | `store` | LocalMemory[target][buffer] | Scratchpad |

**Verification rules**:
- V7: `size` > 0.
- V8: `buffer` ∈ {0, 1}.
- V9: `load` path: `src_addr + size` ≤ `SCRATCHPAD_SIZE`, `dst_addr + size` ≤ `LOCAL_MEM_SIZE` (64 KB).
- V10: `store` path: `src_addr + size` ≤ `LOCAL_MEM_SIZE`, `dst_addr + size` ≤ `SCRATCHPAD_SIZE`.
- V11: Broadcast (`pu01`) is only valid for `load` direction. Broadcast `store` is illegal (ambiguous source).

---

### 4.5 `xtpu.compute` — Processing Unit Operation

Maps 1:1 to `VLIWPacket::pu0_op` or `VLIWPacket::pu1_op` (`Compute_Command`).

Executes a computation on the specified PU's LocalMemory.

**Syntax**:
```mlir
// Scalar: dst[i] = src[i] + 1
xtpu.compute pu = 0 type = scalar buffer = 0 src_offset = 0 dst_offset = 64 length = 16

// Vector: dst[i] = src[i] * src[i]
xtpu.compute pu = 1 type = vector buffer = 1 src_offset = 0 dst_offset = 128 length = 32

// MatMul: C = A × B (4×4 uint8_t)
// A at src_offset, B at src_offset + 16, C written to dst_offset
xtpu.compute pu = 0 type = matmul buffer = 0 src_offset = 0 dst_offset = 32 length = 16
```

**Attributes**:

| Attribute | Type | Required | Description |
|-----------|------|----------|-------------|
| `pu` | `I32Attr` | Yes | Processing unit index: 0 or 1 |
| `type` | `ComputeType` | Yes | Operation type: `matmul`, `vector`, `scalar` |
| `buffer` | `I32Attr` | Yes | LocalMemory buffer index: 0 or 1 |
| `src_offset` | `UI32Attr` | Yes | Source data offset in LocalMemory buffer |
| `dst_offset` | `UI32Attr` | Yes | Destination offset in LocalMemory buffer |
| `length` | `UI32Attr` | Yes | Operation length in bytes |

**Operation semantics** (all operations on `uint8_t`, natural truncation):

| Type | Read Region | Write Region | Computation |
|------|-------------|--------------|-------------|
| `scalar` | `[src_offset, src_offset + length)` | `[dst_offset, dst_offset + length)` | `dst[i] = src[i] + 1` |
| `vector` | `[src_offset, src_offset + length)` | `[dst_offset, dst_offset + length)` | `dst[i] = src[i] * src[i]` |
| `matmul` | A: `[src_offset, src_offset + 16)`, B: `[src_offset + 16, src_offset + 32)` | `[dst_offset, dst_offset + 16)` | `C = A × B` (4×4, uint32 accum, truncate to uint8) |

**Verification rules**:
- V12: `pu` ∈ {0, 1}.
- V13: `buffer` ∈ {0, 1}.
- V14: `length` > 0.
- V15: For `matmul`: `length` must equal 16 (4×4 matrix).
- V16: For `matmul`: `src_offset + 32` ≤ `LOCAL_MEM_SIZE` (reads 2 matrices).
- V17: For `scalar`/`vector`: `src_offset + length` ≤ `LOCAL_MEM_SIZE` and `dst_offset + length` ≤ `LOCAL_MEM_SIZE`.
- V18: `dst_offset + length` ≤ `LOCAL_MEM_SIZE`.
- V19: Two `xtpu.compute` ops in the same packet must have different `pu` values.

---

## 5. Lowering to VLIWPacket (Binary Encoding Contract)

The `xtpu` dialect is the **last stop before binary emission**. The mapping to
C++ structs is mechanical:

### 5.1 `xtpu.packet` → `VLIWPacket`

```
VLIWPacket {
    .sync_mask = OR of sync_mask attribute bits
    .sDMA_op   = from contained xtpu.sdma   (or DMAType::NOP if absent)
    .iDMA_op   = from contained xtpu.idma   (or DMAType::NOP if absent)
    .pu0_op    = from contained xtpu.compute where pu=0 (or ComputeType::NOP)
    .pu1_op    = from contained xtpu.compute where pu=1 (or ComputeType::NOP)
}
```

### 5.2 `xtpu.sdma` → `DMA_Command`

```
DMA_Command {
    .type       = DMAType::MEMCPY
    .src_addr   = src_addr attribute
    .dst_addr   = dst_addr attribute
    .size       = size attribute
    .target_mask = 0             // unused for SDMA
    .buffer_idx  = 0             // unused for SDMA
    .direction   = direction attribute → DMADirection enum
}
```

### 5.3 `xtpu.idma` → `DMA_Command`

```
DMA_Command {
    .type        = DMAType::MEMCPY
    .src_addr    = src_addr attribute
    .dst_addr    = dst_addr attribute
    .size        = size attribute
    .target_mask = target attribute → TARGET_PU0/PU1/both
    .buffer_idx  = buffer attribute
    .direction   = direction attribute → DMADirection enum
}
```

### 5.4 `xtpu.compute` → `Compute_Command`

```
Compute_Command {
    .type                  = type attribute → ComputeType enum
    .buffer_idx            = buffer attribute
    .simulated_duration_ms = 0        // deprecated, always 0 for compiled code
    .src_offset            = src_offset attribute
    .dst_offset            = dst_offset attribute
    .length                = length attribute
}
```

---

## 6. Verification Pass (`xtpu-verify`)

The dialect verifier enforces all V-rules from Section 4. The verification is
split into two categories:

### 6.1 Structural Verification (always enforced)

| Rule | Check | Severity |
|------|-------|----------|
| V1 | No duplicate engine slots per packet | Error |
| V2 | Valid sync_mask strings | Error |
| V4, V7, V14 | Size/length > 0 | Error |
| V8, V13 | Buffer index ∈ {0, 1} | Error |
| V12 | PU index ∈ {0, 1} | Error |
| V15 | MATMUL length == 16 | Error |
| V19 | Unique PU per packet | Error |

### 6.2 Address Bounds Verification (opt-in, requires hardware config)

| Rule | Check | Severity |
|------|-------|----------|
| V5, V6 | SDMA address bounds | Error |
| V9, V10 | IDMA address bounds | Error |
| V11 | No broadcast store | Error |
| V16, V17, V18 | Compute offset bounds | Error |

### 6.3 Hazard Analysis (warning-level, informational)

| Check | Description | Severity |
|-------|-------------|----------|
| V3 | sync_mask references un-dispatched engine | Warning |
| RAW | Read-after-write on same memory region without sync | Warning |
| WAW | Write-after-write on same memory region without sync | Warning |
| Redundant sync | sync_mask includes engine that has already been synced | Warning |

---

## 7. Timing Model Integration

The compiler uses the same `TimingConfig` as the simulator to estimate latencies
at compile time. This enables the VLIW scheduler (P5-5) to make optimal packing
decisions.

### 7.1 Latency Table (from `TimingConfig` defaults)

| Operation | Latency Formula | Default Value |
|-----------|----------------|---------------|
| SDMA transfer | `⌈size / 64⌉ × sdma_latency_per_cacheline` | 10 ticks/CL |
| IDMA transfer | `⌈size / 64⌉ × idma_latency_per_cacheline` | 5 ticks/CL |
| MATMUL | `matmul_latency` (fixed per tile) | 100 ticks |
| VECTOR | `vector_latency` (fixed per op) | 20 ticks |
| SCALAR | `scalar_latency` (fixed per op) | 5 ticks |
| LPDDR5 SDMA | Cycle-accurate model (replaces fixed SDMA latency) | ~44 ticks/CL (cold) |

### 7.2 Compiler-Simulator Contract

The compiler and simulator **must share the same TimingConfig instance** for
schedule correctness. This is enforced by:

1. The `.xbin` header embeds the `TimingConfig` used during compilation.
2. The simulator's `XBinLoader` (P5-7) validates that embedded config matches
   the runtime `TimingConfig`, or emits a warning on mismatch.
3. `validate_clock_consistency()` (P3-CR-6) ensures `xtpu_tck_ps × ms_to_ticks`
   coherence at both compile time and load time.

---

## 8. Example Programs

### 8.1 Minimal: Single SDMA Load

```mlir
xtpu.program @test_sdma_load {
  // Load 64 bytes from System Memory[0x0] to Scratchpad[0x0]
  xtpu.packet {
    xtpu.sdma load src_addr = 0 dst_addr = 0 size = 64
  }
  // Wait for SDMA completion
  xtpu.packet sync_mask = ["sdma"] {
  }
}
```

### 8.2 SDMA + IDMA Pipeline

```mlir
xtpu.program @test_sdma_idma {
  // Packet 0: Load from LPDDR5 to Scratchpad
  xtpu.packet {
    xtpu.sdma load src_addr = 0 dst_addr = 0 size = 256
  }
  // Packet 1: Sync SDMA, then IDMA to PU0 LocalMem
  xtpu.packet sync_mask = ["sdma"] {
    xtpu.idma load src_addr = 0 dst_addr = 0 size = 64 target = pu0 buffer = 0
  }
  // Packet 2: Sync IDMA, done
  xtpu.packet sync_mask = ["pu0_dma"] {
  }
}
```

### 8.3 Full Pipeline: Load → Compute → Writeback

```mlir
xtpu.program @matmul_e2e {
  // 1. SDMA: LPDDR5[0x0000..0x003F] → Scratchpad[0x0000]
  xtpu.packet {
    xtpu.sdma load src_addr = 0 dst_addr = 0 size = 64
  }

  // 2. Sync SDMA, then IDMA: Scratchpad → PU0 LM buf0[0..31] (A + B matrices)
  xtpu.packet sync_mask = ["sdma"] {
    xtpu.idma load src_addr = 0 dst_addr = 0 size = 32 target = pu0 buffer = 0
  }

  // 3. Sync IDMA, then PU0 MATMUL: A@0 × B@16 → C@32
  xtpu.packet sync_mask = ["pu0_dma"] {
    xtpu.compute pu = 0 type = matmul buffer = 0 src_offset = 0 dst_offset = 32 length = 16
  }

  // 4. Sync PU0, then IDMA writeback: PU0 LM buf0[32..47] → Scratchpad[0x0000]
  xtpu.packet sync_mask = ["pu0_cmd"] {
    xtpu.idma store src_addr = 32 dst_addr = 0 size = 16 target = pu0 buffer = 0
  }

  // 5. Sync IDMA, then SDMA writeback: Scratchpad[0x0000..0x000F] → LPDDR5[0x1000]
  xtpu.packet sync_mask = ["pu0_dma"] {
    xtpu.sdma store src_addr = 0 dst_addr = 4096 size = 16
  }

  // 6. Final sync
  xtpu.packet sync_mask = ["sdma"] {
  }
}
```

### 8.4 Double-Buffered Compute with Overlap

```mlir
xtpu.program @double_buffer_demo {
  // === Prologue: Load first tile ===
  xtpu.packet {
    xtpu.sdma load src_addr = 0 dst_addr = 0 size = 64
  }
  xtpu.packet sync_mask = ["sdma"] {
    xtpu.idma load src_addr = 0 dst_addr = 0 size = 32 target = pu0 buffer = 0
  }

  // === Steady state: Overlap IDMA(buf1) + Compute(buf0) ===
  xtpu.packet sync_mask = ["pu0_dma"] {
    // SDMA loads next tile while PU0 computes current tile
    xtpu.sdma  load src_addr = 64 dst_addr = 64 size = 64
    xtpu.compute pu = 0 type = matmul buffer = 0 src_offset = 0 dst_offset = 32 length = 16
  }
  xtpu.packet sync_mask = ["sdma", "pu0_cmd"] {
    // IDMA loads into buf1 (PU0 was using buf0)
    xtpu.idma load src_addr = 64 dst_addr = 0 size = 32 target = pu0 buffer = 1
  }
  xtpu.packet sync_mask = ["pu0_dma"] {
    // PU0 computes on buf1, meanwhile writeback buf0 result
    xtpu.compute pu = 0 type = matmul buffer = 1 src_offset = 0 dst_offset = 32 length = 16
    xtpu.idma store src_addr = 32 dst_addr = 0 size = 16 target = pu0 buffer = 0
  }

  // === Epilogue: Final writeback ===
  xtpu.packet sync_mask = ["pu0_cmd", "pu0_dma"] {
    xtpu.idma store src_addr = 32 dst_addr = 16 size = 16 target = pu0 buffer = 1
  }
  xtpu.packet sync_mask = ["pu0_dma"] {
    xtpu.sdma store src_addr = 0 dst_addr = 8192 size = 32
  }
  xtpu.packet sync_mask = ["sdma"] {
  }
}
```

### 8.5 Dual-PU Parallel Compute with Broadcast

```mlir
xtpu.program @dual_pu_broadcast {
  // 1. Load shared weights to Scratchpad
  xtpu.packet {
    xtpu.sdma load src_addr = 0 dst_addr = 0 size = 64
  }

  // 2. Broadcast weights to both PUs (same data, same buffer)
  xtpu.packet sync_mask = ["sdma"] {
    xtpu.idma load src_addr = 0 dst_addr = 0 size = 32 target = pu01 buffer = 0
  }

  // 3. Load different activations for each PU (requires 2 separate IDMAs → 2 packets)
  //    PU0 activation from scratchpad[64..95]
  xtpu.packet sync_mask = ["pu0_dma", "pu1_dma"] {
    xtpu.sdma load src_addr = 64 dst_addr = 64 size = 64
  }
  xtpu.packet sync_mask = ["sdma"] {
    xtpu.idma load src_addr = 64 dst_addr = 32 size = 16 target = pu0 buffer = 0
  }
  xtpu.packet sync_mask = ["pu0_dma"] {
    xtpu.idma load src_addr = 80 dst_addr = 32 size = 16 target = pu1 buffer = 0
  }

  // 4. Both PUs compute in parallel
  xtpu.packet sync_mask = ["pu1_dma"] {
    xtpu.compute pu = 0 type = matmul buffer = 0 src_offset = 0 dst_offset = 48 length = 16
    xtpu.compute pu = 1 type = matmul buffer = 0 src_offset = 0 dst_offset = 48 length = 16
  }

  // 5. Writeback results
  xtpu.packet sync_mask = ["pu0_cmd", "pu1_cmd"] {
    xtpu.idma store src_addr = 48 dst_addr = 0 size = 16 target = pu0 buffer = 0
  }
  xtpu.packet sync_mask = ["pu0_dma"] {
    xtpu.idma store src_addr = 48 dst_addr = 16 size = 16 target = pu1 buffer = 0
  }
  xtpu.packet sync_mask = ["pu1_dma"] {
    xtpu.sdma store src_addr = 0 dst_addr = 4096 size = 32
  }
  xtpu.packet sync_mask = ["sdma"] {
  }
}
```

---

## 9. ODS (TableGen) Reference

The following is the canonical ODS definition for the dialect. Implementation
files will be auto-generated from this definition.

### 9.1 Dialect Definition

```tablegen
def XTPU_Dialect : Dialect {
  let name = "xtpu";
  let summary = "xTPU low-level VLIW dialect for AI accelerator simulation";
  let description = [{
    The `xtpu` dialect represents the lowest MLIR abstraction for the xTPU
    accelerator simulator. It maps 1:1 to the simulator's VLIWPacket structure,
    enabling offline compilation of AI models into deterministic instruction
    streams following the LPU-inspired static scheduling philosophy.
  }];
  let cppNamespace = "::mlir::xtpu";
}
```

### 9.2 Op Definitions

```tablegen
//===----------------------------------------------------------------------===//
// xtpu.program
//===----------------------------------------------------------------------===//
def XTPU_ProgramOp : XTPU_Op<"program", [
    IsolatedFromAbove, SymbolTable, Symbol,
    SingleBlockImplicitTerminator<"EndOp">
]> {
  let summary = "Top-level xTPU program container";
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name $body attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// xtpu.packet
//===----------------------------------------------------------------------===//
def XTPU_PacketOp : XTPU_Op<"packet", [
    SingleBlockImplicitTerminator<"YieldOp">
]> {
  let summary = "VLIW packet — dispatched atomically to all engines";
  let arguments = (ins
    DefaultValuedAttr<StrArrayAttr, "{}">:$sync_mask
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "(`sync_mask` `=` $sync_mask^)? $body attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// xtpu.sdma
//===----------------------------------------------------------------------===//
def XTPU_SDMAOp : XTPU_Op<"sdma"> {
  let summary = "System DMA: System Memory <-> Scratchpad";
  let arguments = (ins
    XTPU_DMADirection:$direction,
    UI64Attr:$src_addr,
    UI64Attr:$dst_addr,
    UI64Attr:$size
  );
  let assemblyFormat = [{
    custom<DMADirection>($direction)
    `src_addr` `=` $src_addr
    `dst_addr` `=` $dst_addr
    `size` `=` $size
    attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// xtpu.idma
//===----------------------------------------------------------------------===//
def XTPU_IDMAOp : XTPU_Op<"idma"> {
  let summary = "Internal DMA: Scratchpad <-> LocalMemory (with broadcast)";
  let arguments = (ins
    XTPU_DMADirection:$direction,
    UI64Attr:$src_addr,
    UI64Attr:$dst_addr,
    UI64Attr:$size,
    XTPU_Target:$target,
    I32Attr:$buffer
  );
  let assemblyFormat = [{
    custom<DMADirection>($direction)
    `src_addr` `=` $src_addr
    `dst_addr` `=` $dst_addr
    `size` `=` $size
    `target` `=` $target
    `buffer` `=` $buffer
    attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// xtpu.compute
//===----------------------------------------------------------------------===//
def XTPU_ComputeOp : XTPU_Op<"compute"> {
  let summary = "Compute operation on PU LocalMemory";
  let arguments = (ins
    I32Attr:$pu,
    XTPU_ComputeType:$type,
    I32Attr:$buffer,
    UI32Attr:$src_offset,
    UI32Attr:$dst_offset,
    UI32Attr:$length
  );
  let assemblyFormat = [{
    `pu` `=` $pu
    `type` `=` $type
    `buffer` `=` $buffer
    `src_offset` `=` $src_offset
    `dst_offset` `=` $dst_offset
    `length` `=` $length
    attr-dict
  }];
  let hasVerifier = 1;
}
```

---

## 10. Future Extensions (Post-MVP)

These extensions are **not part of P5-1** but are designed to be addable without
breaking the existing op set:

| Extension | Description | When |
|-----------|-------------|------|
| **`xtpu.barrier`** | Explicit named barrier (beyond sync_mask) for complex DAGs | P5-5+ |
| **`xtpu.dma_chain`** | Linked DMA descriptors for scatter/gather patterns | When hardware supports it |
| **`xtpu.conv2d`** | Native 2D convolution op (beyond 4×4 matmul tiling) | When ComputeEngine is extended |
| **`xtpu.activation`** | ReLU, GELU, etc. as native ops | When ComputeEngine supports them |
| **`xtpu.quantize`** / `xtpu.dequantize` | INT8 ↔ FP32 conversion ops | For mixed-precision support |
| **Configurable tile size** | Parameterize MATMUL beyond 4×4 | When hardware resources grow |
| **Multi-channel LPDDR5** | Per-channel address mapping attributes | When multi-channel is exposed |

---

## Appendix A: Memory Map Reference

```
┌─────────────────────────────────────────────────┐
│  System Memory (LPDDR5 / SimpleRAM)             │
│  16 MB  [0x0000_0000 .. 0x00FF_FFFF]           │
│  Accessed by: SDMA only                         │
└──────────────────────┬──────────────────────────┘
                       │ SDMA (load/store)
┌──────────────────────▼──────────────────────────┐
│  Scratchpad (on-chip SRAM)                      │
│  1 MB   [0x0000_0000 .. 0x000F_FFFF]           │
│  Accessed by: SDMA, IDMA                        │
└──────────┬──────────────────────┬───────────────┘
           │ IDMA (load/store)    │ IDMA (load/store)
┌──────────▼──────────┐ ┌────────▼────────────────┐
│  LocalMem PU0       │ │  LocalMem PU1           │
│  Buffer 0: 64 KB    │ │  Buffer 0: 64 KB        │
│  Buffer 1: 64 KB    │ │  Buffer 1: 64 KB        │
│  [0x0000 .. 0xFFFF] │ │  [0x0000 .. 0xFFFF]     │
│  Compute: PU0 only  │ │  Compute: PU1 only      │
└─────────────────────┘ └─────────────────────────┘
```

Note: Each memory space has its own address space starting at 0. The `xtpu`
dialect does **not** use a unified address space — the memory space is implicit
in the op type (sdma → sys_mem/scratchpad, idma → scratchpad/local_mem,
compute → local_mem).

---

## Appendix B: Sync Mask Quick Reference

| Mnemonic | Bit | Hex | Wait for... |
|----------|-----|-----|-------------|
| `sdma` | 0 | `0x01` | SDMA engine idle |
| `pu0_dma` | 1 | `0x02` | IDMA→PU0 transfer done |
| `pu0_cmd` | 2 | `0x04` | PU0 compute done |
| `pu1_dma` | 3 | `0x08` | IDMA→PU1 transfer done |
| `pu1_cmd` | 4 | `0x10` | PU1 compute done |

Common combinations:
- `["sdma"]` — wait for system memory transfer
- `["pu0_dma", "pu1_dma"]` — wait for broadcast IDMA to both PUs
- `["pu0_cmd", "pu1_cmd"]` — wait for both PUs to finish computing
- `["sdma", "pu0_dma", "pu0_cmd", "pu1_dma", "pu1_cmd"]` — full drain (all engines idle)
