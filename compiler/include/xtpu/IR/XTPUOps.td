//===----------------------------------------------------------------------===//
// xTPU Operation Definitions
//
// Each op maps 1:1 to a field in the simulator's VLIWPacket struct.
// See docs/DialectSpec.md for full specification and verification rules.
//===----------------------------------------------------------------------===//

#ifndef XTPU_OPS_TD
#define XTPU_OPS_TD

include "xtpu/IR/XTPUDialect.td"
include "xtpu/IR/XTPUEnums.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// xtpu.end — implicit terminator for program and packet regions
//===----------------------------------------------------------------------===//
def XTPU_EndOp : XTPU_Op<"end", [Terminator]> {
  let summary = "Implicit terminator for xtpu regions";
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// xtpu.program — top-level container
//
// Contains an ordered sequence of xtpu.packet ops representing a complete
// compiled program. Serialized to .xbin binary by xtpu-translate.
//
// Verification:
//   - Body must contain only xtpu.packet ops (+ terminator)
//   - At least one xtpu.packet must be present
//===----------------------------------------------------------------------===//
def XTPU_ProgramOp : XTPU_Op<"program", [
    IsolatedFromAbove,
    Symbol,
    SingleBlockImplicitTerminator<"EndOp">
]> {
  let summary = "Top-level xTPU program container";
  let description = [{
    Represents a complete xTPU program as an ordered sequence of VLIW packets.
    The program name becomes the symbol in the .xbin binary.

    Example:
    ```mlir
    xtpu.program @my_kernel {
      xtpu.packet { ... }
      xtpu.packet sync_mask = ["sdma"] { ... }
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$sym_name $body attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// xtpu.packet — VLIW packet container
//
// Maps 1:1 to VLIWPacket. Contains up to 4 engine ops dispatched atomically.
// sync_mask specifies which engine status bits to wait on BEFORE dispatch.
//
// Verification (V-rules from DialectSpec.md):
//   V1: No duplicate engine slots
//   V2: Valid sync_mask strings
//   V19: Unique PU index per packet
//===----------------------------------------------------------------------===//
def XTPU_PacketOp : XTPU_Op<"packet", [
    SingleBlockImplicitTerminator<"EndOp">
]> {
  let summary = "VLIW packet — dispatched atomically to all engines";
  let description = [{
    A VLIW packet contains up to 4 engine operations that are dispatched
    simultaneously. The optional sync_mask causes the simulator to wait for
    the specified engine(s) to become idle before dispatching.

    sync_mask values: "sdma", "pu0_dma", "pu0_cmd", "pu1_dma", "pu1_cmd"

    Example:
    ```mlir
    xtpu.packet sync_mask = ["sdma", "pu0_cmd"] {
      xtpu.sdma load src_addr = 0 dst_addr = 0 size = 64
      xtpu.compute pu = 1 type = scalar buffer = 0
                   src_offset = 0 dst_offset = 0 length = 16
    }
    ```
  }];

  let arguments = (ins
    DefaultValuedAttr<StrArrayAttr, "{}">:$sync_mask
  );
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "(`sync_mask` `=` $sync_mask^)? $body attr-dict";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// xtpu.sdma — System DMA: System Memory <-> Scratchpad
//
// Maps 1:1 to VLIWPacket::sDMA_op (DMA_Command).
//
// Direction semantics:
//   load  (to_device):   System Memory[src] → Scratchpad[dst]
//   store (from_device): Scratchpad[src]    → System Memory[dst]
//
// Verification:
//   V4: size > 0
//   V5: load:  src_addr + size <= SYSTEM_MEMORY_SIZE, dst_addr + size <= SCRATCHPAD_SIZE
//   V6: store: src_addr + size <= SCRATCHPAD_SIZE,    dst_addr + size <= SYSTEM_MEMORY_SIZE
//===----------------------------------------------------------------------===//
def XTPU_SDMAOp : XTPU_Op<"sdma"> {
  let summary = "System DMA: System Memory <-> Scratchpad";
  let description = [{
    Transfers data between System Memory (LPDDR5) and Scratchpad (on-chip SRAM).

    `load` reads from System Memory and writes to Scratchpad.
    `store` reads from Scratchpad and writes back to System Memory.

    Example:
    ```mlir
    xtpu.sdma load  src_addr = 0x0000 dst_addr = 0x0000 size = 4096
    xtpu.sdma store src_addr = 0x0000 dst_addr = 0x1000 size = 256
    ```
  }];

  let arguments = (ins
    XTPU_DMADirection:$direction,
    UI64Attr:$src_addr,
    UI64Attr:$dst_addr,
    UI64Attr:$size
  );

  let assemblyFormat = [{
    custom<DMAKeyword>($direction)
    `src_addr` `=` $src_addr
    `dst_addr` `=` $dst_addr
    `size` `=` $size
    attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// xtpu.idma — Internal DMA: Scratchpad <-> LocalMemory
//
// Maps 1:1 to VLIWPacket::iDMA_op (DMA_Command).
// Supports broadcast (target = pu01) for load direction only.
//
// Direction semantics:
//   load  (to_device):   Scratchpad[src] → LocalMem[target][buffer][dst]
//   store (from_device): LocalMem[target][buffer][src] → Scratchpad[dst]
//
// Verification:
//   V7:  size > 0
//   V8:  buffer in {0, 1}
//   V9:  load:  src + size <= SCRATCHPAD_SIZE, dst + size <= LOCAL_MEM_SIZE
//   V10: store: src + size <= LOCAL_MEM_SIZE,  dst + size <= SCRATCHPAD_SIZE
//   V11: broadcast (pu01) is only valid for load direction
//===----------------------------------------------------------------------===//
def XTPU_IDMAOp : XTPU_Op<"idma"> {
  let summary = "Internal DMA: Scratchpad <-> LocalMemory (with broadcast)";
  let description = [{
    Transfers data between Scratchpad and a PU's LocalMemory buffer.
    Supports broadcast to both PUs simultaneously (target = pu01, load only).

    Example:
    ```mlir
    xtpu.idma load  src_addr = 0 dst_addr = 0 size = 64 target = pu0  buffer = 0
    xtpu.idma load  src_addr = 0 dst_addr = 0 size = 64 target = pu01 buffer = 1
    xtpu.idma store src_addr = 0 dst_addr = 0 size = 16 target = pu0  buffer = 0
    ```
  }];

  let arguments = (ins
    XTPU_DMADirection:$direction,
    UI64Attr:$src_addr,
    UI64Attr:$dst_addr,
    UI64Attr:$size,
    XTPU_Target:$target,
    I32Attr:$buffer
  );

  let assemblyFormat = [{
    custom<DMAKeyword>($direction)
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
// xtpu.compute — PU Compute Operation
//
// Maps 1:1 to VLIWPacket::pu0_op or pu1_op (Compute_Command).
// Operates on data in the specified PU's LocalMemory buffer.
//
// Operation semantics (uint8_t, natural truncation):
//   scalar: dst[i] = src[i] + 1
//   vector: dst[i] = src[i] * src[i]
//   matmul: C = A × B (4×4, uint32 accum → uint8 truncate)
//           A at [src_offset, src_offset+16), B at [src_offset+16, src_offset+32)
//           C at [dst_offset, dst_offset+16), length must be 16
//
// Verification:
//   V12: pu in {0, 1}
//   V13: buffer in {0, 1}
//   V14: length > 0
//   V15: matmul requires length == 16
//   V16: matmul requires src_offset + 32 <= LOCAL_MEM_SIZE
//   V17: scalar/vector requires src_offset + length <= LOCAL_MEM_SIZE
//   V18: dst_offset + length <= LOCAL_MEM_SIZE
//===----------------------------------------------------------------------===//
def XTPU_ComputeOp : XTPU_Op<"compute"> {
  let summary = "Compute operation on PU LocalMemory";
  let description = [{
    Executes a computation on the specified PU's LocalMemory buffer.

    Example:
    ```mlir
    xtpu.compute pu = 0 type = matmul buffer = 0
                 src_offset = 0 dst_offset = 32 length = 16
    xtpu.compute pu = 1 type = scalar buffer = 1
                 src_offset = 0 dst_offset = 64 length = 256
    ```
  }];

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

#endif // XTPU_OPS_TD
