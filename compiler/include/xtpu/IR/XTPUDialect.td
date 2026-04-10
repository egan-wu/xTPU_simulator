//===----------------------------------------------------------------------===//
// xTPU Dialect Definition
//===----------------------------------------------------------------------===//

#ifndef XTPU_DIALECT_TD
#define XTPU_DIALECT_TD

include "mlir/IR/OpBase.td"

def XTPU_Dialect : Dialect {
  let name = "xtpu";
  let summary = "xTPU low-level VLIW dialect for AI accelerator simulation";
  let description = [{
    The `xtpu` dialect represents the lowest MLIR abstraction for the xTPU
    accelerator simulator. It maps 1:1 to the simulator's VLIWPacket structure,
    enabling offline compilation of AI models into deterministic instruction
    streams following the LPU-inspired static scheduling philosophy.

    Design principles:
    - All data movement is explicit (no implicit caches or prefetch)
    - All memory addresses are static (resolved at compile time)
    - All latencies are deterministic (compiler and simulator share TimingConfig)
    - Execution is a linear stream of VLIWPackets (no branches, no OoO)
  }];
  let cppNamespace = "::mlir::xtpu";
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
}

// Base class for all xTPU ops
class XTPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<XTPU_Dialect, mnemonic, traits>;

#endif // XTPU_DIALECT_TD
