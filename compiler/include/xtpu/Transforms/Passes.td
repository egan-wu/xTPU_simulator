//===----------------------------------------------------------------------===//
// xTPU Transformation Passes
//===----------------------------------------------------------------------===//

#ifndef XTPU_TRANSFORMS_PASSES_TD
#define XTPU_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LinalgToXTPU : Pass<"linalg-to-xtpu", "mlir::ModuleOp"> {
  let summary = "Lower Linalg-on-tensors to xTPU dialect";
  let description = [{
    Converts a func.func containing linalg ops (batch_matmul, generic add/relu)
    on small tensors into an xtpu.program with explicit DMA and compute packets.

    All addresses are resolved statically using a bump allocator.
    MVP constraints:
      - Only 4x4 tiles (batch=1, M=4, N=4, K=4)
      - Single PU (pu0, buffer 0)
      - Serial execution (no overlap/double-buffering)
      - Supported ops: linalg.batch_matmul, linalg.generic (add, relu-as-noop)
  }];

  let dependentDialects = [
    "mlir::xtpu::XTPUDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // XTPU_TRANSFORMS_PASSES_TD
