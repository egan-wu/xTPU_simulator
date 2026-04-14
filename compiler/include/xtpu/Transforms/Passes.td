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

def XTPUMemoryPlan : Pass<"xtpu-memory-plan", "mlir::ModuleOp"> {
  let summary = "Verify and report xTPU static memory allocation";
  let description = [{
    Analyzes an xtpu.program and verifies:
      1. No overlapping writes to the same address range in system memory,
         scratchpad, or local memory at the same time.
      2. All addresses fit within hardware bounds.
      3. Emits a compile report with memory utilization statistics.

    This pass does not modify the IR — it is a pure analysis/verification pass.
  }];
  let dependentDialects = ["mlir::xtpu::XTPUDialect"];
}

def XTPUSchedule : Pass<"xtpu-schedule", "mlir::ModuleOp"> {
  let summary = "VLIW packet scheduling — merge compatible packets";
  let description = [{
    Optimizes the xTPU packet stream by:
      1. Merging consecutive packets that use different engines
         (e.g., an SDMA-only packet followed by an IDMA-only packet
         can be merged into a single VLIW packet).
      2. Removing redundant sync barriers.
      3. Emitting engine utilization statistics.

    Correctness invariant: the merged schedule must produce the same
    memory-visible behavior as the serial schedule.
  }];
  let dependentDialects = ["mlir::xtpu::XTPUDialect"];
}

#endif // XTPU_TRANSFORMS_PASSES_TD
