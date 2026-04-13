//===----------------------------------------------------------------------===//
// xtpu-opt — MLIR optimizer driver for the xTPU dialect
//
// Registers the xtpu dialect and all upstream dialects needed in the
// compilation pipeline (TOSA, Linalg, Arith, Tensor, Func, etc.),
// plus all standard conversion passes (--tosa-to-linalg-pipeline etc.).
//
// Usage:
//   xtpu-opt input.mlir                           # parse + verify
//   xtpu-opt input.mlir --tosa-to-linalg-pipeline # TOSA → Linalg
//   xtpu-opt input.mlir -o out.mlir               # round-trip
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "xtpu/IR/XTPUDialect.h"
#include "xtpu/IR/XTPUOps.h"

int main(int argc, char **argv) {
  // Register all upstream dialects and passes so we can use
  // --tosa-to-linalg-pipeline, --linalg-bufferize, etc.
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::xtpu::XTPUDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "xTPU MLIR Optimizer\n", registry));
}
