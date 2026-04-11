//===----------------------------------------------------------------------===//
// xtpu-opt — MLIR optimizer driver for the xTPU dialect
//
// Registers the xtpu dialect and delegates to MlirOptMain.
// Usage:
//   xtpu-opt input.mlir              # parse + verify
//   xtpu-opt input.mlir --verify     # explicit verify-only
//   xtpu-opt input.mlir -o out.mlir  # round-trip
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "xtpu/IR/XTPUDialect.h"
#include "xtpu/IR/XTPUOps.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::xtpu::XTPUDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "xTPU MLIR Optimizer\n", registry));
}
