//===----------------------------------------------------------------------===//
// xTPU Dialect Registration
//===----------------------------------------------------------------------===//

#include "xtpu/IR/XTPUDialect.h"
#include "xtpu/IR/XTPUOps.h"

using namespace mlir;
using namespace mlir::xtpu;

// TableGen-generated dialect definition
#include "xtpu/IR/XTPUDialect.cpp.inc"

void XTPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xtpu/IR/XTPUOps.cpp.inc"
      >();
}
