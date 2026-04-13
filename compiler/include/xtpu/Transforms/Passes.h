//===----------------------------------------------------------------------===//
// xTPU Transformation Passes — Registration and Declarations
//===----------------------------------------------------------------------===//

#ifndef XTPU_TRANSFORMS_PASSES_H
#define XTPU_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace xtpu {

#define GEN_PASS_DECL
#include "xtpu/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "xtpu/Transforms/Passes.h.inc"

} // namespace xtpu
} // namespace mlir

#endif // XTPU_TRANSFORMS_PASSES_H
