//===----------------------------------------------------------------------===//
// xTPU Enum Attribute Definitions
//
// These enums map 1:1 to the simulator's C++ enums in commands.hpp
// and common_types.hpp.
//===----------------------------------------------------------------------===//

#ifndef XTPU_ENUMS_TD
#define XTPU_ENUMS_TD

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// DMADirection — maps to DMADirection in commands.hpp
//===----------------------------------------------------------------------===//
def XTPU_DMADir_ToDevice   : I32EnumAttrCase<"to_device",   0>;
def XTPU_DMADir_FromDevice : I32EnumAttrCase<"from_device", 1>;

def XTPU_DMADirection : I32EnumAttr<
    "DMADirection",
    "DMA transfer direction (to_device = load, from_device = store)",
    [XTPU_DMADir_ToDevice, XTPU_DMADir_FromDevice]> {
  let cppNamespace = "::mlir::xtpu";
}

//===----------------------------------------------------------------------===//
// ComputeType — maps to ComputeType in commands.hpp
//===----------------------------------------------------------------------===//
def XTPU_Compute_Matmul : I32EnumAttrCase<"matmul", 0>;
def XTPU_Compute_Vector : I32EnumAttrCase<"vector", 1>;
def XTPU_Compute_Scalar : I32EnumAttrCase<"scalar", 2>;

def XTPU_ComputeType : I32EnumAttr<
    "ComputeType",
    "Compute operation type",
    [XTPU_Compute_Matmul, XTPU_Compute_Vector, XTPU_Compute_Scalar]> {
  let cppNamespace = "::mlir::xtpu";
}

//===----------------------------------------------------------------------===//
// Target — maps to TARGET_PU0 / TARGET_PU1 bitmask in common_types.hpp
//===----------------------------------------------------------------------===//
def XTPU_Target_PU0  : I32EnumAttrCase<"pu0",  1>;
def XTPU_Target_PU1  : I32EnumAttrCase<"pu1",  2>;
def XTPU_Target_PU01 : I32EnumAttrCase<"pu01", 3>;

def XTPU_Target : I32EnumAttr<
    "Target",
    "IDMA target PU mask (pu01 = broadcast to both)",
    [XTPU_Target_PU0, XTPU_Target_PU1, XTPU_Target_PU01]> {
  let cppNamespace = "::mlir::xtpu";
}

#endif // XTPU_ENUMS_TD
