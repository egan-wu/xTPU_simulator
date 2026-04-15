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
// Values must match C++ enum: NOP=0, MATMUL=1, VECTOR=2, SCALAR=3, ADD=4, ...
//===----------------------------------------------------------------------===//
def XTPU_Compute_Matmul     : I32EnumAttrCase<"matmul",     1>;
def XTPU_Compute_Vector     : I32EnumAttrCase<"vector",     2>;
def XTPU_Compute_Scalar     : I32EnumAttrCase<"scalar",     3>;
// P5-11: Extended ISA ops
def XTPU_Compute_Add        : I32EnumAttrCase<"add",        4>;
def XTPU_Compute_Mul        : I32EnumAttrCase<"mul",        5>;
def XTPU_Compute_Sub        : I32EnumAttrCase<"sub",        6>;
def XTPU_Compute_Relu       : I32EnumAttrCase<"relu",       7>;
def XTPU_Compute_Max        : I32EnumAttrCase<"max",        8>;
def XTPU_Compute_ReduceSum  : I32EnumAttrCase<"reduce_sum", 9>;
def XTPU_Compute_ReduceMax  : I32EnumAttrCase<"reduce_max", 10>;

def XTPU_ComputeType : I32EnumAttr<
    "ComputeType",
    "Compute operation type",
    [XTPU_Compute_Matmul, XTPU_Compute_Vector, XTPU_Compute_Scalar,
     XTPU_Compute_Add, XTPU_Compute_Mul, XTPU_Compute_Sub,
     XTPU_Compute_Relu, XTPU_Compute_Max,
     XTPU_Compute_ReduceSum, XTPU_Compute_ReduceMax]> {
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
