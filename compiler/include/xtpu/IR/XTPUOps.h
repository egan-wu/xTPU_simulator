#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "xtpu/IR/XTPUDialect.h"
#include "xtpu/IR/XTPUEnums.h"

// TableGen-generated op declarations
#define GET_OP_CLASSES
#include "xtpu/IR/XTPUOps.h.inc"
