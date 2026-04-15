# ---------------------------------------------------------------------------
# TableGen: generate .inc files from ODS .td definitions
# ---------------------------------------------------------------------------

# Tell TableGen where to find our .td includes
set(LLVM_TARGET_DEFINITIONS XTPUOps.td)

# Generate op declarations / definitions
mlir_tablegen(XTPUOps.h.inc -gen-op-decls)
mlir_tablegen(XTPUOps.cpp.inc -gen-op-defs)

# Generate dialect declarations / definitions
mlir_tablegen(XTPUDialect.h.inc -gen-dialect-decls -dialect=xtpu)
mlir_tablegen(XTPUDialect.cpp.inc -gen-dialect-defs -dialect=xtpu)

# Generate enum declarations / definitions
set(LLVM_TARGET_DEFINITIONS XTPUEnums.td)
mlir_tablegen(XTPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(XTPUEnums.cpp.inc -gen-enum-defs)

# Create a target that other targets can depend on
add_public_tablegen_target(XTPUIncGen)
