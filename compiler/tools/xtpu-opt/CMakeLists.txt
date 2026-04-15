get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(xtpu-opt
  xtpu-opt.cpp
)

target_link_libraries(xtpu-opt
  PRIVATE
  XTPUDialect
  XTPUTransforms
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRIR
)

llvm_update_compile_flags(xtpu-opt)
