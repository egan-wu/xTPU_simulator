get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_llvm_executable(xtpu-opt
  xtpu-opt.cpp
)

target_link_libraries(xtpu-opt
  PRIVATE
  XTPUDialect
  ${dialect_libs}
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRSupport
  MLIRIR
)

llvm_update_compile_flags(xtpu-opt)
