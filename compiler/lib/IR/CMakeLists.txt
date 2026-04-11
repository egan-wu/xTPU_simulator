add_mlir_dialect_library(XTPUDialect
  XTPUDialect.cpp
  XTPUOps.cpp

  DEPENDS
  XTPUIncGen
)
