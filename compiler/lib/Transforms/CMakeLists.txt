add_mlir_dialect_library(XTPUTransforms
  LinalgToXTPU.cpp
  XTPUMemoryPlan.cpp
  XTPUSchedule.cpp

  DEPENDS
  XTPUTransformsIncGen
  XTPUIncGen

  LINK_LIBS PUBLIC
  XTPUDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRPass
  MLIRIR
)
