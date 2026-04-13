// P5-3 test: Single matmul lowering (Linalg → xTPU)
//
// Input: linalg.batch_matmul on 1x4x4 i8 tensors
// Expected: xtpu.program with SDMA→IDMA→compute→IDMA→SDMA→drain
//
// RUN: xtpu-opt --linalg-to-xtpu %s | xtpu-opt --verify

module {
  func.func @main(%arg0: tensor<1x4x4xi8>) -> tensor<1x4x4xi32> {
    %0 = "tosa.const"() <{values = dense<[[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]> : tensor<1x4x4xi8>}> : () -> tensor<1x4x4xi8>
    %c0_i32 = arith.constant 0 : i32
    %1 = tensor.empty() : tensor<1x4x4xi32>
    %2 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<1x4x4xi32>) -> tensor<1x4x4xi32>
    %3 = linalg.batch_matmul ins(%arg0, %0 : tensor<1x4x4xi8>, tensor<1x4x4xi8>) outs(%2 : tensor<1x4x4xi32>) -> tensor<1x4x4xi32>
    return %3 : tensor<1x4x4xi32>
  }
}
