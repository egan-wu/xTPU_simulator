#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, 0, d2)>
module {
  func.func @main(%arg0: tensor<1x4x4xi8>) -> tensor<1x4x4xi32> {
    %0 = "tosa.const"() <{values = dense<1> : tensor<1x1x4xi32>}> : () -> tensor<1x1x4xi32>
    %1 = "tosa.const"() <{values = dense<[[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]> : tensor<1x4x4xi32>}> : () -> tensor<1x4x4xi32>
    %2 = "tosa.const"() <{values = dense<0> : tensor<1x4x4xi32>}> : () -> tensor<1x4x4xi32>
    %3 = "tosa.const"() <{values = dense<[[[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]> : tensor<1x4x4xi8>}> : () -> tensor<1x4x4xi8>
    %c0_i32 = arith.constant 0 : i32
    %4 = tensor.empty() : tensor<1x4x4xi32>
    %5 = linalg.fill ins(%c0_i32 : i32) outs(%4 : tensor<1x4x4xi32>) -> tensor<1x4x4xi32>
    %6 = linalg.batch_matmul ins(%arg0, %3 : tensor<1x4x4xi8>, tensor<1x4x4xi8>) outs(%5 : tensor<1x4x4xi32>) -> tensor<1x4x4xi32>
    %7 = tensor.empty() : tensor<1x4x4xi32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %0 : tensor<1x4x4xi32>, tensor<1x1x4xi32>) outs(%7 : tensor<1x4x4xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %14 = arith.addi %in, %in_0 : i32
      linalg.yield %14 : i32
    } -> tensor<1x4x4xi32>
    %9 = tensor.empty() : tensor<1x4x4xi32>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %2 : tensor<1x4x4xi32>, tensor<1x4x4xi32>) outs(%9 : tensor<1x4x4xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      %14 = arith.maxsi %in, %in_0 : i32
      linalg.yield %14 : i32
    } -> tensor<1x4x4xi32>
    %11 = tensor.empty() : tensor<1x4x4xi32>
    %12 = linalg.fill ins(%c0_i32 : i32) outs(%11 : tensor<1x4x4xi32>) -> tensor<1x4x4xi32>
    %13 = linalg.batch_matmul ins(%10, %1 : tensor<1x4x4xi32>, tensor<1x4x4xi32>) outs(%12 : tensor<1x4x4xi32>) -> tensor<1x4x4xi32>
    return %13 : tensor<1x4x4xi32>
  }
}

