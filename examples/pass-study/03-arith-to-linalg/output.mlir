#map = affine_map<(d0) -> (d0)>
module {
  func.func @kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: tensor<1024x!tt.ptr<f32>>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %0 = tensor.empty() : tensor<1024xi32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<1024xi32>) {
    ^bb0(%out: i32):
      %13 = linalg.index 0 : index
      %14 = arith.index_cast %13 : index to i32
      linalg.yield %14 : i32
    } -> tensor<1024xi32>
    %splat = tensor.splat %arg0 : tensor<1024x!tt.ptr<f32>>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%splat, %1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) outs(%splat : tensor<1024x!tt.ptr<f32>>) {
    ^bb0(%in: !tt.ptr<f32>, %in_1: i32, %out: !tt.ptr<f32>):
      %13 = tt.addptr %in, %in_1 : !tt.ptr<f32>, i32
      linalg.yield %13 : !tt.ptr<f32>
    } -> tensor<1024x!tt.ptr<f32>>
    %splat_0 = tensor.splat %arg1 : tensor<1024x!tt.ptr<f32>>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%splat_0, %1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) outs(%splat_0 : tensor<1024x!tt.ptr<f32>>) {
    ^bb0(%in: !tt.ptr<f32>, %in_1: i32, %out: !tt.ptr<f32>):
      %13 = tt.addptr %in, %in_1 : !tt.ptr<f32>, i32
      linalg.yield %13 : !tt.ptr<f32>
    } -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.load %2 : tensor<1024x!tt.ptr<f32>>
    %5 = tt.load %3 : tensor<1024x!tt.ptr<f32>>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%4, %5 : tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %13 = arith.addf %in, %in_1 : f32
      linalg.yield %13 : f32
    } -> tensor<1024xf32>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%6, %5 : tensor<1024xf32>, tensor<1024xf32>) outs(%6 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %13 = arith.subf %in, %in_1 : f32
      linalg.yield %13 : f32
    } -> tensor<1024xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%7, %5 : tensor<1024xf32>, tensor<1024xf32>) outs(%7 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %13 = arith.mulf %in, %in_1 : f32
      linalg.yield %13 : f32
    } -> tensor<1024xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %5 : tensor<1024xf32>, tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %13 = arith.divf %in, %in_1 : f32
      linalg.yield %13 : f32
    } -> tensor<1024xf32>
    %10 = tensor.empty() : tensor<1024xi1>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%9, %5 : tensor<1024xf32>, tensor<1024xf32>) outs(%10 : tensor<1024xi1>) {
    ^bb0(%in: f32, %in_1: f32, %out: i1):
      %13 = arith.cmpf oeq, %in, %in_1 : f32
      linalg.yield %13 : i1
    } -> tensor<1024xi1>
    %12 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins(%11, %4, %5 : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs(%4 : tensor<1024xf32>) {
    ^bb0(%in: i1, %in_1: f32, %in_2: f32, %out: f32):
      %13 = arith.select %in, %in_1, %in_2 : f32
      linalg.yield %13 : f32
    } -> tensor<1024xf32>
    tt.store %arg2, %12 : tensor<1024x!tt.ptr<f32>>
    return
  }
}

