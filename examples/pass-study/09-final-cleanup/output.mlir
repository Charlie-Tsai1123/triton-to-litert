#map = affine_map<(d0) -> (d0)>
module {
  func.func @add_kernel_01234(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = arith.muli %arg7, %c1024_i32 : i32
    %1 = tensor.empty() : tensor<1024xi32>
    %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%1 : tensor<1024xi32>) {
    ^bb0(%out: i32):
      %14 = linalg.index 0 : index
      %15 = arith.index_cast %14 : index to i32
      linalg.yield %15 : i32
    } -> tensor<1024xi32>
    %3 = linalg.fill ins(%0 : i32) outs(%1 : tensor<1024xi32>) -> tensor<1024xi32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %2 : tensor<1024xi32>, tensor<1024xi32>) outs(%3 : tensor<1024xi32>) {
    ^bb0(%in: i32, %in_2: i32, %out: i32):
      %14 = arith.addi %in, %in_2 : i32
      linalg.yield %14 : i32
    } -> tensor<1024xi32>
    %5 = linalg.fill ins(%arg3 : i32) outs(%1 : tensor<1024xi32>) -> tensor<1024xi32>
    %6 = tensor.empty() : tensor<1024xi1>
    %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%4, %5 : tensor<1024xi32>, tensor<1024xi32>) outs(%6 : tensor<1024xi1>) {
    ^bb0(%in: i32, %in_2: i32, %out: i1):
      %14 = arith.cmpi slt, %in, %in_2 : i32
      linalg.yield %14 : i1
    } -> tensor<1024xi1>
    %cast = memref.cast %arg0 : memref<*xf32> to memref<?xf32>
    %8 = bufferization.to_tensor %cast restrict : memref<?xf32> to tensor<?xf32>
    %9 = tensor.empty() : tensor<1024xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%4, %7 : tensor<1024xi32>, tensor<1024xi1>) outs(%9 : tensor<1024xf32>) {
    ^bb0(%in: i32, %in_2: i1, %out: f32):
      %14 = scf.if %in_2 -> (f32) {
        %15 = arith.index_cast %in : i32 to index
        %extracted = tensor.extract %8[%15] : tensor<?xf32>
        scf.yield %extracted : f32
      } else {
        scf.yield %cst : f32
      }
      linalg.yield %14 : f32
    } -> tensor<1024xf32>
    %cast_0 = memref.cast %arg1 : memref<*xf32> to memref<?xf32>
    %11 = bufferization.to_tensor %cast_0 restrict : memref<?xf32> to tensor<?xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%4, %7 : tensor<1024xi32>, tensor<1024xi1>) outs(%9 : tensor<1024xf32>) {
    ^bb0(%in: i32, %in_2: i1, %out: f32):
      %14 = scf.if %in_2 -> (f32) {
        %15 = arith.index_cast %in : i32 to index
        %extracted = tensor.extract %11[%15] : tensor<?xf32>
        scf.yield %extracted : f32
      } else {
        scf.yield %cst : f32
      }
      linalg.yield %14 : f32
    } -> tensor<1024xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%10, %12 : tensor<1024xf32>, tensor<1024xf32>) outs(%10 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %14 = arith.addf %in, %in_2 : f32
      linalg.yield %14 : f32
    } -> tensor<1024xf32>
    %cast_1 = memref.cast %arg2 : memref<*xf32> to memref<?xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%4, %13, %7 : tensor<1024xi32>, tensor<1024xf32>, tensor<1024xi1>) {
    ^bb0(%in: i32, %in_2: f32, %in_3: i1):
      scf.if %in_3 {
        %14 = arith.index_cast %in : i32 to index
        memref.store %in_2, %cast_1[%14] : memref<?xf32>
      }
      linalg.yield
    }
    return
  }
}

