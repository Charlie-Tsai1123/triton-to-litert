#map = affine_map<(d0) -> (d0)>
module {
  func.func @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to memref<*xf32>
    %1 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to memref<*xf32>
    %2 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
    %3 = arith.muli %arg7, %c1024_i32 : i32
    %4 = tensor.empty() : tensor<1024xi32>
    %5 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%4 : tensor<1024xi32>) {
    ^bb0(%out: i32):
      %20 = linalg.index 0 : index
      %21 = arith.index_cast %20 : index to i32
      linalg.yield %21 : i32
    } -> tensor<1024xi32>
    %6 = tensor.empty() : tensor<1024xi32>
    %7 = linalg.fill ins(%3 : i32) outs(%6 : tensor<1024xi32>) -> tensor<1024xi32>
    %8 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%7, %5 : tensor<1024xi32>, tensor<1024xi32>) outs(%7 : tensor<1024xi32>) {
    ^bb0(%in: i32, %in_2: i32, %out: i32):
      %20 = arith.addi %in, %in_2 : i32
      linalg.yield %20 : i32
    } -> tensor<1024xi32>
    %9 = tensor.empty() : tensor<1024xi32>
    %10 = linalg.fill ins(%arg3 : i32) outs(%9 : tensor<1024xi32>) -> tensor<1024xi32>
    %11 = tensor.empty() : tensor<1024xi1>
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %10 : tensor<1024xi32>, tensor<1024xi32>) outs(%11 : tensor<1024xi1>) {
    ^bb0(%in: i32, %in_2: i32, %out: i1):
      %20 = arith.cmpi slt, %in, %in_2 : i32
      linalg.yield %20 : i1
    } -> tensor<1024xi1>
    %cast = memref.cast %2 : memref<*xf32> to memref<?xf32>
    %13 = bufferization.to_tensor %cast restrict : memref<?xf32> to tensor<?xf32>
    %14 = tensor.empty() : tensor<1024xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %12 : tensor<1024xi32>, tensor<1024xi1>) outs(%14 : tensor<1024xf32>) {
    ^bb0(%in: i32, %in_2: i1, %out: f32):
      %20 = scf.if %in_2 -> (f32) {
        %21 = arith.index_cast %in : i32 to index
        %extracted = tensor.extract %13[%21] : tensor<?xf32>
        scf.yield %extracted : f32
      } else {
        scf.yield %cst : f32
      }
      linalg.yield %20 : f32
    } -> tensor<1024xf32>
    %cast_0 = memref.cast %1 : memref<*xf32> to memref<?xf32>
    %16 = bufferization.to_tensor %cast_0 restrict : memref<?xf32> to tensor<?xf32>
    %17 = tensor.empty() : tensor<1024xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %12 : tensor<1024xi32>, tensor<1024xi1>) outs(%17 : tensor<1024xf32>) {
    ^bb0(%in: i32, %in_2: i1, %out: f32):
      %20 = scf.if %in_2 -> (f32) {
        %21 = arith.index_cast %in : i32 to index
        %extracted = tensor.extract %16[%21] : tensor<?xf32>
        scf.yield %extracted : f32
      } else {
        scf.yield %cst : f32
      }
      linalg.yield %20 : f32
    } -> tensor<1024xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%15, %18 : tensor<1024xf32>, tensor<1024xf32>) outs(%15 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %20 = arith.addf %in, %in_2 : f32
      linalg.yield %20 : f32
    } -> tensor<1024xf32>
    %cast_1 = memref.cast %0 : memref<*xf32> to memref<?xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %19, %12 : tensor<1024xi32>, tensor<1024xf32>, tensor<1024xi1>) {
    ^bb0(%in: i32, %in_2: f32, %in_3: i1):
      scf.if %in_3 {
        %20 = arith.index_cast %in : i32 to index
        memref.store %in_2, %cast_1[%20] : memref<?xf32>
      }
      linalg.yield
    }
    return
  }
}

