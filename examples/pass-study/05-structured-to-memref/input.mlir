#map = affine_map<(d0) -> (d0)>
module {
  func.func @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %0 = arith.muli %arg7, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%1], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %3 = arith.addi %1, %c1024 : index
    %4 = arith.index_cast %arg3 : i32 to index
    %5 = arith.minsi %3, %4 : index
    %6 = arith.maxsi %5, %1 : index
    %7 = arith.subi %6, %1 : index
    %8 = "tts.load"(%2, %7) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, index) -> tensor<1024xf32>
    %9 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%1], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %10 = "tts.load"(%9, %7) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, index) -> tensor<1024xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %10 : tensor<1024xf32>, tensor<1024xf32>) outs(%8 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %13 = arith.addf %in, %in_0 : f32
      linalg.yield %13 : f32
    } -> tensor<1024xf32>
    %12 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [%1], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%12, %11, %7) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>, index) -> ()
    return
  }
}

