#map = affine_map<(d0) -> (d0)>
module attributes {triton_to_litert.buffer_roles = ["input", "input", "output"], triton_to_litert.buffers_distinct, triton_to_litert.launch_grid = array<i64: 1, 1, 1>} {
  func.func @vector_add(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = arith.muli %arg6, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %2 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%1], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %3 = "tts.load"(%2) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %4 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%1], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %5 = "tts.load"(%4) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%3, %5 : tensor<1024xf32>, tensor<1024xf32>) outs(%3 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %8 = arith.addf %in, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<1024xf32>
    %7 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [%1], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%7, %6) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

