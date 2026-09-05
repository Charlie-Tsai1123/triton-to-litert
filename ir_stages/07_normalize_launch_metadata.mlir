#map = affine_map<(d0) -> (d0)>
module attributes {triton_to_litert.buffer_roles = ["input", "input", "output"], triton_to_litert.buffers_distinct} {
  func.func @vector_add(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %0 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %2 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %3 = "tts.load"(%2) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%1, %3 : tensor<1024xf32>, tensor<1024xf32>) outs(%1 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %6 = arith.addf %in, %in_0 : f32
      linalg.yield %6 : f32
    } -> tensor<1024xf32>
    %5 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%5, %4) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

