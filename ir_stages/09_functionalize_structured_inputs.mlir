#map = affine_map<(d0) -> (d0)>
module attributes {triton_to_litert.buffer_roles = ["input", "input", "output"], triton_to_litert.buffers_distinct} {
  func.func @vector_add(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: !tt.ptr<f32>) {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32>) outs(%arg0 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<1024xf32>
    %1 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%1, %0) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

