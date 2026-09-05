module attributes {triton_to_litert.buffer_roles = ["input", "input", "output"], triton_to_litert.buffers_distinct} {
  func.func @vector_add(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>, %arg2: !tt.ptr<f32>) {
    %0 = tensor.empty() : tensor<1024xf32>
    %1 = linalg.elementwise kind=#linalg.elementwise_kind<add> ins(%arg0, %arg1 : tensor<1024xf32>, tensor<1024xf32>) outs(%0 : tensor<1024xf32>) -> tensor<1024xf32>
    %2 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%2, %1) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

