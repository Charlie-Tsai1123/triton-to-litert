module attributes {triton_to_litert.buffer_roles = ["input", "input", "output"], triton_to_litert.buffers_distinct, triton_to_litert.launch_grid = array<i64: 1, 1, 1>} {
  tt.func public @vector_add(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%2], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %4 = "tts.load"(%3) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %5 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%2], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %6 = "tts.load"(%5) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %7 = arith.addf %4, %6 : tensor<1024xf32>
    %8 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [%2], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%8, %7) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    tt.return
  }
}

