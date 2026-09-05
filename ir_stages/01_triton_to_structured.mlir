module attributes {triton_to_litert.buffer_roles = ["input", "input", "output"], triton_to_litert.buffers_distinct, triton_to_litert.launch_grid = array<i64: 1, 1, 1>} {
  tt.func public @vector_add(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = arith.index_cast %1 : i32 to index
    %5 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %6 = tt.splat %1 : i32 -> tensor<1024xi32>
    %7 = arith.addi %6, %5 : tensor<1024xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%4], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %10 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %11 = "tts.load"(%9) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %13 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%3], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %14 = tt.addptr %12, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %15 = "tts.load"(%13) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %16 = arith.addf %11, %15 : tensor<1024xf32>
    %17 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %18 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [%2], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %17, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    "tts.store"(%18, %16) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    tt.return
  }
}

