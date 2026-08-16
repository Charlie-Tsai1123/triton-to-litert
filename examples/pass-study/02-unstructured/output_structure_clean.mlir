module {
  tt.func public @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = arith.index_cast %1 : i32 to index
    %5 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%4], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %6 = arith.index_cast %1 : i32 to index
    %7 = arith.addi %6, %c1024 : index
    %8 = arith.index_cast %arg3 : i32 to index
    %9 = arith.minsi %7, %8 : index
    %10 = arith.maxsi %9, %6 : index
    %11 = arith.subi %10, %6 : index
    %12 = "tts.load"(%5, %11) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, index) -> tensor<1024xf32>
    %13 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%3], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %14 = arith.index_cast %1 : i32 to index
    %15 = arith.addi %14, %c1024 : index
    %16 = arith.index_cast %arg3 : i32 to index
    %17 = arith.minsi %15, %16 : index
    %18 = arith.maxsi %17, %14 : index
    %19 = arith.subi %18, %14 : index
    %20 = "tts.load"(%13, %19) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, index) -> tensor<1024xf32>
    %21 = arith.addf %12, %20 : tensor<1024xf32>
    %22 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [%2], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %23 = arith.index_cast %1 : i32 to index
    %24 = arith.addi %23, %c1024 : index
    %25 = arith.index_cast %arg3 : i32 to index
    %26 = arith.minsi %24, %25 : index
    %27 = arith.maxsi %26, %23 : index
    %28 = arith.subi %27, %23 : index
    "tts.store"(%22, %21, %28) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>, index) -> ()
    tt.return
  }
}

