module {
  tt.func public @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = arith.index_cast %1 : i32 to index
    %3 = arith.index_cast %1 : i32 to index
    %4 = arith.index_cast %1 : i32 to index
    %5 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %6 = tt.splat %1 : i32 -> tensor<1024xi32>
    %7 = arith.addi %6, %5 : tensor<1024xi32>
    %8 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %9 = arith.cmpi slt, %7, %8 : tensor<1024xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %11 = tts.make_tptr %arg0 to sizes: [1024], strides: [1], offsets: [%4], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %12 = tt.addptr %10, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %13 = arith.index_cast %1 : i32 to index
    %c1024 = arith.constant 1024 : index
    %14 = arith.addi %c1024, %13 : index
    %15 = arith.index_cast %arg3 : i32 to index
    %16 = arith.minsi %14, %15 : index
    %17 = arith.maxsi %16, %13 : index
    %18 = arith.subi %17, %13 : index
    %19 = "tts.load"(%11, %18) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, index) -> tensor<1024xf32>
    %20 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %21 = tts.make_tptr %arg1 to sizes: [1024], strides: [1], offsets: [%3], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %22 = tt.addptr %20, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %23 = arith.index_cast %1 : i32 to index
    %c1024_0 = arith.constant 1024 : index
    %24 = arith.addi %c1024_0, %23 : index
    %25 = arith.index_cast %arg3 : i32 to index
    %26 = arith.minsi %24, %25 : index
    %27 = arith.maxsi %26, %23 : index
    %28 = arith.subi %27, %23 : index
    %29 = "tts.load"(%21, %28) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, index) -> tensor<1024xf32>
    %30 = arith.addf %19, %29 : tensor<1024xf32>
    %31 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %32 = tts.make_tptr %arg2 to sizes: [1024], strides: [1], offsets: [%2], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
    %33 = tt.addptr %31, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %34 = arith.index_cast %1 : i32 to index
    %c1024_1 = arith.constant 1024 : index
    %35 = arith.addi %c1024_1, %34 : index
    %36 = arith.index_cast %arg3 : i32 to index
    %37 = arith.minsi %35, %36 : index
    %38 = arith.maxsi %37, %34 : index
    %39 = arith.subi %38, %34 : index
    "tts.store"(%32, %30, %39) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>, index) -> ()
    tt.return
  }
}

