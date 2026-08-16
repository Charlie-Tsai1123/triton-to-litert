module {
  tt.func @kernel(%arg0: !tt.ptr<bf16>, %arg1: i32) {
    %cst = arith.constant dense<256> : tensor<4xi32>
    %cst_0 = arith.constant dense<256> : tensor<1x256xi32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x256xbf16>
    %c256_i32 = arith.constant 256 : i32
    %c3 = arith.constant 3 : index
    %c12 = arith.constant 12 : index
    %c0 = arith.constant 0 : index
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %2 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x256x!tt.ptr<bf16>>
    %3 = tts.make_tptr %arg0 to sizes: [1, 256], strides: [0, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16> to tensor<1x256x!tt.ptr<bf16>>
    %4 = tt.addptr %2, %1 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
    %c0_2 = arith.constant 0 : index
    %c0_3 = arith.constant 0 : index
    %c0_4 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %5 = "tts.load"(%3) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1x256x!tt.ptr<bf16>>) -> tensor<1x256xbf16>
    %6 = tt.splat %arg1 : i32 -> tensor<1x256xi32>
    %7 = arith.index_cast %arg1 : i32 to index
    %8 = tts.make_tptr %arg0 to sizes: [1, 256], strides: [0, 1], offsets: [%7, 0], shape: [0, 0], order: [] : <bf16> to tensor<1x256x!tt.ptr<bf16>>
    %9 = tt.addptr %4, %6 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
    "tts.store"(%8, %5) <{static_mask_dims = array<i64>}> : (tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xbf16>) -> ()
    %10:6 = scf.for %arg2 = %c0 to %c12 step %c3 iter_args(%arg3 = %cst_1, %arg4 = %3, %arg5 = %c0_2, %arg6 = %c0_3, %arg7 = %c0_4, %arg8 = %c1) -> (tensor<4x256xbf16>, tensor<1x256x!tt.ptr<bf16>>, index, index, index, index) {
      %18 = tts.make_tptr %arg0 to sizes: [1, 256], strides: [%arg7, %arg8], offsets: [%arg5, %arg6], shape: [0, 0], order: [] : <bf16> to tensor<1x256x!tt.ptr<bf16>>
      %19 = tt.broadcast %arg4 : tensor<1x256x!tt.ptr<bf16>> -> tensor<4x256x!tt.ptr<bf16>>
      %20 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
      %21 = arith.index_cast %arg2 : index to i32
      %22 = arith.muli %21, %c256_i32 : i32
      %23 = arith.index_cast %22 : i32 to index
      %24 = tt.splat %22 : i32 -> tensor<4xi32>
      %25 = arith.muli %20, %24 : tensor<4xi32>
      %26 = tt.expand_dims %25 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
      %27 = tt.broadcast %26 : tensor<4x1xi32> -> tensor<4x256xi32>
      %28 = arith.addi %arg7, %23 : index
      %29 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [%28, %arg8], offsets: [%arg5, %arg6], shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
      %30 = tt.addptr %19, %27 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
      %31 = "tts.load"(%29) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>) -> tensor<4x256xbf16>
      %32 = arith.addf %arg3, %31 : tensor<4x256xbf16>
      %c256_5 = arith.constant 256 : index
      %33 = arith.addi %arg5, %c256_5 : index
      %34 = tts.make_tptr %arg0 to sizes: [1, 256], strides: [%arg7, %arg8], offsets: [%33, %arg6], shape: [0, 0], order: [] : <bf16> to tensor<1x256x!tt.ptr<bf16>>
      %35 = tt.addptr %arg4, %cst_0 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
      scf.yield %32, %34, %33, %arg6, %arg7, %arg8 : tensor<4x256xbf16>, tensor<1x256x!tt.ptr<bf16>>, index, index, index, index
    }
    %11 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %12 = arith.muli %11, %cst : tensor<4xi32>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %14 = tt.broadcast %13 : tensor<4x1xi32> -> tensor<4x256xi32>
    %15 = tt.broadcast %4 : tensor<1x256x!tt.ptr<bf16>> -> tensor<4x256x!tt.ptr<bf16>>
    %c256 = arith.constant 256 : index
    %16 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [%c256, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
    %17 = tt.addptr %15, %14 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    "tts.store"(%16, %10#0) <{static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xbf16>) -> ()
    tt.return
  }
}

