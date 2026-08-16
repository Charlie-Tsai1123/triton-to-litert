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
    %3 = tt.addptr %2, %1 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
    %structured, %offsets:2, %strides:2 = "tts.get_structured_state"(%3) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<1x256x!tt.ptr<bf16>>) -> (tensor<1x256x!tt.ptr<bf16>>, index, index, index, index)
    %4 = tt.load %3 : tensor<1x256x!tt.ptr<bf16>>
    %5 = tt.splat %arg1 : i32 -> tensor<1x256xi32>
    %6 = tt.addptr %3, %5 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
    tt.store %6, %4 : tensor<1x256x!tt.ptr<bf16>>
    %7:6 = scf.for %arg2 = %c0 to %c12 step %c3 iter_args(%arg3 = %cst_1, %arg4 = %structured, %arg5 = %offsets#0, %arg6 = %offsets#1, %arg7 = %strides#0, %arg8 = %strides#1) -> (tensor<4x256xbf16>, tensor<1x256x!tt.ptr<bf16>>, index, index, index, index) {
      %14 = tt.broadcast %arg4 : tensor<1x256x!tt.ptr<bf16>> -> tensor<4x256x!tt.ptr<bf16>>
      %15 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
      %16 = arith.index_cast %arg2 : index to i32
      %17 = arith.muli %16, %c256_i32 : i32
      %18 = tt.splat %17 : i32 -> tensor<4xi32>
      %19 = arith.muli %15, %18 : tensor<4xi32>
      %20 = tt.expand_dims %19 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
      %21 = tt.broadcast %20 : tensor<4x1xi32> -> tensor<4x256xi32>
      %22 = tt.addptr %14, %21 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
      %23 = tt.load %22 : tensor<4x256x!tt.ptr<bf16>>
      %24 = arith.addf %arg3, %23 : tensor<4x256xbf16>
      %25 = tt.addptr %arg4, %cst_0 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
      %structured_2, %offsets_3:2, %strides_4:2 = "tts.get_structured_state"(%25) <{resultSegmentSizes = array<i32: 1, 2, 2>}> : (tensor<1x256x!tt.ptr<bf16>>) -> (tensor<1x256x!tt.ptr<bf16>>, index, index, index, index)
      scf.yield %24, %structured_2, %offsets_3#0, %offsets_3#1, %strides_4#0, %strides_4#1 : tensor<4x256xbf16>, tensor<1x256x!tt.ptr<bf16>>, index, index, index, index
    }
    %8 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %9 = arith.muli %8, %cst : tensor<4xi32>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %11 = tt.broadcast %10 : tensor<4x1xi32> -> tensor<4x256xi32>
    %12 = tt.broadcast %3 : tensor<1x256x!tt.ptr<bf16>> -> tensor<4x256x!tt.ptr<bf16>>
    %13 = tt.addptr %12, %11 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    tt.store %13, %7#0 : tensor<4x256x!tt.ptr<bf16>>
    tt.return
  }
}

