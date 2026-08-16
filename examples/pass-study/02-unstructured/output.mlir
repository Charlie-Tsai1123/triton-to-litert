module {
  tt.func public @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tts.gather %arg0[%4] mask = %6 : (<f32>, tensor<1024xi32>) -> tensor<1024xf32>
    %8 = tts.gather %arg1[%4] mask = %6 : (<f32>, tensor<1024xi32>) -> tensor<1024xf32>
    %9 = arith.addf %7, %8 : tensor<1024xf32>
    tts.scatter %9 into %arg2[%4] mask = %6 : tensor<1024xf32> into (<f32>, tensor<1024xi32>)
    tt.return
  }
}

