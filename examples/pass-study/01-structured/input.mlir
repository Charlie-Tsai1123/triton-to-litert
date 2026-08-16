// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : i32
  )
  {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>

    %splat_arg0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x256x!tt.ptr<bf16>>
    %2 = tt.addptr %splat_arg0, %1 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
    // source = %arg0, offset = [0, 0], size = [1, 256], stride = [0, 1]

    // 1x256 pointer should have meaningful stride in outer dimension
    %3 = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<1x256x!tt.ptr<bf16>>

    %4 = tt.splat %arg1 : i32 -> tensor<1x256xi32>
    // 1x256 pointer should have meaningful stride in outer dimension
    %5 = tt.addptr %2, %4 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
    // source = %arg0, offset = [%arg1, 0], size = [1, 256], stride = [0, 1]

    tt.store %5, %3 : tensor<1x256x!tt.ptr<bf16>>

    %10 = arith.constant 0.0 : bf16
    %11 = tt.splat %10 : bf16 -> tensor<4x256xbf16>

    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %c256 = arith.constant 256 : i32
    %sum_out, %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%sum_iter = %11, %ptr = %2) -> (tensor<4x256xbf16>, tensor<1x256x!tt.ptr<bf16>>) {
        %bptr = tt.broadcast %ptr : tensor<1x256x!tt.ptr<bf16>> -> tensor<4x256x!tt.ptr<bf16>>
        // source = %arg0, offset = [0, 0], size = [4, 256], stride = [0, 1]

        %20 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
        %i_i32 = arith.index_cast %i : index to i32
        %21 = arith.muli %c256, %i_i32 : i32
        %22 = tt.splat %21 : i32 -> tensor<4xi32>
        %23 = arith.muli %20, %22 : tensor<4xi32>
        %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
        %25 = tt.broadcast %24 : tensor<4x1xi32> -> tensor<4x256xi32>
        // offset = [0, 0], size = [4, 256], stride = [i*256, 1]

        // %bptr should have zero stride and %30 should have correct stride
        %30 = tt.addptr %bptr, %25 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
        // source = %arg0, offset = [0, 0], size = [4, 256], stride = [i*256, 1]

        %31 = tt.load %30 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>>
        %32 = arith.addf %sum_iter, %31 : tensor<4x256xbf16>

        %40 = tt.splat %c256 : i32 -> tensor<1x256xi32>
        %41 = tt.addptr %ptr, %40 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
        // source = %arg0, offset = [i*256, 0], size = [4, 256], stride = [i*256, 1]

        scf.yield %32, %41 : tensor<4x256xbf16>, tensor<1x256x!tt.ptr<bf16>>
    }

    %31 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
    %splat_c256 = tt.splat %c256 : i32 -> tensor<4xi32>
    %32 = arith.muli %31, %splat_c256 : tensor<4xi32>
    %33 = tt.expand_dims %32 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %34 = tt.broadcast %33 : tensor<4x1xi32> -> tensor<4x256xi32>
    %35 = tt.broadcast %2 : tensor<1x256x!tt.ptr<bf16>> -> tensor<4x256x!tt.ptr<bf16>>
    %36 = tt.addptr %35, %34 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    tt.store %36, %sum_out : tensor<4x256x!tt.ptr<bf16>>
    tt.return
  }
}

