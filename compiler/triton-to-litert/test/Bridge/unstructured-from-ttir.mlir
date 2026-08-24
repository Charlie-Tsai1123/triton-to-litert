// RUN: not triton-to-litert-opt --pass-pipeline='builtin.module(triton-to-litert-bridge)' --verify-each %s 2>&1 | FileCheck %s

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  tt.func public @unstructured_load(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %indices: tensor<1024xi32>) {
    %a_base = tt.splat %a
        : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %a_ptrs = tt.addptr %a_base, %indices
        : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %value = tt.load %a_ptrs : tensor<1024x!tt.ptr<f32>>
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32}
        : tensor<1024xi32>
    %out_base = tt.splat %out
        : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %range
        : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %out_ptrs, %value : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK: Triton-to-LiteRT Bridge Input milestone 1 does not support unstructured memory operation 'tts.gather'
// CHECK-NOT: memref
// CHECK-NOT: bufferization
