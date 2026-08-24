// RUN: triton-to-litert-opt --pass-pipeline='builtin.module(verify-triton-to-litert-bridge-input,normalize-triton-to-litert-launch-metadata,functionalize-triton-to-litert-structured-inputs)' --verify-each %s > %t
// RUN: FileCheck %s --input-file=%t
// RUN: not triton-to-litert-opt --verify-triton-to-litert-bridge-ir-v1 --verify-each %t 2>&1 | FileCheck %s --check-prefix=PARTIAL

#identity = affine_map<(d0) -> (d0)>
module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @vector_add(
      %a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>,
      %num_x: i32, %num_y: i32, %num_z: i32,
      %pid_x: i32, %pid_y: i32, %pid_z: i32) {
    %c1024 = arith.constant 1024 : i32
    %tile_i32 = arith.muli %pid_x, %c1024 : i32
    %tile = arith.index_cast %tile_i32 : i32 to index
    %a_ptr = tts.make_tptr %a to sizes: [1024], strides: [1],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    %b_value = "tts.load"(%b_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %sum = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]}
        ins(%a_value, %b_value : tensor<1024xf32>, tensor<1024xf32>)
        outs(%a_value : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %init: f32):
        %added = arith.addf %lhs, %rhs : f32
        linalg.yield %added : f32
    } -> tensor<1024xf32>
    %out_ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%out_ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// CHECK: module attributes {
// CHECK-SAME: triton_to_litert.buffer_roles = ["input", "input", "output"]
// CHECK-SAME: triton_to_litert.buffers_distinct
// CHECK-NOT: triton_to_litert.launch_grid
// CHECK-LABEL: func.func @vector_add(
// CHECK-SAME: %[[A:.*]]: tensor<1024xf32>, %[[B:.*]]: tensor<1024xf32>, %[[OUT:.*]]: !tt.ptr<f32>) {
// CHECK-NOT: "tts.load"
// CHECK-NOT: tts.make_tptr %[[A]]
// CHECK-NOT: tts.make_tptr %[[B]]
// CHECK: %[[SUM:.*]] = linalg.generic
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<1024xf32>, tensor<1024xf32>)
// CHECK-SAME: outs(%[[A]] : tensor<1024xf32>)
// CHECK: %[[OUT_PTR:.*]] = tts.make_tptr %[[OUT]] to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: []
// CHECK: "tts.store"(%[[OUT_PTR]], %[[SUM]])
// CHECK: return
// CHECK-NOT: memref
// CHECK-NOT: alloc
// CHECK-NOT: subview
// CHECK-NOT: copy
// CHECK-NOT: bufferization
// CHECK-NOT: triton_to_litert.bridge_version
// CHECK-NOT: triton_to_litert.entry_point

// PARTIAL: error: Triton-to-LiteRT unsupported Bridge IR version
