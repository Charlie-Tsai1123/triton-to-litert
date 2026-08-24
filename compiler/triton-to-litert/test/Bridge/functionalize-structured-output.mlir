// RUN: triton-to-litert-opt --pass-pipeline='builtin.module(functionalize-triton-to-litert-structured-output,verify-triton-to-litert-bridge-ir-v1)' --verify-each %s | FileCheck %s

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @vector_add(
      %a: tensor<1024xf32>,
      %b: tensor<1024xf32>,
      %out: !tt.ptr<f32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise
        kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>)
        -> tensor<1024xf32>
    %out_ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%out_ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// CHECK: module attributes {
// CHECK-SAME: triton_to_litert.bridge_version = 1 : i32
// CHECK-SAME: triton_to_litert.entry_point = "vector_add"
// CHECK-NOT: triton_to_litert.buffer_roles
// CHECK-NOT: triton_to_litert.buffers_distinct
// CHECK-LABEL: func.func @vector_add(
// CHECK-SAME: %[[A:.*]]: tensor<1024xf32>, %[[B:.*]]: tensor<1024xf32>) -> tensor<1024xf32> {
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SUM:.*]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<1024xf32>, tensor<1024xf32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<1024xf32>)
// CHECK: return %[[SUM]] : tensor<1024xf32>
// CHECK-NOT: tts.make_tptr
// CHECK-NOT: "tts.store"
// CHECK-NOT: !tt.ptr
// CHECK-NOT: tt.
// CHECK-NOT: tts.
// CHECK-NOT: memref
// CHECK-NOT: bufferization
// CHECK-NOT: linalg.generic
