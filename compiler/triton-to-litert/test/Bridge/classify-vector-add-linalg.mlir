// RUN: triton-to-litert-opt --classify-triton-to-litert-vector-add-linalg --verify-each %s | FileCheck %s

#identity = affine_map<(d0) -> (d0)>

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @vector_add(
      %a: tensor<1024xf32>,
      %b: tensor<1024xf32>,
      %out: !tt.ptr<f32>) {
    %init = tensor.empty() : tensor<1024xf32>
    %sum = linalg.generic {
        indexing_maps = [#identity, #identity, #identity],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%init : tensor<1024xf32>) {
      ^bb0(%lhs: f32, %rhs: f32, %unused_init: f32):
        %added = arith.addf %lhs, %rhs : f32
        linalg.yield %added : f32
      } -> tensor<1024xf32>
    %out_ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [0], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%out_ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @vector_add(
// CHECK-SAME: %[[A:[^:]+]]: tensor<1024xf32>
// CHECK-SAME: %[[B:[^:]+]]: tensor<1024xf32>
// CHECK-SAME: %[[OUT:[^:]+]]: !tt.ptr<f32>
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1024xf32>
// CHECK-NOT: tensor.empty
// CHECK: %[[SUM:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<1024xf32>, tensor<1024xf32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK: %[[OUT_PTR:.+]] = tts.make_tptr %[[OUT]]
// CHECK: "tts.store"(%[[OUT_PTR]], %[[SUM]])
// CHECK-NOT: linalg.generic
// CHECK-NOT: memref
// CHECK-NOT: bufferization
// CHECK-NOT: "tfl.
// CHECK-NOT: triton_to_litert.bridge_version
