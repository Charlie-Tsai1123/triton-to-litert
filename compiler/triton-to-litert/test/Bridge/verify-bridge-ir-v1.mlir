// RUN: triton-to-litert-opt --verify-triton-to-litert-bridge-ir-v1 --verify-each %s > %t
// RUN: triton-to-litert-opt --verify-triton-to-litert-bridge-ir-v1 --verify-each %t | FileCheck %s

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(
      %a: tensor<1024xf32>,
      %b: tensor<1024xf32>) -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise
        kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>)
        -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// CHECK: module attributes {
// CHECK-SAME: triton_to_litert.bridge_version = 1 : i32
// CHECK-SAME: triton_to_litert.entry_point = "vector_add"
// CHECK-LABEL: func.func @vector_add(
// CHECK-SAME: %[[A:.*]]: tensor<1024xf32>, %[[B:.*]]: tensor<1024xf32>) -> tensor<1024xf32> {
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK: %[[SUM:.*]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME: ins(%[[A]], %[[B]] : tensor<1024xf32>, tensor<1024xf32>)
// CHECK-SAME: outs(%[[EMPTY]] : tensor<1024xf32>)
// CHECK: return %[[SUM]] : tensor<1024xf32>
// CHECK-NOT: tt.
// CHECK-NOT: tts.
// CHECK-NOT: !tt.ptr
// CHECK-NOT: memref
// CHECK-NOT: bufferization
// CHECK-NOT: linalg.generic
