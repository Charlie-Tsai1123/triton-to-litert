// RUN: linalg-to-tfl-opt --triton-to-litert-bridge-to-tfl --verify-each %s > %t
// RUN: linalg-to-tfl-opt --verify-triton-to-litert-tfl-output --verify-each %t | FileCheck %s

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

// CHECK-LABEL: func.func @vector_add(
// CHECK-SAME: %[[A:[^:]+]]: tensor<1024xf32>
// CHECK-SAME: %[[B:[^:]+]]: tensor<1024xf32>
// CHECK-SAME: -> tensor<1024xf32>
// CHECK-COUNT-1: %[[SUM:.+]] = tfl.add %[[A]], %[[B]]
// CHECK-SAME: fused_activation_function = "NONE"
// CHECK-SAME: tensor<1024xf32>
// CHECK: return %[[SUM]] : tensor<1024xf32>
// CHECK-NOT: triton_to_litert.
// CHECK-NOT: linalg.
// CHECK-NOT: tensor.empty
// CHECK-NOT: tt.
// CHECK-NOT: tts.
// CHECK-NOT: !tt.ptr
// CHECK-NOT: memref.
// CHECK-NOT: bufferization.
// CHECK-NOT: unrealized_conversion_cast
