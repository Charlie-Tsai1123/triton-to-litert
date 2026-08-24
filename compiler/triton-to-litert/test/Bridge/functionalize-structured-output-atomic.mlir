// RUN: not triton-to-litert-opt --functionalize-triton-to-litert-structured-output --mlir-print-ir-after-failure --verify-each %s 2>&1 | FileCheck %s

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct
} {
  func.func @invalid_output(
      %a: tensor<1024xf32>, %b: tensor<1024xf32>, %out: !tt.ptr<f32>) {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    %ptr = tts.make_tptr %out to sizes: [1024], strides: [1],
        offsets: [1], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    "tts.store"(%ptr, %sum) <{static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
    return
  }
}

// CHECK: IR Dump After
// CHECK-SAME: FunctionalizeStructuredOutputPass Failed
// CHECK: triton_to_litert.buffer_roles = ["input", "input", "output"]
// CHECK: triton_to_litert.buffers_distinct
// CHECK-NOT: triton_to_litert.bridge_version
// CHECK-NOT: triton_to_litert.entry_point
// CHECK-LABEL: func.func @invalid_output(
// CHECK-SAME: tensor<1024xf32>
// CHECK-SAME: tensor<1024xf32>
// CHECK-SAME: !tt.ptr<f32>) {
// CHECK: tts.make_tptr
// CHECK-SAME: offsets: [1]
// CHECK: "tts.store"
// CHECK: return
