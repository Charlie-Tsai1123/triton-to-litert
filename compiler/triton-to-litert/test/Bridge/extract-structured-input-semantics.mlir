// RUN: triton-to-litert-opt --pass-pipeline='builtin.module(verify-triton-to-litert-bridge-input,normalize-triton-to-litert-launch-metadata,extract-triton-to-litert-structured-input-semantics{emit-descriptors=true})' --verify-each --verify-diagnostics %s > %t.extracted
// RUN: triton-to-litert-opt --pass-pipeline='builtin.module(verify-triton-to-litert-bridge-input,normalize-triton-to-litert-launch-metadata)' --verify-each %s > %t.normalized
// RUN: diff %t.normalized %t.extracted
// RUN: FileCheck %s --input-file=%t.extracted

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
    // expected-remark @+1 {{structured input semantics: abi_input=0, source_buffer=arg0, tensor_shape=[1024], sizes=[1024], strides=[1], offsets=[0], element_type=f32, load_result=tensor<1024xf32>, shape_state=[0], order=[], full_range=true, wrap=false, broadcast=false, mask=none, other=none, boundary_check=none}}
    %a_value = "tts.load"(%a_ptr)
        <{operandSegmentSizes = array<i32: 1, 0, 0>,
          static_mask_dims = array<i64>}>
        : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
    %b_ptr = tts.make_tptr %b to sizes: [1024], strides: [1],
        offsets: [%tile], shape: [0], order: []
        : <f32> to tensor<1024x!tt.ptr<f32>>
    // expected-remark @+1 {{structured input semantics: abi_input=1, source_buffer=arg1, tensor_shape=[1024], sizes=[1024], strides=[1], offsets=[0], element_type=f32, load_result=tensor<1024xf32>, shape_state=[0], order=[], full_range=true, wrap=false, broadcast=false, mask=none, other=none, boundary_check=none}}
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

// CHECK-LABEL: func.func @vector_add(
// CHECK-SAME: %[[A:.*]]: !tt.ptr<f32>, %[[B:.*]]: !tt.ptr<f32>, %[[OUT:.*]]: !tt.ptr<f32>) {
// CHECK: %[[APTR:.*]] = tts.make_tptr %[[A]] to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: []
// CHECK: %[[AVALUE:.*]] = "tts.load"(%[[APTR]])
// CHECK: %[[BPTR:.*]] = tts.make_tptr %[[B]] to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: []
// CHECK: %[[BVALUE:.*]] = "tts.load"(%[[BPTR]])
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[AVALUE]], %[[BVALUE]] : tensor<1024xf32>, tensor<1024xf32>)
// CHECK: %[[OUTPTR:.*]] = tts.make_tptr %[[OUT]] to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: []
// CHECK: "tts.store"(%[[OUTPTR]],
// CHECK-NOT: memref
// CHECK-NOT: alloc
// CHECK-NOT: copy
// CHECK-NOT: subview
// CHECK-NOT: bufferization
