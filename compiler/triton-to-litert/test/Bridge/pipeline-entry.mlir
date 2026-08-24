// RUN: triton-to-litert-opt --pass-pipeline='builtin.module(triton-to-litert-bridge)' %s > %t.bridge
// RUN: FileCheck %s --input-file=%t.bridge --check-prefix=BRIDGE
// RUN: FileCheck %s --input-file=%t.bridge --check-prefix=MAKERS
// RUN: FileCheck %s --input-file=%t.bridge --check-prefix=LOADS
// RUN: FileCheck %s --input-file=%t.bridge --check-prefix=GENERICS
// RUN: FileCheck %s --input-file=%t.bridge --check-prefix=STORES
// RUN: triton-to-litert-opt --pass-pipeline='builtin.module(triton-to-litert-bridge)' --dump-pass-pipeline %s 2>&1 | FileCheck %s --check-prefix=PIPELINE
// RUN: triton-to-litert-opt --pass-pipeline='builtin.module(triton-to-structured,cse,canonicalize,triton-to-unstructured,triton-arith-to-linalg{tensor-ptr-to-linalg=false pids-to-func-args=true})' %s > %t.prefix
// RUN: FileCheck %s --input-file=%t.prefix --check-prefix=PREFIX
// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s --check-prefix=REFERENCE

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  tt.func public @vector_add(
      %a: !tt.ptr<f32>,
      %b: !tt.ptr<f32>,
      %out: !tt.ptr<f32>) {
    %c1024 = arith.constant 1024 : i32
    %pid = tt.get_program_id x : i32
    %tile_base = arith.muli %pid, %c1024 : i32
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32}
        : tensor<1024xi32>
    %base = tt.splat %tile_base : i32 -> tensor<1024xi32>
    %offsets = arith.addi %base, %range : tensor<1024xi32>

    %a_base = tt.splat %a
        : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %a_ptrs = tt.addptr %a_base, %offsets
        : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %a_value = tt.load %a_ptrs : tensor<1024x!tt.ptr<f32>>

    %b_base = tt.splat %b
        : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %b_ptrs = tt.addptr %b_base, %offsets
        : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %b_value = tt.load %b_ptrs : tensor<1024x!tt.ptr<f32>>

    %sum = arith.addf %a_value, %b_value : tensor<1024xf32>

    %out_base = tt.splat %out
        : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %out_ptrs = tt.addptr %out_base, %offsets
        : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %out_ptrs, %sum : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// PIPELINE: Pass Manager with 9 passes:
// PIPELINE: builtin.module(
// PIPELINE-NEXT:   triton-to-structured
// PIPELINE-NEXT:   cse
// PIPELINE-NEXT:   canonicalize
// PIPELINE-NEXT:   triton-to-unstructured
// PIPELINE-NEXT:   triton-arith-to-linalg
// PIPELINE-SAME: pids-to-func-args=true
// PIPELINE-SAME: tensor-ptr-to-linalg=false
// PIPELINE-NEXT:   verify-triton-to-litert-bridge-input
// PIPELINE-NEXT:   normalize-triton-to-litert-launch-metadata
// PIPELINE-NEXT:   extract-triton-to-litert-structured-input-semantics
// PIPELINE-NEXT:   functionalize-triton-to-litert-structured-inputs

// BRIDGE-NOT: triton_to_litert.launch_grid
// BRIDGE-LABEL: func.func @vector_add(
// BRIDGE-SAME: tensor<1024xf32>
// BRIDGE-SAME: tensor<1024xf32>
// BRIDGE-SAME: !tt.ptr<f32>
// BRIDGE-SAME: ) {
// BRIDGE-NOT: arith.constant
// BRIDGE-NOT: arith.muli
// BRIDGE-NOT: arith.index_cast
// BRIDGE: offsets: [0]
// BRIDGE-NOT: "tts.load"
// BRIDGE-NOT: tt.func
// BRIDGE-NOT: tt.get_program_id
// BRIDGE-NOT: tt.make_range
// BRIDGE-NOT: tt.splat
// BRIDGE-NOT: tt.addptr
// BRIDGE-NOT: tt.load
// BRIDGE-NOT: tt.store
// BRIDGE-NOT: tt.return
// BRIDGE-NOT: memref
// BRIDGE-NOT: bufferization
// BRIDGE-NOT: "tfl.

// PREFIX: triton_to_litert.launch_grid = array<i64: 1, 1, 1>
// PREFIX-LABEL: func.func @vector_add(
// PREFIX-SAME: !tt.ptr<f32>
// PREFIX-SAME: !tt.ptr<f32>
// PREFIX-SAME: !tt.ptr<f32>
// PREFIX-SAME: i32
// PREFIX-SAME: i32
// PREFIX-SAME: i32
// PREFIX-SAME: i32
// PREFIX-SAME: i32
// PREFIX-SAME: i32
// PREFIX: arith.muli
// PREFIX: arith.index_cast
// PREFIX: offsets: [

// MAKERS-COUNT-1: tts.make_tptr
// MAKERS-NOT: tts.make_tptr
// LOADS-NOT: "tts.load"
// GENERICS-COUNT-1: linalg.generic
// GENERICS-NOT: linalg.generic
// STORES-COUNT-1: "tts.store"
// STORES-NOT: "tts.store"

// REFERENCE: memref.reinterpret_cast
// REFERENCE: bufferization.to_tensor
