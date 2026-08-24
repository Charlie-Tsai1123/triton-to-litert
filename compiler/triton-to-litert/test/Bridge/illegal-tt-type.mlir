// RUN: not triton-to-litert-opt --pass-pipeline='builtin.module(triton-to-litert-bridge)' %s 2>&1 | FileCheck %s

module attributes {
  triton_to_litert.buffer_roles = ["input", "input", "output"],
  triton_to_litert.buffers_distinct,
  triton_to_litert.launch_grid = array<i64: 1, 1, 1>
} {
  func.func @tensor_descriptor(%arg0: !tt.tensordesc<1024xf32>) {
    return
  }
}

// CHECK: error: Triton-to-LiteRT Bridge Input contains categorically illegal type '!tt.tensordesc<1024xf32>'
