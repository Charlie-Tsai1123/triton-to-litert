// RUN: not triton-to-litert-opt --pass-pipeline='builtin.module(triton-to-litert-bridge)' %s 2>&1 | FileCheck %s

module {
  func.func @tensor_descriptor(%arg0: !tt.tensordesc<1024xf32>) {
    return
  }
}

// CHECK: error: Triton-to-LiteRT Bridge Input contains categorically illegal type '!tt.tensordesc<1024xf32>'
