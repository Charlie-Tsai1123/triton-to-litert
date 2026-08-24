// RUN: not triton-to-litert-opt --pass-pipeline='builtin.module(triton-to-litert-bridge)' %s 2>&1 | FileCheck %s

module {
  func.func @categorically_illegal(%buffer: memref<4xf32>) {
    return
  }
}

// CHECK: error: Triton-to-LiteRT Bridge Input contains categorically illegal type 'memref<4xf32>'
