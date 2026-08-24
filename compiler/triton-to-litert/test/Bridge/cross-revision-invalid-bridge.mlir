// REQUIRES: litert-side-bridge
// RUN: not linalg-to-tfl-opt --triton-to-litert-bridge-to-tfl --verify-each %s 2>&1 | FileCheck %s

module attributes {
  triton_to_litert.bridge_version = 2 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %empty = tensor.empty() : tensor<1024xf32>
    %sum = linalg.elementwise kind=#linalg.elementwise_kind<add>
        ins(%a, %b : tensor<1024xf32>, tensor<1024xf32>)
        outs(%empty : tensor<1024xf32>) -> tensor<1024xf32>
    return %sum : tensor<1024xf32>
  }
}

// CHECK: error: Triton-to-LiteRT unsupported Bridge IR version: expected i32 version 1
