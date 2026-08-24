// REQUIRES: litert-side-bridge
// RUN: triton-to-litert-opt --pass-pipeline='builtin.module(triton-to-litert-bridge)' --verify-each %S/pipeline-entry.mlir > %t.bridge.mlir
// RUN: triton-to-litert-opt --pass-pipeline='builtin.module(triton-to-litert-bridge)' --verify-each %S/pipeline-entry.mlir > %t.bridge-again.mlir
// RUN: diff %t.bridge.mlir %t.bridge-again.mlir
// RUN: FileCheck %s --check-prefix=BRIDGE --input-file=%t.bridge.mlir
// RUN: FileCheck %s --check-prefix=BRIDGE-CLOSED --input-file=%t.bridge.mlir
// RUN: linalg-to-tfl-opt --verify-each %t.bridge.mlir > %t.litert-reparsed.mlir
// RUN: FileCheck %s --check-prefix=BRIDGE --input-file=%t.litert-reparsed.mlir
// RUN: FileCheck %s --check-prefix=BRIDGE-CLOSED --input-file=%t.litert-reparsed.mlir
// RUN: linalg-to-tfl-opt --triton-to-litert-bridge-to-tfl --verify-each %t.litert-reparsed.mlir > %t.tfl.mlir
// RUN: FileCheck %s --check-prefix=TFL --input-file=%t.tfl.mlir
// RUN: FileCheck %s --check-prefix=TFL-CLOSED --input-file=%t.tfl.mlir
// RUN: not triton-to-litert-opt --pass-pipeline='builtin.module(triton-to-litert-bridge)' --verify-each %S/unstructured-from-ttir.mlir > %t.rejected.mlir 2> %t.producer-error
// RUN: test ! -s %t.rejected.mlir
// RUN: FileCheck %s --check-prefix=PRODUCER-ERROR --input-file=%t.producer-error
// RUN: not linalg-to-tfl-opt --triton-to-litert-bridge-to-tfl --verify-each %s > %t.rejected-tfl.mlir 2> %t.consumer-error
// RUN: test ! -s %t.rejected-tfl.mlir
// RUN: FileCheck %s --check-prefix=CONSUMER-ERROR --input-file=%t.consumer-error

// BRIDGE: module attributes {
// BRIDGE-SAME: triton_to_litert.bridge_version = 1 : i32
// BRIDGE-SAME: triton_to_litert.entry_point = "vector_add"
// BRIDGE-LABEL: func.func @vector_add(
// BRIDGE-SAME: tensor<1024xf32>
// BRIDGE-SAME: tensor<1024xf32>
// BRIDGE-SAME: -> tensor<1024xf32>
// BRIDGE: tensor.empty() : tensor<1024xf32>
// BRIDGE: %[[SUM:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// BRIDGE: return %[[SUM]] : tensor<1024xf32>

// BRIDGE-CLOSED: module
// BRIDGE-CLOSED-NOT: tt.
// BRIDGE-CLOSED-NOT: tts.
// BRIDGE-CLOSED-NOT: !tt.ptr
// BRIDGE-CLOSED-NOT: memref
// BRIDGE-CLOSED-NOT: bufferization
// BRIDGE-CLOSED-NOT: scf.
// BRIDGE-CLOSED-NOT: linalg.generic
// BRIDGE-CLOSED-NOT: triton_to_litert.launch_grid
// BRIDGE-CLOSED-NOT: tensor<?

// TFL-LABEL: func.func @vector_add(
// TFL-SAME: %[[A:[^:]+]]: tensor<1024xf32>
// TFL-SAME: %[[B:[^:]+]]: tensor<1024xf32>
// TFL-SAME: -> tensor<1024xf32>
// TFL-COUNT-1: %[[TFL_SUM:.+]] = tfl.add %[[A]], %[[B]]
// TFL-SAME: fused_activation_function = "NONE"
// TFL-SAME: tensor<1024xf32>
// TFL: return %[[TFL_SUM]] : tensor<1024xf32>

// TFL-CLOSED: module
// TFL-CLOSED-NOT: triton_to_litert.
// TFL-CLOSED-NOT: linalg.
// TFL-CLOSED-NOT: tensor.empty
// TFL-CLOSED-NOT: tt.
// TFL-CLOSED-NOT: tts.
// TFL-CLOSED-NOT: !tt.ptr
// TFL-CLOSED-NOT: memref
// TFL-CLOSED-NOT: bufferization
// TFL-CLOSED-NOT: scf.

// PRODUCER-ERROR: Triton-to-LiteRT Bridge Input milestone 1 does not support unstructured memory operation 'tts.gather'

// CONSUMER-ERROR: error:
// CONSUMER-ERROR: operation being parsed with an unregistered dialect
// CONSUMER-ERROR: LiteRT-side Bridge IR verification failed

module attributes {
  triton_to_litert.bridge_version = 1 : i32,
  triton_to_litert.entry_point = "vector_add"
} {
  func.func @vector_add(%a: tensor<1024xf32>, %b: tensor<1024xf32>)
      -> tensor<1024xf32> {
    %illegal = "memref.alloc"() : () -> memref<1024xf32>
    return %a : tensor<1024xf32>
  }
}
