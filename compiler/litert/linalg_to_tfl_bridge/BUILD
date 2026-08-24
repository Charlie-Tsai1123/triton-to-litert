load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "bridge_pass",
    srcs = ["//linalg-to-tfl:bridge_pass_source"],
    hdrs = ["//linalg-to-tfl:bridge_pass_header"],
    deps = [
        "//tflite/converter:tensorflow_lite_ops",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_binary(
    name = "linalg-to-tfl-opt",
    srcs = ["//linalg-to-tfl:bridge_tool_source"],
    deps = [
        ":bridge_pass",
        "//tflite/converter:tensorflow_lite_ops",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgDialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:TensorDialect",
    ],
)
