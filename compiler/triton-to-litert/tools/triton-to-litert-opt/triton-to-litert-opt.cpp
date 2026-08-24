//===- triton-to-litert-opt.cpp - Triton-side bridge driver ---------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TritonToLiteRT/Bridge.h"

#include "RegisterTritonSharedDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerTritonSharedDialects(registry);
  mlir::triton_to_litert::registerTritonToLiteRTBridgePasses();
  mlir::triton_to_litert::registerTritonToLiteRTBridgePipeline();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton-to-LiteRT bridge driver\n", registry));
}
