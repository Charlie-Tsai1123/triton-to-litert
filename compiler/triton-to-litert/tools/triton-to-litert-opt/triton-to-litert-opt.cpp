//===- triton-to-litert-opt.cpp - Triton-side bridge driver ---------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TritonToLiteRT/Bridge.h"

#include "RegisterTritonSharedDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerTritonSharedDialects(registry);
  mlir::triton_to_litert::registerTritonToLiteRTBridgePasses();
  mlir::triton_to_litert::registerTritonToLiteRTBridgePipeline();

  mlir::LogicalResult result = mlir::MlirOptMain(
      argc, argv, "Triton-to-LiteRT bridge driver\n", registry);
  if (mlir::failed(result))
    llvm::errs() << "Triton-side Bridge IR serialization failed\n";
  return mlir::asMainReturnCode(result);
}
