//===- linalg-to-tfl-opt.cpp - LiteRT-side bridge driver -----------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "linalg-to-tfl/include/LinalgToTFL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "tflite/converter/ir/tfl_ops.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace {

bool verifiesTFLOutput(int argc, char **argv) {
  for (int index = 1; index < argc; ++index)
    if (llvm::StringRef(argv[index]) == "--verify-triton-to-litert-tfl-output")
      return true;
  return false;
}

} // namespace

int main(int argc, char **argv) {
  mlir::triton_to_litert::registerBridgeToTFLPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                  mlir::TFL::TensorFlowLiteDialect>();

  mlir::LogicalResult result = mlir::MlirOptMain(
      argc, argv, "Triton-to-LiteRT Bridge IR optimizer\n", registry);
  if (mlir::failed(result)) {
    if (verifiesTFLOutput(argc, argv))
      llvm::errs() << "error: LiteRT-side TFL Output IR verification failed\n";
    else
      llvm::errs() << "error: LiteRT-side Bridge IR verification failed\n";
  }
  return mlir::asMainReturnCode(result);
}
