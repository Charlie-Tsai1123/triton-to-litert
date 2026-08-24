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
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  mlir::triton_to_litert::registerBridgeToTFLPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                  mlir::TFL::TensorFlowLiteDialect>();

  mlir::LogicalResult result = mlir::MlirOptMain(
      argc, argv, "Triton-to-LiteRT Bridge IR optimizer\n", registry);
  if (mlir::failed(result))
    llvm::errs() << "LiteRT-side Bridge IR verification failed; underlying "
                    "parser or verifier diagnostic precedes this message\n";
  return mlir::asMainReturnCode(result);
}
