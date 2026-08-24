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
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "tflite/converter/ir/tfl_ops.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>

namespace {

std::atomic<bool> sawContractDiagnostic = false;

void trackContractDiagnostics(mlir::MLIRContext *context,
                              mlir::func::FuncDialect *) {
  context->getDiagEngine().registerHandler([](mlir::Diagnostic &diagnostic) {
    // Bridge and TFL-output verifier errors use this normative prefix. Parser
    // errors do not, so preserve their detail while adding the phase boundary.
    if (diagnostic.getSeverity() == mlir::DiagnosticSeverity::Error &&
        llvm::StringRef(diagnostic.str()).starts_with("Triton-to-LiteRT "))
      sawContractDiagnostic.store(true, std::memory_order_relaxed);
    return mlir::failure();
  });
}

} // namespace

int main(int argc, char **argv) {
  mlir::triton_to_litert::registerBridgeToTFLPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                  mlir::TFL::TensorFlowLiteDialect>();
  registry.addExtension(trackContractDiagnostics);

  sawContractDiagnostic.store(false, std::memory_order_relaxed);
  mlir::LogicalResult result = mlir::MlirOptMain(
      argc, argv, "Triton-to-LiteRT Bridge IR optimizer\n", registry);
  if (mlir::failed(result) &&
      !sawContractDiagnostic.load(std::memory_order_relaxed))
    llvm::errs() << "error: LiteRT-side Bridge IR verification failed\n";
  return mlir::asMainReturnCode(result);
}
