//===- Pipelines.cpp - Triton-to-LiteRT bridge pipeline ------------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TritonToLiteRT/Bridge.h"

#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::triton_to_litert {

void buildTritonToLiteRTBridgePipeline(OpPassManager &passManager) {
  passManager.addPass(triton::createTritonToStructuredPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(triton::createTritonToUnstructuredPass());

  triton::TritonArithToLinalgOptions arithmeticOptions;
  arithmeticOptions.tensorPtrToLinalg = false;
  arithmeticOptions.pidsToFuncArgs = true;
  passManager.addPass(triton::createTritonArithToLinalg(arithmeticOptions));

  passManager.addPass(createVerifyBridgeInputPass());
  passManager.addPass(createNormalizeLaunchMetadataPass());
  passManager.addPass(createExtractStructuredInputSemanticsPass());
  passManager.addPass(createFunctionalizeStructuredInputsPass());
  passManager.addPass(createClassifyVectorAddLinalgPass());
  passManager.addPass(createFunctionalizeStructuredOutputPass());
  passManager.addPass(createVerifyBridgeIRV1Pass());
}

void registerTritonToLiteRTBridgePipeline() {
  static PassPipelineRegistration<> registration(
      "triton-to-litert-bridge",
      "Recover Triton semantics and emit milestone-1 Bridge IR v1",
      buildTritonToLiteRTBridgePipeline);
}

} // namespace mlir::triton_to_litert
