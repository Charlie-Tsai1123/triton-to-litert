//===- Bridge.h - Triton-to-LiteRT bridge entry points ---------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LITERT_BRIDGE_H
#define TRITON_TO_LITERT_BRIDGE_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class OpPassManager;

namespace triton_to_litert {

std::unique_ptr<Pass> createVerifyBridgeInputPass();
std::unique_ptr<Pass> createNormalizeLaunchMetadataPass();
std::unique_ptr<Pass> createExtractStructuredInputSemanticsPass();

void buildTritonToLiteRTBridgePipeline(OpPassManager &passManager);

void registerTritonToLiteRTBridgePasses();
void registerNormalizeLaunchMetadataPass();
void registerExtractStructuredInputSemanticsPass();
void registerTritonToLiteRTBridgePipeline();

} // namespace triton_to_litert
} // namespace mlir

#endif // TRITON_TO_LITERT_BRIDGE_H
