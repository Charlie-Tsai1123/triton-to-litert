//===- Passes.h - Triton-to-LiteRT TFL bridge passes ------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LITERT_LINALGTOTFL_PASSES_H_
#define TRITON_TO_LITERT_LINALGTOTFL_PASSES_H_

#include <memory>

namespace mlir {
class Pass;

namespace triton_to_litert {

std::unique_ptr<Pass> createBridgeToTFLPass();
std::unique_ptr<Pass> createVerifyTFLOutputPass();
void registerBridgeToTFLPasses();

} // namespace triton_to_litert
} // namespace mlir

#endif // TRITON_TO_LITERT_LINALGTOTFL_PASSES_H_
