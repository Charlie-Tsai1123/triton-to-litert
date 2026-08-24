//===- StructuredOutput.h - Structured output semantic facts ---*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LITERT_LIB_STRUCTURED_OUTPUT_H
#define TRITON_TO_LITERT_LIB_STRUCTURED_OUTPUT_H

#include "StructuredInput.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class ModuleOp;

namespace triton_to_litert {

/// The complete milestone-1 meaning of the sole write-only model output.
struct StructuredOutputSemantics {
  StructuredPointerSemantics pointer;
  tts::StoreOp store;
  Value storedValue;
  RankedTensorType logicalTensorType;
  unsigned abiArgumentIndex;
  linalg::ElementwiseOp producer;
  func::ReturnOp voidReturn;
  bool masked;
  bool hasBoundaryCheck;
};

/// Prove the exact classified, normalized, full-range output contract without
/// mutating the module.
FailureOr<StructuredOutputSemantics>
analyzeMilestoneStructuredOutput(ModuleOp module);

} // namespace triton_to_litert
} // namespace mlir

#endif // TRITON_TO_LITERT_LIB_STRUCTURED_OUTPUT_H
