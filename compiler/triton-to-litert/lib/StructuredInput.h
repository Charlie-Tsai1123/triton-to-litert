//===- StructuredInput.h - Structured input semantic facts -----*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LITERT_LIB_STRUCTURED_INPUT_H
#define TRITON_TO_LITERT_LIB_STRUCTURED_INPUT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
class ModuleOp;

namespace triton_to_litert {

enum class StructuredPointerOffsetForm {
  CanonicalLaunchExpression,
  StaticZero,
};

/// The complete milestone-1 meaning recovered from one tts.make_tptr.
struct StructuredPointerSemantics {
  tts::MakeTensorPtrOp pointer;
  Value base;
  RankedTensorType pointerTensorType;
  Type elementType;
  SmallVector<int64_t> tensorShape;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> wrapShape;
  SmallVector<int32_t> order;
  bool fullRange;
  bool wraps;
  bool broadcasts;
};

/// A proven read-only model input and the load that realizes it.
struct StructuredInputSemantics {
  StructuredPointerSemantics pointer;
  tts::LoadOp load;
  Value loadResult;
  RankedTensorType logicalTensorType;
  unsigned abiArgumentIndex;
  bool masked;
  bool hasOther;
  bool hasBoundaryCheck;
};

bool isMilestoneTensor(Type type);

FailureOr<StructuredPointerSemantics>
analyzeMilestoneStructuredPointer(tts::MakeTensorPtrOp pointer,
                                  StructuredPointerOffsetForm offsetForm);

FailureOr<StructuredInputSemantics>
analyzeMilestoneStructuredInputLoad(tts::LoadOp load,
                                    StructuredPointerOffsetForm offsetForm);

/// Extract exactly the two normalized milestone-1 input accesses, ordered by
/// their model ABI argument index.  This analysis never mutates the IR.
FailureOr<SmallVector<StructuredInputSemantics, 2>>
analyzeMilestoneStructuredInputs(ModuleOp module);

} // namespace triton_to_litert
} // namespace mlir

#endif // TRITON_TO_LITERT_LIB_STRUCTURED_INPUT_H
