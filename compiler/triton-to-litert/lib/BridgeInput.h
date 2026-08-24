//===- BridgeInput.h - Milestone 1 Bridge Input verification ---*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LITERT_LIB_BRIDGE_INPUT_H
#define TRITON_TO_LITERT_LIB_BRIDGE_INPUT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
class ModuleOp;

namespace triton_to_litert {

inline constexpr llvm::StringLiteral kLaunchGridAttrName =
    "triton_to_litert.launch_grid";
inline constexpr llvm::StringLiteral kBufferRolesAttrName =
    "triton_to_litert.buffer_roles";
inline constexpr llvm::StringLiteral kBuffersDistinctAttrName =
    "triton_to_litert.buffers_distinct";
inline constexpr int64_t kMilestoneExtent = 1024;
inline constexpr unsigned kLhsBufferArgument = 0;
inline constexpr unsigned kRhsBufferArgument = 1;
inline constexpr unsigned kOutputBufferArgument = 2;
inline constexpr unsigned kInputBufferCount = 2;
inline constexpr unsigned kModelBufferCount = 3;
inline constexpr unsigned kLaunchDimensionCount = 3;
inline constexpr unsigned kLaunchArgumentCount = 6;
inline constexpr unsigned kProgramIdXArgument = 6;
inline constexpr unsigned kMilestoneArgumentCount =
    kModelBufferCount + kLaunchArgumentCount;

LogicalResult verifyMilestoneBridgeInput(ModuleOp module);

/// Revalidate and return the structured pointers fed by the canonical
/// milestone-1 launch-offset expression.  Keeping this analysis shared makes
/// the verifier and the destructive normalization agree on the exact launch
/// dataflow they accept.
FailureOr<SmallVector<tts::MakeTensorPtrOp>>
analyzeMilestoneLaunchContract(func::FuncOp function);

} // namespace triton_to_litert
} // namespace mlir

#endif // TRITON_TO_LITERT_LIB_BRIDGE_INPUT_H
