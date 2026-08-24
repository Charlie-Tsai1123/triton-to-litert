//===- BridgeIR.h - Versioned Bridge IR contract ---------------*- C++ -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LITERT_LIB_BRIDGE_IR_H
#define TRITON_TO_LITERT_LIB_BRIDGE_IR_H

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
class ModuleOp;
class Type;

namespace triton_to_litert {

inline constexpr llvm::StringLiteral kBridgeVersionAttrName =
    "triton_to_litert.bridge_version";
inline constexpr llvm::StringLiteral kBridgeEntryPointAttrName =
    "triton_to_litert.entry_point";
inline constexpr int64_t kBridgeIRVersion = 1;
inline constexpr int64_t kBridgeIRV1Extent = 1024;

bool isBridgeIRV1Tensor(Type type);
LogicalResult verifyBridgeIRV1(ModuleOp module);

} // namespace triton_to_litert
} // namespace mlir

#endif // TRITON_TO_LITERT_LIB_BRIDGE_IR_H
