//===- FunctionalizeStructuredInputs.cpp - Tensorize input ABI -----------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TritonToLiteRT/Bridge.h"

#include "BridgeInput.h"
#include "StructuredInput.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::triton_to_litert {
namespace {

LogicalResult functionalizeStructuredInputs(ModuleOp module) {
  FailureOr<SmallVector<StructuredInputSemantics, 2>> inputs =
      analyzeMilestoneStructuredInputs(module);
  if (failed(inputs))
    return failure();

  if (failed(validateMilestoneOperationAllowlist(
          module, MilestoneOperationStage::NormalizedBridgePreparation)))
    return failure();

  func::FuncOp function = *module.getOps<func::FuncOp>().begin();
  SmallVector<Type> argumentTypes(function.getArgumentTypes());
  for (StructuredInputSemantics &input : *inputs) {
    if (input.abiArgumentIndex >= kInputBufferCount ||
        input.pointer.base != function.getArgument(input.abiArgumentIndex)) {
      input.load.emitError(
          "Triton-to-LiteRT model ABI cannot be functionalized: structured "
          "input descriptor does not match its source buffer argument");
      return failure();
    }
    argumentTypes[input.abiArgumentIndex] = input.logicalTensorType;
  }

  // All semantic and closed-world checks above complete before source memory
  // operations are erased.  Each replacement is therefore an exact
  // materialization of a proven full-range input descriptor.
  for (StructuredInputSemantics &input : *inputs) {
    BlockArgument tensorArgument = function.getArgument(input.abiArgumentIndex);
    tensorArgument.setType(input.logicalTensorType);
    input.loadResult.replaceAllUsesWith(tensorArgument);
    input.load.erase();
    input.pointer.pointer.erase();
  }

  function.setFunctionType(FunctionType::get(
      function.getContext(), argumentTypes, function.getResultTypes()));
  return success();
}

class FunctionalizeStructuredInputsPass final
    : public PassWrapper<FunctionalizeStructuredInputsPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      FunctionalizeStructuredInputsPass)

  StringRef getArgument() const final {
    return "functionalize-triton-to-litert-structured-inputs";
  }

  StringRef getDescription() const final {
    return "Materialize milestone-1 structured loads as tensor inputs";
  }

  void runOnOperation() final {
    if (failed(functionalizeStructuredInputs(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFunctionalizeStructuredInputsPass() {
  return std::make_unique<FunctionalizeStructuredInputsPass>();
}

void registerFunctionalizeStructuredInputsPass() {
  static PassRegistration<FunctionalizeStructuredInputsPass> registration;
}

} // namespace mlir::triton_to_litert
