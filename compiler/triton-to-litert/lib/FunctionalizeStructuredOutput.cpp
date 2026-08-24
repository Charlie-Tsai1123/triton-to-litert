//===- FunctionalizeStructuredOutput.cpp - Tensorize output ABI ----------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TritonToLiteRT/Bridge.h"

#include "BridgeIR.h"
#include "BridgeInput.h"
#include "StructuredOutput.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::triton_to_litert {
namespace {

template <typename T>
LogicalResult requireExactlyOne(ModuleOp module, SmallVectorImpl<T> &values,
                                const Twine &message) {
  module.walk([&](T value) { values.push_back(value); });
  if (values.size() == 1)
    return success();
  Operation *anchor =
      values.empty() ? module.getOperation() : values.front().getOperation();
  anchor->emitError() << message << ": found " << values.size();
  return failure();
}

FailureOr<StructuredOutputSemantics> emitOutputError(Operation *operation,
                                                     const Twine &message) {
  operation->emitError()
      << "Triton-to-LiteRT model ABI cannot be functionalized: " << message;
  return failure();
}

LogicalResult functionalizeStructuredOutput(ModuleOp module) {
  FailureOr<StructuredOutputSemantics> output =
      analyzeMilestoneStructuredOutput(module);
  if (failed(output))
    return failure();

  func::FuncOp function = output->store->getParentOfType<func::FuncOp>();
  Value result = output->storedValue;

  // Every proof obligation above completes before this destructive rewrite.
  // The erased memory side effect is therefore exactly the returned tensor.
  OpBuilder builder(output->voidReturn);
  func::ReturnOp::create(builder, output->voidReturn.getLoc(), result);
  output->voidReturn.erase();
  output->store.erase();
  output->pointer.pointer.erase();
  if (failed(function.eraseArgument(kOutputBufferArgument))) {
    function.emitError(
        "Triton-to-LiteRT failed to remove the proven-unused output buffer");
    return failure();
  }
  function.setFunctionType(FunctionType::get(
      function.getContext(), function.getArgumentTypes(), result.getType()));

  module->removeAttr(kBufferRolesAttrName);
  module->removeAttr(kBuffersDistinctAttrName);
  module->setAttr(kBridgeVersionAttrName,
                  IntegerAttr::get(IntegerType::get(module.getContext(), 32),
                                   kBridgeIRVersion));
  module->setAttr(kBridgeEntryPointAttrName,
                  StringAttr::get(module.getContext(), function.getSymName()));
  return success();
}

class FunctionalizeStructuredOutputPass final
    : public PassWrapper<FunctionalizeStructuredOutputPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      FunctionalizeStructuredOutputPass)

  StringRef getArgument() const final {
    return "functionalize-triton-to-litert-structured-output";
  }

  StringRef getDescription() const final {
    return "Functionalize the milestone-1 full-range structured output";
  }

  void runOnOperation() final {
    if (failed(functionalizeStructuredOutput(getOperation())))
      signalPassFailure();
  }
};

} // namespace

FailureOr<StructuredOutputSemantics>
analyzeMilestoneStructuredOutput(ModuleOp module) {
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  if (functions.size() != 1)
    return emitOutputError(module, "expected exactly one entry function");

  func::FuncOp function = functions.front();
  FunctionType type = function.getFunctionType();
  if (!function.isPublic() || !function.getBody().hasOneBlock() ||
      type.getNumInputs() != kModelBufferCount || type.getNumResults() != 0 ||
      !isBridgeIRV1Tensor(type.getInput(0)) ||
      !isBridgeIRV1Tensor(type.getInput(1)))
    return emitOutputError(
        function,
        "expected two tensor<1024xf32> inputs and one !tt.ptr<f32> output");

  auto outputPointerType = dyn_cast<triton::PointerType>(type.getInput(2));
  if (!outputPointerType || !outputPointerType.getPointeeType().isF32())
    return emitOutputError(
        function, "declared output buffer must have type !tt.ptr<f32>");
  if (!hasMilestoneModelABI(module))
    return emitOutputError(
        function, "expected distinct buffer roles [input, input, output]");
  if (module->getAttrs().size() != 2 ||
      !function->getDiscardableAttrDictionary().empty() ||
      function.getArgAttrsAttr() || function.getResAttrsAttr())
    return emitOutputError(
        function, "classified bridge preparation has unsupported metadata");

  for (tts::LoadOp load : function.getOps<tts::LoadOp>()) {
    auto pointer = load.getPtr().getDefiningOp<tts::MakeTensorPtrOp>();
    if (pointer && pointer.getBase() == function.getArgument(2))
      return emitOutputError(
          load, "declared write-only output buffer is read, creating "
                "unsupported read-modify-write state");
  }
  if (failed(validateMilestoneOperationAllowlist(
          module, MilestoneOperationStage::ClassifiedBridgePreparation)))
    return failure();

  SmallVector<tts::StoreOp> stores;
  if (failed(requireExactlyOne(
          module, stores,
          "Triton-to-LiteRT structured store is not a single full-tensor "
          "output: expected exactly one tts.store")))
    return failure();
  tts::StoreOp store = stores.front();
  if (store.hasMask() || !store->getDiscardableAttrDictionary().empty()) {
    store.emitError(
        "Triton-to-LiteRT structured access must be unmasked and cover the "
        "full tensor");
    return failure();
  }
  if (!isBridgeIRV1Tensor(store.getValue().getType())) {
    store.emitError(
        "Triton-to-LiteRT structured store is not a single full-tensor "
        "output: expected tensor<1024xf32>");
    return failure();
  }

  auto pointer = store.getPtr().getDefiningOp<tts::MakeTensorPtrOp>();
  if (!pointer)
    return emitOutputError(
        store, "output store must use one supported structured pointer");
  FailureOr<StructuredPointerSemantics> pointerSemantics =
      analyzeMilestoneStructuredPointer(
          pointer, StructuredPointerOffsetForm::StaticZero);
  if (failed(pointerSemantics))
    return failure();

  BlockArgument base = dyn_cast<BlockArgument>(pointer.getBase());
  if (!base || base.getOwner() != &function.getBody().front() ||
      base.getArgNumber() != kOutputBufferArgument)
    return emitOutputError(
        store,
        "output store must target the declared write-only output buffer");
  if (!base.hasOneUse() || *base.user_begin() != pointer.getOperation() ||
      !pointer.getResult().hasOneUse() ||
      *pointer.getResult().user_begin() != store.getOperation())
    return emitOutputError(
        pointer, "output pointer must have one store interpretation and must "
                 "not escape or be read");

  SmallVector<tts::MakeTensorPtrOp> pointers;
  module.walk([&](tts::MakeTensorPtrOp value) { pointers.push_back(value); });
  if (pointers.size() != 1 || pointers.front() != pointer)
    return emitOutputError(function,
                           "exactly one output structured pointer may remain");

  SmallVector<linalg::ElementwiseOp> producers;
  if (failed(requireExactlyOne(
          module, producers,
          "Triton-to-LiteRT structured store requires exactly one classified "
          "elementwise add")))
    return failure();
  linalg::ElementwiseOp producer = producers.front();
  if (producer.getKind() != linalg::ElementwiseKind::add ||
      producer->getNumResults() != 1 ||
      store.getValue() != producer.getResult(0)) {
    store.emitError(
        "Triton-to-LiteRT structured store is not a single full-tensor "
        "output: stored value must be the classified elementwise add result");
    return failure();
  }
  SmallVector<func::ReturnOp> returns;
  if (failed(requireExactlyOne(
          module, returns,
          "Triton-to-LiteRT output functionalization requires exactly one "
          "void return")))
    return failure();
  func::ReturnOp returnOp = returns.front();
  if (returnOp.getNumOperands() != 0 ||
      store->getBlock() != returnOp->getBlock() ||
      store->getNextNode() != returnOp.getOperation())
    return emitOutputError(
        store, "the sole output store must be immediately followed by the "
               "void return with no observable operation after it");
  if (failed(verifyBridgeIRV1ClassifiedAdd(function)))
    return failure();

  return StructuredOutputSemantics{
      *pointerSemantics,
      store,
      store.getValue(),
      cast<RankedTensorType>(store.getValue().getType()),
      kOutputBufferArgument,
      producer,
      returnOp,
      /*masked=*/false,
      /*hasBoundaryCheck=*/false};
}

std::unique_ptr<Pass> createFunctionalizeStructuredOutputPass() {
  return std::make_unique<FunctionalizeStructuredOutputPass>();
}

void registerFunctionalizeStructuredOutputPass() {
  static PassRegistration<FunctionalizeStructuredOutputPass> registration;
}

} // namespace mlir::triton_to_litert
