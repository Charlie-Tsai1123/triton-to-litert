//===- NormalizeLaunchMetadata.cpp - Specialize static Triton launch ------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TritonToLiteRT/Bridge.h"

#include "BridgeInput.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "llvm/ADT/BitVector.h"

namespace mlir::triton_to_litert {
namespace {

LogicalResult emitNormalizationError(Operation *operation,
                                     const Twine &reason) {
  operation->emitError()
      << "Triton-to-LiteRT launch metadata did not fully normalize: " << reason;
  return failure();
}

LogicalResult foldLaunchConstants(func::FuncOp function) {
  RewritePatternSet patterns(function.getContext());
  if (failed(applyPatternsGreedily(function, std::move(patterns)))) {
    function.emitError(
        "Triton-to-LiteRT launch metadata did not fully normalize: "
        "canonicalization did not converge");
    return failure();
  }
  return success();
}

LogicalResult normalizeLaunchMetadata(func::FuncOp function) {
  FailureOr<SmallVector<tts::MakeTensorPtrOp>> pointers =
      analyzeMilestoneLaunchContract(function);
  if (failed(pointers))
    return failure();

  OpBuilder builder(&function.getBody().front(),
                    function.getBody().front().begin());
  for (unsigned index = kModelBufferCount; index < kMilestoneArgumentCount;
       ++index) {
    int64_t value = index < kProgramIdXArgument ? 1 : 0;
    Value constant = arith::ConstantIntOp::create(builder, function.getLoc(),
                                                  value, /*width=*/32);
    function.getArgument(index).replaceAllUsesWith(constant);
  }

  BitVector eraseArguments(kMilestoneArgumentCount);
  eraseArguments.set(kModelBufferCount, kMilestoneArgumentCount);
  FunctionType normalizedType = FunctionType::get(
      function.getContext(),
      function.getArgumentTypes().take_front(kModelBufferCount),
      function.getResultTypes());
  function_interface_impl::eraseFunctionArguments(function, eraseArguments,
                                                  normalizedType);

  if (failed(foldLaunchConstants(function)))
    return failure();

  for (tts::MakeTensorPtrOp pointer : *pointers) {
    if (pointer.getOffsets().size() == 1 &&
        matchPattern(pointer.getOffsets().front(), m_Zero())) {
      pointer.getOffsetsMutable().clear();
      pointer.setStaticOffsets({0});
      continue;
    }
    if (pointer.getOffsets().empty() &&
        pointer.getStaticOffsets() == ArrayRef<int64_t>{0})
      continue;
    (void)emitNormalizationError(
        pointer, "canonical structured offset did not fold to zero");
    return failure();
  }

  if (failed(foldLaunchConstants(function)))
    return failure();

  WalkResult residualLaunchState = function.walk([&](Operation *operation) {
    if (!isa<arith::ConstantOp, arith::MulIOp, arith::IndexCastOp>(operation))
      return WalkResult::advance();
    operation->emitError(
        "Triton-to-LiteRT launch metadata did not fully normalize: residual "
        "launch arithmetic remains");
    return WalkResult::interrupt();
  });
  if (residualLaunchState.wasInterrupted())
    return failure();
  if (function.getNumArguments() != kModelBufferCount) {
    function.emitError(
        "Triton-to-LiteRT launch metadata did not fully normalize: expected "
        "only the three model-buffer arguments");
    return failure();
  }

  return success();
}

class NormalizeLaunchMetadataPass final
    : public PassWrapper<NormalizeLaunchMetadataPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeLaunchMetadataPass)

  StringRef getArgument() const final {
    return "normalize-triton-to-litert-launch-metadata";
  }

  StringRef getDescription() const final {
    return "Normalize the milestone-1 static Triton launch metadata";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (failed(verifyMilestoneBridgeInput(module))) {
      signalPassFailure();
      return;
    }

    func::FuncOp function = *module.getOps<func::FuncOp>().begin();
    Operation *clonedOperation = function->clone();
    auto clonedFunction = cast<func::FuncOp>(clonedOperation);
    if (failed(normalizeLaunchMetadata(clonedFunction))) {
      clonedOperation->destroy();
      signalPassFailure();
      return;
    }

    Block *moduleBody = function->getBlock();
    moduleBody->push_back(clonedOperation);
    clonedOperation->moveBefore(function);
    function.erase();
    module->removeAttr(kLaunchGridAttrName);
  }
};

} // namespace

std::unique_ptr<Pass> createNormalizeLaunchMetadataPass() {
  return std::make_unique<NormalizeLaunchMetadataPass>();
}

void registerNormalizeLaunchMetadataPass() {
  static PassRegistration<NormalizeLaunchMetadataPass> registration;
}

} // namespace mlir::triton_to_litert
