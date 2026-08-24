//===- ClassifyVectorAddLinalg.cpp - Exact vector-add category -----------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TritonToLiteRT/Bridge.h"

#include "BridgeIR.h"
#include "BridgeInput.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton_to_litert {
namespace {

constexpr StringLiteral kClassificationDiagnostic =
    "Triton-to-LiteRT cannot classify linalg.generic as milestone-1 "
    "elementwise add: ";

LogicalResult reject(linalg::GenericOp generic, const Twine &reason) {
  generic.emitError() << kClassificationDiagnostic << reason;
  return failure();
}

LogicalResult verifyMilestoneVectorAdd(linalg::GenericOp generic) {
  SmallVector<Value> inputs(generic.getInputs());
  SmallVector<Value> outputs(generic.getOutputs());
  if (inputs.size() != 2 || outputs.size() != 1 ||
      generic->getNumResults() != 1)
    return reject(generic,
                  "expected exactly two tensor inputs, one tensor init, and "
                  "one tensor result");

  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != 3 || !llvm::all_of(maps, [](AffineMap map) {
        return map.getNumDims() == 1 && map.getNumResults() == 1 &&
               map.isIdentity();
      }))
    return reject(generic,
                  "expected three exact rank-1 identity indexing maps; "
                  "broadcast and permutation maps are unsupported");

  SmallVector<utils::IteratorType> iterators = generic.getIteratorTypesArray();
  if (iterators.size() != 1 ||
      iterators.front() != utils::IteratorType::parallel)
    return reject(generic,
                  "expected exactly one parallel iterator and no reduction "
                  "iterators");

  if (!llvm::all_of(generic->getOperandTypes(), isBridgeIRV1Tensor) ||
      !llvm::all_of(generic->getResultTypes(), isBridgeIRV1Tensor))
    return reject(
        generic,
        "all operands and results must be exactly tensor<1024xf32> without "
        "an encoding");

  auto function = generic->getParentOfType<func::FuncOp>();
  if (!function || function.getNumArguments() < 2 ||
      inputs[0] != function.getArgument(0) ||
      inputs[1] != function.getArgument(1))
    return reject(generic,
                  "inputs must be the first two model arguments in source "
                  "order, without swapping or repetition");

  if (generic.getDocAttr() || generic.getLibraryCallAttr() ||
      !generic->getDiscardableAttrDictionary().empty())
    return reject(generic,
                  "documentation, library-call, and unversioned operation "
                  "attributes are unsupported");

  Region &region = generic.getRegion();
  if (!region.hasOneBlock())
    return reject(generic, "expected exactly one scalar body block");

  Block &body = region.front();
  if (body.getNumArguments() != 3 ||
      !llvm::all_of(body.getArgumentTypes(),
                    [](Type type) { return type.isF32(); }))
    return reject(generic, "body must have exactly three f32 block arguments");

  if (std::distance(body.begin(), body.end()) != 2)
    return reject(generic,
                  "body must contain only arith.addf followed by "
                  "linalg.yield; extra, nested, or side-effecting operations "
                  "are unsupported");

  auto add = dyn_cast<arith::AddFOp>(body.front());
  if (!add)
    return reject(generic,
                  "body arithmetic must be exactly arith.addf, with no "
                  "alternative arithmetic, calls, casts, or constants");

  if (!body.getArgument(2).use_empty())
    return reject(generic,
                  "the init/output block argument must be unused so the full "
                  "output domain is proven overwritten");

  if (add.getLhs() != body.getArgument(0) ||
      add.getRhs() != body.getArgument(1))
    return reject(generic,
                  "arith.addf must consume the two input block arguments in "
                  "source order");

  if (add.getFastmath() != arith::FastMathFlags::none ||
      add.getRoundingmode().has_value() ||
      !add->getDiscardableAttrDictionary().empty())
    return reject(generic,
                  "arith.addf must have default floating-point semantics; "
                  "fast-math, explicit rounding, and unversioned attributes "
                  "are unsupported");

  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (!yield || yield.getValues().size() != 1 ||
      yield.getValues().front() != add.getResult() ||
      !yield->getDiscardableAttrDictionary().empty())
    return reject(generic,
                  "linalg.yield must yield only the exact arith.addf result");

  return success();
}

LogicalResult classifyVectorAddLinalg(ModuleOp module) {
  SmallVector<linalg::GenericOp> generics;
  module.walk([&](linalg::GenericOp generic) { generics.push_back(generic); });

  if (generics.size() != 1) {
    module.emitError() << kClassificationDiagnostic
                       << "expected exactly one linalg.generic, found "
                       << generics.size();
    return failure();
  }

  linalg::GenericOp generic = generics.front();
  if (failed(verifyMilestoneVectorAdd(generic)))
    return failure();
  if (failed(validateMilestoneOperationAllowlist(
          module, MilestoneOperationStage::FunctionalizedBridgePreparation)))
    return failure();

  // Classification is intentionally atomic: only after every shape, map,
  // iterator, dataflow, and floating-point invariant has been proven do we
  // replace the semantically irrelevant init value with tensor.empty.
  OpBuilder builder(generic);
  auto tensorType = cast<RankedTensorType>(generic.getResult(0).getType());
  Value init = generic.getOutputs().front();
  auto empty = init.getDefiningOp<tensor::EmptyOp>();
  if (!empty)
    empty = tensor::EmptyOp::create(builder, generic.getLoc(),
                                    tensorType.getShape(),
                                    tensorType.getElementType());
  auto kind = linalg::ElementwiseKindAttr::get(module.getContext(),
                                               linalg::ElementwiseKind::add);
  auto indexingMaps =
      builder.getAffineMapArrayAttr(generic.getIndexingMapsArray());
  auto elementwise = linalg::ElementwiseOp::create(
      builder, generic.getLoc(), generic.getInputs(), ValueRange{empty}, kind,
      indexingMaps);

  generic.getResult(0).replaceAllUsesWith(elementwise.getResult(0));
  generic.erase();
  return validateMilestoneOperationAllowlist(
      module, MilestoneOperationStage::ClassifiedBridgePreparation);
}

class ClassifyVectorAddLinalgPass final
    : public PassWrapper<ClassifyVectorAddLinalgPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ClassifyVectorAddLinalgPass)

  StringRef getArgument() const final {
    return "classify-triton-to-litert-vector-add-linalg";
  }

  StringRef getDescription() const final {
    return "Classify the exact milestone-1 linalg.generic vector addition";
  }

  void runOnOperation() final {
    if (failed(classifyVectorAddLinalg(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createClassifyVectorAddLinalgPass() {
  return std::make_unique<ClassifyVectorAddLinalgPass>();
}

void registerClassifyVectorAddLinalgPass() {
  static PassRegistration<ClassifyVectorAddLinalgPass> registration;
}

} // namespace mlir::triton_to_litert
