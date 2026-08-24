//===- VerifyBridgeIR.cpp - Milestone 1 Bridge IR verifier ---------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TritonToLiteRT/Bridge.h"

#include "BridgeIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::triton_to_litert {

bool isBridgeIRV1Tensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.hasStaticShape() && tensor.getRank() == 1 &&
         tensor.getShape() == ArrayRef<int64_t>{kBridgeIRV1Extent} &&
         tensor.getElementType().isF32() && !tensor.getEncoding();
}

namespace {

LogicalResult emitIllegalBridgeConstruct(Operation *operation,
                                         const Twine &construct,
                                         const Twine &invariant) {
  operation->emitError() << "Triton-to-LiteRT illegal Bridge IR v1 construct '"
                         << construct << "': " << invariant;
  return failure();
}

LogicalResult verifyBridgeMetadata(ModuleOp module,
                                   func::FuncOp &entryFunction) {
  auto version = module->getAttrOfType<IntegerAttr>(kBridgeVersionAttrName);
  if (!version || !version.getType().isInteger(32) ||
      version.getInt() != kBridgeIRVersion) {
    module.emitError("Triton-to-LiteRT unsupported Bridge IR version: "
                     "expected i32 version 1");
    return failure();
  }

  auto entryPoint =
      module->getAttrOfType<StringAttr>(kBridgeEntryPointAttrName);
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  if (!entryPoint || functions.size() != 1 ||
      functions.front().getSymName() != entryPoint.getValue() ||
      !functions.front().isPublic()) {
    module.emitError(
        "Triton-to-LiteRT invalid Bridge IR entry point: expected one public "
        "function named by 'triton_to_litert.entry_point'");
    return failure();
  }

  for (NamedAttribute attribute : module->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (name != kBridgeVersionAttrName && name != kBridgeEntryPointAttrName)
      return emitIllegalBridgeConstruct(
          module, name,
          "module attributes are closed to version and entry "
          "point metadata");
  }

  if (!llvm::hasSingleElement(module.getBody()->getOperations()))
    return emitIllegalBridgeConstruct(
        module, "extra module operation",
        "Bridge IR v1 contains exactly one entry function");

  entryFunction = functions.front();
  return success();
}

LogicalResult verifyElementwiseAdd(linalg::ElementwiseOp elementwise,
                                   func::FuncOp function,
                                   tensor::EmptyOp empty) {
  SmallVector<Value> inputs(elementwise.getInputs());
  SmallVector<Value> outputs(elementwise.getOutputs());
  if (elementwise.getKind() != linalg::ElementwiseKind::add ||
      inputs.size() != 2 || outputs.size() != 1 ||
      elementwise->getNumResults() != 1 ||
      !llvm::all_of(elementwise->getOperandTypes(), isBridgeIRV1Tensor) ||
      !llvm::all_of(elementwise->getResultTypes(), isBridgeIRV1Tensor))
    return emitIllegalBridgeConstruct(
        elementwise, "linalg.elementwise",
        "expected one same-shaped binary f32 add with one tensor result");

  if (inputs[0] != function.getArgument(0) ||
      inputs[1] != function.getArgument(1) || outputs[0] != empty.getResult())
    return emitIllegalBridgeConstruct(
        elementwise, "linalg.elementwise",
        "operands must preserve ABI input order and use tensor.empty as the "
        "destination");

  SmallVector<AffineMap> indexingMaps = elementwise.getIndexingMapsArray();
  if (indexingMaps.size() != 3 ||
      !llvm::all_of(indexingMaps, [](AffineMap map) {
        return map.getNumDims() == 1 && map.getNumResults() == 1 &&
               map.isIdentity();
      }))
    return emitIllegalBridgeConstruct(
        elementwise, "linalg.elementwise",
        "all milestone-1 indexing maps must be rank-1 identity maps");

  Region &region = elementwise.getRegion();
  if (!region.hasOneBlock())
    return emitIllegalBridgeConstruct(
        elementwise, "linalg.elementwise",
        "the category operation must have its canonical scalar region");

  Block &body = region.front();
  if (body.getNumArguments() != 3 ||
      !llvm::all_of(body.getArgumentTypes(),
                    [](Type type) { return type.isF32(); }) ||
      std::distance(body.begin(), body.end()) != 2)
    return emitIllegalBridgeConstruct(
        elementwise, "linalg.elementwise",
        "the category operation must have its canonical scalar region");

  auto add = dyn_cast<arith::AddFOp>(body.front());
  auto yield = dyn_cast<linalg::YieldOp>(body.back());
  if (!add || !yield || add.getLhs() != body.getArgument(0) ||
      add.getRhs() != body.getArgument(1) || !body.getArgument(2).use_empty() ||
      add.getFastmath() != arith::FastMathFlags::none ||
      add.getRoundingmode().has_value() ||
      !add->getDiscardableAttrDictionary().empty() ||
      yield.getValues().size() != 1 ||
      yield.getValues().front() != add.getResult() ||
      !yield->getDiscardableAttrDictionary().empty())
    return emitIllegalBridgeConstruct(
        elementwise, "linalg.elementwise",
        "the category operation must have its canonical scalar add region");

  return success();
}

LogicalResult verifyEntryFunction(func::FuncOp function) {
  FunctionType type = function.getFunctionType();
  if (type.getNumInputs() != 2 || type.getNumResults() != 1 ||
      !llvm::all_of(type.getInputs(), isBridgeIRV1Tensor) ||
      !llvm::all_of(type.getResults(), isBridgeIRV1Tensor) ||
      !function.getBody().hasOneBlock())
    return emitIllegalBridgeConstruct(
        function, "func.func",
        "entry signature must be (tensor<1024xf32>, tensor<1024xf32>) -> "
        "tensor<1024xf32> with one block");

  if (!function->getDiscardableAttrDictionary().empty())
    return emitIllegalBridgeConstruct(
        function, "func.func attribute",
        "Bridge IR v1 does not permit unversioned function attributes");
  if (function.getArgAttrsAttr() || function.getResAttrsAttr())
    return emitIllegalBridgeConstruct(
        function, "func.func argument/result attribute",
        "Bridge IR v1 tensor ABI does not permit argument or result "
        "attributes");

  func::ReturnOp returnOp;
  for (Operation &operation : function.getBody().front()) {
    if (isa<tensor::EmptyOp, linalg::ElementwiseOp>(operation)) {
      continue;
    } else if (auto candidate = dyn_cast<func::ReturnOp>(operation)) {
      if (returnOp)
        return emitIllegalBridgeConstruct(
            candidate, "func.return",
            "Bridge IR v1 requires exactly one return");
      returnOp = candidate;
    } else {
      return emitIllegalBridgeConstruct(
          &operation, operation.getName().getStringRef(),
          "operation is outside the closed Bridge IR v1 allowlist");
    }
  }

  if (failed(verifyBridgeIRV1ClassifiedAdd(function)))
    return failure();
  linalg::ElementwiseOp elementwise =
      *function.getOps<linalg::ElementwiseOp>().begin();

  if (!returnOp || returnOp.getNumOperands() != 1 ||
      returnOp.getOperand(0) != elementwise->getResult(0) ||
      !returnOp->getDiscardableAttrDictionary().empty())
    return emitIllegalBridgeConstruct(
        function, "func.return",
        "entry point must return the sole elementwise-add result");

  return success();
}

class VerifyBridgeIRV1Pass final
    : public PassWrapper<VerifyBridgeIRV1Pass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyBridgeIRV1Pass)

  StringRef getArgument() const final {
    return "verify-triton-to-litert-bridge-ir-v1";
  }

  StringRef getDescription() const final {
    return "Verify the closed milestone-1 textual Bridge IR v1 contract";
  }

  void runOnOperation() final {
    if (failed(verifyBridgeIRV1(getOperation())))
      signalPassFailure();
  }
};

} // namespace

LogicalResult verifyBridgeIRV1(ModuleOp module) {
  func::FuncOp entryFunction;
  if (failed(verifyBridgeMetadata(module, entryFunction)))
    return failure();
  return verifyEntryFunction(entryFunction);
}

LogicalResult verifyBridgeIRV1ClassifiedAdd(func::FuncOp function) {
  tensor::EmptyOp empty;
  linalg::ElementwiseOp elementwise;
  for (Operation &operation : function.getBody().front()) {
    if (auto candidate = dyn_cast<tensor::EmptyOp>(operation)) {
      if (empty)
        return emitIllegalBridgeConstruct(
            candidate, "tensor.empty",
            "Bridge IR v1 requires exactly one tensor.empty");
      empty = candidate;
    } else if (auto candidate = dyn_cast<linalg::ElementwiseOp>(operation)) {
      if (elementwise)
        return emitIllegalBridgeConstruct(
            candidate, "linalg.elementwise",
            "Bridge IR v1 requires exactly one elementwise add");
      elementwise = candidate;
    }
  }

  if (!empty || !isBridgeIRV1Tensor(empty.getType()) ||
      !empty.getDynamicSizes().empty() ||
      !empty->getDiscardableAttrDictionary().empty())
    return emitIllegalBridgeConstruct(
        function, "tensor.empty",
        "expected one static tensor.empty of tensor<1024xf32>");
  if (!elementwise)
    return emitIllegalBridgeConstruct(
        function, "linalg.elementwise",
        "expected one milestone-1 elementwise add");
  if (!elementwise->getDiscardableAttrDictionary().empty())
    return emitIllegalBridgeConstruct(
        elementwise, "linalg.elementwise attribute",
        "Bridge IR v1 does not permit unversioned operation attributes");
  return verifyElementwiseAdd(elementwise, function, empty);
}

std::unique_ptr<Pass> createVerifyBridgeIRV1Pass() {
  return std::make_unique<VerifyBridgeIRV1Pass>();
}

void registerVerifyBridgeIRV1Pass() {
  static PassRegistration<VerifyBridgeIRV1Pass> registration;
}

} // namespace mlir::triton_to_litert
