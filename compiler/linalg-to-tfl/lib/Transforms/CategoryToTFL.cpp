//===- CategoryToTFL.cpp - Exact milestone category lowering -------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "linalg-to-tfl/include/LinalgToTFL/Passes.h"

#include <memory>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tflite/converter/ir/tfl_ops.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace mlir::triton_to_litert {
namespace {

constexpr llvm::StringLiteral kBridgeVersionAttr =
    "triton_to_litert.bridge_version";
constexpr llvm::StringLiteral kEntryPointAttr = "triton_to_litert.entry_point";
constexpr int64_t kVectorLength = 1024;

bool isMilestoneTensor(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getRank() == 1 &&
         tensorType.getDimSize(0) == kVectorLength &&
         tensorType.getElementType().isF32() && !tensorType.getEncoding();
}

bool hasOnlyAttrs(Operation *op, llvm::ArrayRef<llvm::StringRef> allowed) {
  for (NamedAttribute attr : op->getAttrs()) {
    if (!llvm::is_contained(allowed, attr.getName().strref()))
      return false;
  }
  return true;
}

LogicalResult emitBridgeError(Operation *op, const Twine &message) {
  op->emitError() << "Triton-to-LiteRT illegal Bridge IR v1 construct '"
                  << op->getName() << "': " << message;
  return failure();
}

LogicalResult emitOutputError(Operation *op, llvm::StringRef construct,
                              const Twine &message) {
  op->emitError() << "Triton-to-LiteRT illegal TFL output construct '"
                  << construct << "': " << message;
  return failure();
}

LogicalResult verifyFunctionEnvelope(func::FuncOp func, StringRef entryPoint) {
  if (func.getName() != entryPoint)
    return emitBridgeError(func, "function name must match entry_point");
  if (!func.isPublic())
    return emitBridgeError(func, "entry function must be public");
  if (!hasOnlyAttrs(func, {"sym_name", "function_type"}))
    return emitBridgeError(func, "entry function has unsupported attributes");
  if (func.getNumArguments() != 2 || func.getNumResults() != 1)
    return emitBridgeError(func, "entry signature must be (tensor<1024xf32>, "
                                 "tensor<1024xf32>) -> tensor<1024xf32>");
  for (Type type : func.getArgumentTypes()) {
    if (!isMilestoneTensor(type))
      return emitBridgeError(func, "both inputs must be tensor<1024xf32>");
  }
  if (!isMilestoneTensor(func.getResultTypes().front()))
    return emitBridgeError(func, "result must be tensor<1024xf32>");
  if (func.getArgAttrsAttr() || func.getResAttrsAttr())
    return emitBridgeError(func,
                           "argument and result attributes are unsupported");
  if (!llvm::hasSingleElement(func.getBody()))
    return emitBridgeError(func, "entry function must contain one block");
  return success();
}

LogicalResult verifyElementwiseAdd(linalg::ElementwiseOp op, func::FuncOp func,
                                   tensor::EmptyOp empty) {
  if (op.getKind() != linalg::ElementwiseKind::add)
    return emitBridgeError(op, "linalg.elementwise kind must be add");
  if (!op->getDiscardableAttrDictionary().empty())
    return emitBridgeError(op, "linalg.elementwise has unsupported attributes");
  SmallVector<Value> inputs(op.getDpsInputs());
  SmallVector<Value> inits(op.getDpsInits());
  if (inputs.size() != 2 || inits.size() != 1 || op.getNumResults() != 1)
    return emitBridgeError(
        op,
        "linalg.elementwise must have two inputs, one init, and one result");
  if (inputs[0] != func.getArgument(0) || inputs[1] != func.getArgument(1))
    return emitBridgeError(op,
                           "linalg.elementwise inputs must preserve ABI order");
  if (inits[0] != empty.getResult())
    return emitBridgeError(op,
                           "linalg.elementwise init must be the tensor.empty");
  for (Value value : inputs) {
    if (!isMilestoneTensor(value.getType()))
      return emitBridgeError(op, "all operands must be tensor<1024xf32>");
  }
  if (!isMilestoneTensor(inits.front().getType()))
    return emitBridgeError(op, "all operands must be tensor<1024xf32>");
  if (!isMilestoneTensor(op.getResult(0).getType()))
    return emitBridgeError(op, "result must be tensor<1024xf32>");

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  AffineMap identity = AffineMap::getMultiDimIdentityMap(1, op.getContext());
  if (maps.size() != 3 ||
      !llvm::all_of(maps, [&](AffineMap map) { return map == identity; }))
    return emitBridgeError(
        op, "all linalg.elementwise indexing maps must be rank-1 identity");
  if (op.getNumLoops() != 1 || op.getNumParallelLoops() != 1)
    return emitBridgeError(op,
                           "linalg.elementwise must have one parallel loop");

  Region &region = op.getRegion();
  if (!llvm::hasSingleElement(region))
    return emitBridgeError(op, "linalg.elementwise must have one region block");
  Block &block = region.front();
  if (block.getNumArguments() != 3 ||
      !llvm::all_of(block.getArgumentTypes(),
                    [](Type type) { return type.isF32(); }))
    return emitBridgeError(
        op, "linalg.elementwise region must have three f32 args");

  SmallVector<Operation *> bodyOps;
  for (Operation &nested : block)
    bodyOps.push_back(&nested);
  if (bodyOps.size() != 2)
    return emitBridgeError(
        op, "linalg.elementwise region must contain only arith.addf and yield");
  auto add = dyn_cast<arith::AddFOp>(bodyOps[0]);
  auto yield = dyn_cast<linalg::YieldOp>(bodyOps[1]);
  if (!add || !yield)
    return emitBridgeError(
        op, "linalg.elementwise region must contain only arith.addf and yield");
  if (add.getLhs() != block.getArgument(0) ||
      add.getRhs() != block.getArgument(1) ||
      add.getFastmath() != arith::FastMathFlags::none ||
      add.getRoundingmode().has_value() ||
      !add->getDiscardableAttrDictionary().empty())
    return emitBridgeError(
        op, "arith.addf must add the two input elements exactly");
  if (yield.getNumOperands() != 1 || yield.getOperand(0) != add.getResult())
    return emitBridgeError(op, "linalg.yield must yield the arith.addf result");
  if (!yield->getDiscardableAttrDictionary().empty())
    return emitBridgeError(op, "linalg.yield attributes are unsupported");
  if (!block.getArgument(2).use_empty())
    return emitBridgeError(op, "output init element must not be read");
  return success();
}

LogicalResult verifyBridgeIR(ModuleOp module) {
  if (!hasOnlyAttrs(module, {kBridgeVersionAttr, kEntryPointAttr}))
    return emitBridgeError(module, "module has unsupported attributes");
  auto version = module->getAttrOfType<IntegerAttr>(kBridgeVersionAttr);
  if (!version || !version.getType().isInteger(32) || version.getInt() != 1) {
    module.emitError("Triton-to-LiteRT unsupported Bridge IR version: "
                     "expected i32 version 1");
    return failure();
  }
  auto entryPoint = module->getAttrOfType<StringAttr>(kEntryPointAttr);
  if (!entryPoint || entryPoint.getValue() != "vector_add") {
    module.emitError("Triton-to-LiteRT invalid Bridge IR entry point: expected "
                     "'vector_add'");
    return failure();
  }

  SmallVector<Operation *> topLevelOps;
  for (Operation &op : module.getBody()->getOperations())
    topLevelOps.push_back(&op);
  if (topLevelOps.size() != 1 || !isa<func::FuncOp>(topLevelOps.front()))
    return emitBridgeError(module, "module must contain exactly one function");
  auto func = cast<func::FuncOp>(topLevelOps.front());
  if (func.getName() != entryPoint.getValue() || !func.isPublic()) {
    module.emitError(
        "Triton-to-LiteRT invalid Bridge IR entry point: expected one public "
        "function named by 'triton_to_litert.entry_point'");
    return failure();
  }
  if (failed(verifyFunctionEnvelope(func, entryPoint.getValue())))
    return failure();

  Block &block = func.front();
  SmallVector<Operation *> ops;
  for (Operation &op : block)
    ops.push_back(&op);
  if (ops.size() != 3)
    return emitBridgeError(
        func,
        "body must contain only tensor.empty, linalg.elementwise, return");
  auto empty = dyn_cast<tensor::EmptyOp>(ops[0]);
  auto elementwise = dyn_cast<linalg::ElementwiseOp>(ops[1]);
  auto ret = dyn_cast<func::ReturnOp>(ops[2]);
  if (!empty)
    return emitBridgeError(ops[0], "expected tensor.empty");
  if (!elementwise)
    return emitBridgeError(ops[1], "expected linalg.elementwise");
  if (!ret)
    return emitBridgeError(ops[2], "expected func.return");
  if (!empty.getDynamicSizes().empty() || !isMilestoneTensor(empty.getType()))
    return emitBridgeError(empty,
                           "tensor.empty must be static tensor<1024xf32>");
  if (!empty->getDiscardableAttrDictionary().empty())
    return emitBridgeError(empty, "tensor.empty attributes are unsupported");
  if (failed(verifyElementwiseAdd(elementwise, func, empty)))
    return failure();
  if (ret.getNumOperands() != 1 ||
      ret.getOperand(0) != elementwise.getResult(0))
    return emitBridgeError(ret,
                           "return must return the linalg.elementwise result");
  if (!ret->getDiscardableAttrDictionary().empty())
    return emitBridgeError(ret, "func.return attributes are unsupported");
  return success();
}

LogicalResult verifyTFLOutput(ModuleOp module) {
  if (!module->getAttrs().empty())
    return emitOutputError(module, "builtin.module",
                           "Bridge IR attributes must not survive");
  SmallVector<func::FuncOp> funcs;
  for (Operation &op : module.getBody()->getOperations()) {
    auto func = dyn_cast<func::FuncOp>(op);
    if (!func)
      return emitOutputError(&op, op.getName().getStringRef(),
                             "only the entry function is legal");
    funcs.push_back(func);
  }
  if (funcs.size() != 1)
    return emitOutputError(module, "builtin.module",
                           "exactly one entry function is required");
  func::FuncOp func = funcs.front();
  if (func.getName() != "vector_add" || !func.isPublic() ||
      !hasOnlyAttrs(func, {"sym_name", "function_type"}) ||
      func.getNumArguments() != 2 || func.getNumResults() != 1 ||
      !isMilestoneTensor(func.getArgumentTypes()[0]) ||
      !isMilestoneTensor(func.getArgumentTypes()[1]) ||
      !isMilestoneTensor(func.getResultTypes()[0]) || func.getArgAttrsAttr() ||
      func.getResAttrsAttr() || !llvm::hasSingleElement(func.getBody()))
    return emitOutputError(func, "func.func",
                           "output function must retain the milestone ABI");

  SmallVector<Operation *> ops;
  for (Operation &op : func.front())
    ops.push_back(&op);
  if (ops.size() != 2)
    return emitOutputError(func, "func.func",
                           "body must contain only tfl.add and return");
  auto add = dyn_cast<TFL::AddOp>(ops[0]);
  auto ret = dyn_cast<func::ReturnOp>(ops[1]);
  if (!add)
    return emitOutputError(ops[0], ops[0]->getName().getStringRef(),
                           "only tfl.add is supported");
  if (!ret)
    return emitOutputError(ops[1], ops[1]->getName().getStringRef(),
                           "function must terminate with func.return");
  if (add.getLhs() != func.getArgument(0) ||
      add.getRhs() != func.getArgument(1) ||
      !isMilestoneTensor(add.getOutput().getType()))
    return emitOutputError(add, "tfl.add",
                           "operands/result must preserve the milestone ABI");
  if (add.getFusedActivationFunction() != "NONE")
    return emitOutputError(add, "tfl.add", "fused activation must be NONE");
  if (!add->getDiscardableAttrDictionary().empty())
    return emitOutputError(add, "tfl.add",
                           "unversioned attributes are unsupported");
  if (ret.getNumOperands() != 1 || ret.getOperand(0) != add.getOutput())
    return emitOutputError(ret, "func.return",
                           "must return the tfl.add result");
  if (!ret->getDiscardableAttrDictionary().empty())
    return emitOutputError(ret, "func.return",
                           "unversioned attributes are unsupported");
  return success();
}

class ConvertElementwiseAdd final
    : public OpConversionPattern<linalg::ElementwiseOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::ElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getKind() != linalg::ElementwiseKind::add ||
        adaptor.getInputs().size() != 2 || op.getNumResults() != 1 ||
        !isMilestoneTensor(op.getResult(0).getType()))
      return rewriter.notifyMatchFailure(
          op, "not the exact milestone-1 elementwise add");
    auto add = TFL::AddOp::create(
        rewriter, op.getLoc(), op.getResult(0).getType(),
        adaptor.getInputs()[0], adaptor.getInputs()[1], "NONE");
    rewriter.replaceOp(op, add.getOutput());
    return success();
  }
};

class BridgeToTFLPass final
    : public PassWrapper<BridgeToTFLPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BridgeToTFLPass)

  StringRef getArgument() const final {
    return "triton-to-litert-bridge-to-tfl";
  }
  StringRef getDescription() const final {
    return "Lower exact milestone-1 Bridge IR vector add to TFL";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, linalg::LinalgDialect,
                tensor::TensorDialect, TFL::TensorFlowLiteDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (failed(verifyBridgeIR(module))) {
      signalPassFailure();
      return;
    }

    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp, tensor::EmptyOp,
                      TFL::AddOp>();
    target.addIllegalDialect<linalg::LinalgDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return false; });
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertElementwiseAdd>(&getContext());
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      module.emitError()
          << "Triton-to-LiteRT conversion failed closed: unsupported operation";
      signalPassFailure();
      return;
    }

    SmallVector<tensor::EmptyOp> empties;
    module.walk([&](tensor::EmptyOp empty) { empties.push_back(empty); });
    for (tensor::EmptyOp empty : empties) {
      if (!empty->use_empty()) {
        (void)emitOutputError(empty, "tensor.empty",
                              "unexpected live tensor initialization");
        signalPassFailure();
        return;
      }
      empty.erase();
    }
    module->removeAttr(kBridgeVersionAttr);
    module->removeAttr(kEntryPointAttr);
    if (failed(verifyTFLOutput(module)) || failed(verify(module)))
      signalPassFailure();
  }
};

class VerifyTFLOutputPass final
    : public PassWrapper<VerifyTFLOutputPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyTFLOutputPass)

  StringRef getArgument() const final {
    return "verify-triton-to-litert-tfl-output";
  }
  StringRef getDescription() const final {
    return "Verify the closed-world milestone-1 TFL output contract";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, TFL::TensorFlowLiteDialect>();
  }
  void runOnOperation() override {
    if (failed(verifyTFLOutput(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createBridgeToTFLPass() {
  return std::make_unique<BridgeToTFLPass>();
}

std::unique_ptr<Pass> createVerifyTFLOutputPass() {
  return std::make_unique<VerifyTFLOutputPass>();
}

void registerBridgeToTFLPasses() {
  PassRegistration<BridgeToTFLPass>();
  PassRegistration<VerifyTFLOutputPass>();
}

} // namespace mlir::triton_to_litert
