//===- VerifyBridgeInput.cpp - Milestone 1 Bridge Input gate -------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TritonToLiteRT/Bridge.h"

#include "BridgeInput.h"
#include "StructuredInput.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"

#include <array>
#include <iterator>
#include <optional>

namespace mlir::triton_to_litert {
namespace {

bool hasStaticUnitLaunch(ModuleOp module) {
  auto launchGrid =
      module->getAttrOfType<DenseI64ArrayAttr>(kLaunchGridAttrName);
  return launchGrid && launchGrid.size() == kLaunchDimensionCount &&
         llvm::all_of(launchGrid.asArrayRef(),
                      [](int64_t extent) { return extent == 1; });
}

bool isAllowedMilestoneOperation(StringRef operationName,
                                 MilestoneOperationStage stage) {
  bool isBridgeInput = stage == MilestoneOperationStage::BridgeInput;
  bool isClassified =
      stage == MilestoneOperationStage::ClassifiedBridgePreparation;
  bool hasStructuredInputLoads =
      isBridgeInput ||
      stage == MilestoneOperationStage::NormalizedBridgePreparation;
  return llvm::StringSwitch<bool>(operationName)
      .Cases({"builtin.module", "func.func", "func.return"}, true)
      .Case("arith.addf", true)
      .Cases({"arith.constant", "arith.muli", "arith.index_cast"},
             isBridgeInput)
      .Cases({"tensor.empty", "linalg.yield", "tts.make_tptr", "tts.store"},
             true)
      .Case("linalg.generic", !isClassified)
      .Case("linalg.elementwise", isClassified)
      .Case("tts.load", hasStructuredInputLoads)
      .Default(false);
}

StringRef getOperationStageName(MilestoneOperationStage stage) {
  switch (stage) {
  case MilestoneOperationStage::BridgeInput:
    return "Bridge Input";
  case MilestoneOperationStage::NormalizedBridgePreparation:
    return "normalized bridge preparation";
  case MilestoneOperationStage::FunctionalizedBridgePreparation:
    return "functionalized bridge preparation";
  case MilestoneOperationStage::ClassifiedBridgePreparation:
    return "classified bridge preparation";
  }
  llvm_unreachable("unknown milestone operation stage");
}

bool isUnstructuredMemoryOperation(StringRef operationName) {
  return llvm::StringSwitch<bool>(operationName)
      .Cases({"tts.make_gather_scatter_tptr", "tts.gather", "tts.scatter",
              "tts.unstructured_atomic_rmw"},
             true)
      .Default(false);
}

bool hasAllowedMilestonePlacement(Operation *operation,
                                  MilestoneOperationStage stage) {
  StringRef name = operation->getName().getStringRef();
  if (name != "arith.addf" && name != "linalg.yield")
    return true;

  Operation *parent = operation->getParentOp();
  if (stage == MilestoneOperationStage::ClassifiedBridgePreparation)
    return isa_and_nonnull<linalg::ElementwiseOp>(parent);
  return isa_and_nonnull<linalg::GenericOp>(parent);
}

LogicalResult validateOperationAllowlistImpl(ModuleOp module,
                                             MilestoneOperationStage stage) {
  WalkResult result = module.walk([&](Operation *operation) {
    StringRef operationName = operation->getName().getStringRef();
    if (isAllowedMilestoneOperation(operationName, stage) &&
        hasAllowedMilestonePlacement(operation, stage))
      return WalkResult::advance();

    if (stage != MilestoneOperationStage::BridgeInput) {
      operation->emitError()
          << "Triton-to-LiteRT " << getOperationStageName(stage)
          << " does not support operation '" << operationName << "'";
      return WalkResult::interrupt();
    }

    StringRef dialectNamespace = operation->getName().getDialectNamespace();
    if (operationName == "arith.trunci" || operationName == "arith.truncf") {
      operation->emitError()
          << "Triton-to-LiteRT semantic narrowing is unsupported at Bridge "
             "Input ('"
          << operationName << "')";
    } else if (dialectNamespace == "tt") {
      operation->emitError()
          << "Triton-to-LiteRT bridge input contains residual Triton "
             "operation '"
          << operationName << "'";
    } else if (isUnstructuredMemoryOperation(operationName)) {
      operation->emitError()
          << "Triton-to-LiteRT Bridge Input milestone 1 does not support "
             "unstructured memory operation '"
          << operationName << "'";
    } else if (dialectNamespace == "tts") {
      operation->emitError()
          << "Triton-to-LiteRT Bridge Input milestone 1 does not support TTS "
             "operation '"
          << operationName << "'";
    } else {
      operation->emitError()
          << "Triton-to-LiteRT Bridge Input milestone 1 does not support "
             "operation '"
          << operationName << "'";
    }
    return WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

bool isConstantInteger(Value value, int64_t expected) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) &&
         constant.getSExtValue() == expected;
}

LogicalResult validateEntryBlockPlacement(Operation *operation,
                                          const Twine &semanticClass) {
  auto function = operation->getParentOfType<func::FuncOp>();
  if (function && !function.getBody().empty() &&
      operation->getBlock() == &function.getBody().front())
    return success();

  operation->emitError() << "Triton-to-LiteRT Bridge Input " << semanticClass
                         << " operation '" << operation->getName()
                         << "' must be in the function entry block";
  return failure();
}

LogicalResult validateStructuredPointer(tts::MakeTensorPtrOp operation) {
  return success(succeeded(analyzeMilestoneStructuredPointer(
      operation, StructuredPointerOffsetForm::CanonicalLaunchExpression)));
}

LogicalResult validateStructuredLoad(tts::LoadOp operation) {
  return success(succeeded(analyzeMilestoneStructuredInputLoad(
      operation, StructuredPointerOffsetForm::CanonicalLaunchExpression)));
}

LogicalResult validateStructuredStore(tts::StoreOp operation) {
  if (operation.hasMask()) {
    operation.emitError(
        "Triton-to-LiteRT Bridge Input structured access must be unmasked "
        "and cover the full tensor");
    return failure();
  }

  if (!isMilestoneTensor(operation.getValue().getType())) {
    operation.emitError(
        "Triton-to-LiteRT Bridge Input structured store is not a single "
        "full-tensor output: expected tensor<1024xf32>");
    return failure();
  }

  auto pointer = operation.getPtr().getDefiningOp<tts::MakeTensorPtrOp>();
  auto function = operation->getParentOfType<func::FuncOp>();
  if (!pointer || !function || function.getNumArguments() < kModelBufferCount ||
      pointer.getBase() != function.getArgument(kOutputBufferArgument)) {
    operation.emitError(
        "Triton-to-LiteRT Bridge Input model ABI cannot be functionalized: "
        "output store must use the declared output buffer");
    return failure();
  }
  return success();
}

LogicalResult validateTensorEmpty(tensor::EmptyOp operation) {
  if (!isMilestoneTensor(operation.getType())) {
    operation.emitError(
        "Triton-to-LiteRT Bridge Input rank-1 f32 tensors must have static "
        "extent 1024");
    return failure();
  }
  return success();
}

LogicalResult validateFunctionSignature(func::FuncOp function) {
  FunctionType type = function.getFunctionType();
  if (type.getNumInputs() != kMilestoneArgumentCount ||
      type.getNumResults() != 0 || !function.getBody().hasOneBlock()) {
    function.emitError(
        "Triton-to-LiteRT Bridge Input milestone 1 requires three model "
        "buffers followed by six i32 launch arguments");
    return failure();
  }

  for (Type pointerType : type.getInputs().take_front(kModelBufferCount)) {
    auto pointer = dyn_cast<triton::PointerType>(pointerType);
    if (!pointer || !pointer.getPointeeType().isF32()) {
      function.emitError(
          "Triton-to-LiteRT Bridge Input milestone 1 requires three model "
          "buffers followed by six i32 launch arguments");
      return failure();
    }
  }
  if (!llvm::all_of(type.getInputs().drop_front(kModelBufferCount),
                    [](Type type) { return type.isInteger(32); })) {
    function.emitError(
        "Triton-to-LiteRT Bridge Input milestone 1 requires three model "
        "buffers followed by six i32 launch arguments");
    return failure();
  }
  return success();
}

LogicalResult
validateCandidateGeneric(linalg::GenericOp generic, ArrayRef<tts::LoadOp> loads,
                         MutableArrayRef<tensor::EmptyOp> empties) {
  if (generic.getInputs().size() != kInputBufferCount ||
      generic.getOutputs().size() != 1 || generic.getNumResults() != 1) {
    generic.emitError("Triton-to-LiteRT Bridge Input requires one candidate "
                      "linalg.generic with two inputs and one output");
    return failure();
  }

  if (!llvm::all_of(generic->getOperandTypes(), isMilestoneTensor) ||
      !llvm::all_of(generic->getResultTypes(), isMilestoneTensor)) {
    generic.emitError(
        "Triton-to-LiteRT Bridge Input rank-1 f32 tensors must have static "
        "extent 1024");
    return failure();
  }

  if (loads.size() != kInputBufferCount ||
      !llvm::is_contained(generic.getInputs(), loads[0]->getResult(0)) ||
      !llvm::is_contained(generic.getInputs(), loads[1]->getResult(0))) {
    generic.emitError(
        "Triton-to-LiteRT Bridge Input model ABI cannot be functionalized: "
        "linalg.generic inputs must be the two declared input loads");
    return failure();
  }

  if (empties.size() > 1 ||
      (empties.size() == 1 &&
       !llvm::is_contained(generic.getOutputs(),
                           empties.front().getResult()))) {
    Operation *anchor = empties.size() > 1 ? empties[1].getOperation()
                                           : empties.front().getOperation();
    anchor->emitError(
        "Triton-to-LiteRT Bridge Input tensor.empty is permitted only as "
        "the candidate linalg.generic output initializer");
    return failure();
  }

  return success();
}

FailureOr<SmallVector<tts::MakeTensorPtrOp>>
analyzeMilestoneLaunchContractImpl(func::FuncOp function) {
  SmallVector<arith::ConstantOp> constants;
  SmallVector<arith::MulIOp> multiplies;
  SmallVector<arith::IndexCastOp> casts;
  function.walk([&](Operation *operation) {
    if (auto constant = dyn_cast<arith::ConstantOp>(operation))
      constants.push_back(constant);
    else if (auto multiply = dyn_cast<arith::MulIOp>(operation))
      multiplies.push_back(multiply);
    else if (auto cast = dyn_cast<arith::IndexCastOp>(operation))
      casts.push_back(cast);
  });

  if (constants.size() != 1 || !constants.front().getType().isInteger(32) ||
      !isConstantInteger(constants.front().getResult(), kMilestoneExtent)) {
    Operation *anchor = constants.size() > 1 ? constants[1].getOperation()
                                             : function.getOperation();
    anchor->emitError(
        "Triton-to-LiteRT Bridge Input milestone 1 permits only the "
        "canonical i32 constant 1024 used by the launch offset");
    return failure();
  }

  arith::ConstantOp constant = constants.front();
  if (multiplies.size() != 1 || !multiplies.front().getType().isInteger(32)) {
    Operation *anchor = multiplies.size() > 1 ? multiplies[1].getOperation()
                                              : function.getOperation();
    anchor->emitError(
        "Triton-to-LiteRT Bridge Input milestone 1 requires exactly one "
        "canonical program_id.x times 1024 launch expression");
    return failure();
  }

  arith::MulIOp multiply = multiplies.front();
  Value programIdX = function.getArgument(kProgramIdXArgument);
  bool hasCanonicalOperands = (multiply.getLhs() == programIdX &&
                               multiply.getRhs() == constant.getResult()) ||
                              (multiply.getRhs() == programIdX &&
                               multiply.getLhs() == constant.getResult());
  if (!hasCanonicalOperands || !constant.getResult().hasOneUse()) {
    multiply.emitError(
        "Triton-to-LiteRT Bridge Input milestone 1 requires exactly one "
        "canonical program_id.x times 1024 launch expression");
    return failure();
  }

  if (casts.size() != 1 || casts.front().getIn() != multiply.getResult() ||
      !multiply.getResult().hasOneUse() ||
      !casts.front().getIn().getType().isInteger(32) ||
      !casts.front().getOut().getType().isIndex()) {
    Operation *anchor =
        casts.size() > 1 ? casts[1].getOperation() : function.getOperation();
    anchor->emitError(
        "Triton-to-LiteRT Bridge Input milestone 1 requires the canonical "
        "launch offset to have one i32-to-index cast");
    return failure();
  }

  arith::IndexCastOp cast = casts.front();
  SmallVector<tts::MakeTensorPtrOp> pointers;
  for (Operation *user : cast.getResult().getUsers()) {
    auto pointer = dyn_cast<tts::MakeTensorPtrOp>(user);
    if (!pointer) {
      cast.emitError(
          "Triton-to-LiteRT Bridge Input milestone 1 requires the canonical "
          "launch offset to feed exactly three structured pointers");
      return failure();
    }
    pointers.push_back(pointer);
  }
  if (pointers.size() != kModelBufferCount) {
    cast.emitError(
        "Triton-to-LiteRT Bridge Input milestone 1 requires the canonical "
        "launch offset to feed exactly three structured pointers");
    return failure();
  }

  return pointers;
}

LogicalResult validateMilestoneProgram(func::FuncOp function) {
  SmallVector<tts::MakeTensorPtrOp> pointers;
  SmallVector<tts::LoadOp> loads;
  SmallVector<tts::StoreOp> stores;
  SmallVector<linalg::GenericOp> generics;
  SmallVector<tensor::EmptyOp> empties;
  for (Operation &operation : function.getBody().front()) {
    if (auto pointer = dyn_cast<tts::MakeTensorPtrOp>(operation))
      pointers.push_back(pointer);
    else if (auto load = dyn_cast<tts::LoadOp>(operation))
      loads.push_back(load);
    else if (auto store = dyn_cast<tts::StoreOp>(operation))
      stores.push_back(store);
    else if (auto generic = dyn_cast<linalg::GenericOp>(operation))
      generics.push_back(generic);
    else if (auto empty = dyn_cast<tensor::EmptyOp>(operation))
      empties.push_back(empty);
  }

  if (stores.size() != 1) {
    function.emitError(
        "Triton-to-LiteRT Bridge Input structured store is not a single "
        "full-tensor output: expected exactly one tts.store");
    return failure();
  }
  if (pointers.size() != kModelBufferCount ||
      loads.size() != kInputBufferCount) {
    function.emitError(
        "Triton-to-LiteRT Bridge Input model ABI cannot be functionalized: "
        "expected three structured pointers and two input loads");
    return failure();
  }
  if (generics.size() != 1) {
    function.emitError(
        "Triton-to-LiteRT Bridge Input requires exactly one candidate "
        "linalg.generic");
    return failure();
  }
  if (failed(validateCandidateGeneric(generics.front(), loads, empties)))
    return failure();

  linalg::GenericOp generic = generics.front();
  tts::StoreOp store = stores.front();
  if (store.getValue() != generic->getResult(0)) {
    store.emitError(
        "Triton-to-LiteRT Bridge Input structured store is not a single "
        "full-tensor output: stored value must be the candidate generic "
        "result");
    return failure();
  }

  for (tts::LoadOp load : loads) {
    if (!load->isBeforeInBlock(generic)) {
      load.emitError(
          "Triton-to-LiteRT Bridge Input model ABI cannot be functionalized: "
          "both input loads must precede linalg.generic");
      return failure();
    }
  }
  if (!generic->isBeforeInBlock(store)) {
    store.emitError(
        "Triton-to-LiteRT Bridge Input model ABI cannot be functionalized: "
        "the output store must follow linalg.generic");
    return failure();
  }

  for (unsigned index = 0; index < kModelBufferCount; ++index) {
    BlockArgument base = function.getArgument(index);
    if (!base.hasOneUse() || !isa<tts::MakeTensorPtrOp>(*base.user_begin())) {
      function.emitError(
          "Triton-to-LiteRT Bridge Input model ABI cannot be functionalized: "
          "each model buffer must have exactly one structured pointer use");
      return failure();
    }
  }
  for (tts::MakeTensorPtrOp pointer : pointers) {
    if (!pointer.getResult().hasOneUse() ||
        !isa<tts::LoadOp, tts::StoreOp>(*pointer.getResult().user_begin())) {
      pointer.emitError(
          "Triton-to-LiteRT Bridge Input model ABI cannot be functionalized: "
          "structured pointers may only feed their single declared access");
      return failure();
    }
  }

  return success();
}

std::optional<Type> findIllegalType(Type type) {
  if (isa<BaseMemRefType>(type))
    return type;

  if (auto pointerType = dyn_cast<triton::PointerType>(type))
    return findIllegalType(pointerType.getPointeeType());

  StringRef dialectNamespace = type.getDialect().getNamespace();
  if (dialectNamespace != "builtin")
    return type;

  if (auto functionType = dyn_cast<FunctionType>(type)) {
    for (Type input : functionType.getInputs())
      if (std::optional<Type> illegalType = findIllegalType(input))
        return illegalType;
    for (Type result : functionType.getResults())
      if (std::optional<Type> illegalType = findIllegalType(result))
        return illegalType;
  }

  if (auto tupleType = dyn_cast<TupleType>(type)) {
    for (Type elementType : tupleType.getTypes())
      if (std::optional<Type> illegalType = findIllegalType(elementType))
        return illegalType;
  }

  if (auto shapedType = dyn_cast<ShapedType>(type))
    return findIllegalType(shapedType.getElementType());

  if (auto complexType = dyn_cast<ComplexType>(type))
    return findIllegalType(complexType.getElementType());

  return std::nullopt;
}

std::optional<Type> findIllegalType(Attribute attribute) {
  if (auto typeAttribute = dyn_cast<TypeAttr>(attribute))
    return findIllegalType(typeAttribute.getValue());

  if (auto arrayAttribute = dyn_cast<ArrayAttr>(attribute)) {
    for (Attribute element : arrayAttribute)
      if (std::optional<Type> illegalType = findIllegalType(element))
        return illegalType;
  }

  if (auto dictionaryAttribute = dyn_cast<DictionaryAttr>(attribute)) {
    for (NamedAttribute element : dictionaryAttribute)
      if (std::optional<Type> illegalType = findIllegalType(element.getValue()))
        return illegalType;
  }

  return std::nullopt;
}

class VerifyBridgeInputPass final
    : public PassWrapper<VerifyBridgeInputPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifyBridgeInputPass)

  StringRef getArgument() const final {
    return "verify-triton-to-litert-bridge-input";
  }

  StringRef getDescription() const final {
    return "Verify the milestone-1 Triton-to-LiteRT Bridge Input boundary";
  }

  void runOnOperation() final {
    if (failed(verifyMilestoneBridgeInput(getOperation())))
      signalPassFailure();
  }
};

} // namespace

LogicalResult verifyMilestoneBridgeInput(ModuleOp module) {
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  if (functions.size() != 1 || !hasStaticUnitLaunch(module)) {
    Operation *diagnosticAnchor = functions.empty()
                                      ? module.getOperation()
                                      : functions.front().getOperation();
    diagnosticAnchor->emitError(
        "Triton-to-LiteRT Bridge Input milestone 1 requires a static "
        "launch grid of (1, 1, 1)");
    return failure();
  }

  if (!hasMilestoneModelABI(module)) {
    functions.front()->emitError(
        "Triton-to-LiteRT Bridge Input model ABI cannot be functionalized: "
        "expected distinct buffer roles [input, input, output]");
    return failure();
  }

  if (failed(validateOperationAllowlistImpl(
          module, MilestoneOperationStage::BridgeInput)))
    return failure();

  WalkResult result = module.walk([&](Operation *operation) {
    if (auto pointer = dyn_cast<tts::MakeTensorPtrOp>(operation)) {
      if (failed(validateEntryBlockPlacement(pointer, "structured memory")) ||
          failed(validateStructuredPointer(pointer)))
        return WalkResult::interrupt();
    } else if (auto load = dyn_cast<tts::LoadOp>(operation)) {
      if (failed(validateEntryBlockPlacement(load, "structured memory")) ||
          failed(validateStructuredLoad(load)))
        return WalkResult::interrupt();
    } else if (auto store = dyn_cast<tts::StoreOp>(operation)) {
      if (failed(validateEntryBlockPlacement(store, "structured memory")) ||
          failed(validateStructuredStore(store)))
        return WalkResult::interrupt();
    } else if (auto empty = dyn_cast<tensor::EmptyOp>(operation)) {
      if (failed(validateEntryBlockPlacement(empty, "tensor scaffolding")) ||
          failed(validateTensorEmpty(empty)))
        return WalkResult::interrupt();
    } else if (auto generic = dyn_cast<linalg::GenericOp>(operation)) {
      if (failed(validateEntryBlockPlacement(generic, "candidate computation")))
        return WalkResult::interrupt();
    } else if (auto add = dyn_cast<arith::AddFOp>(operation)) {
      if (!add->getParentOfType<linalg::GenericOp>() ||
          !add.getType().isF32()) {
        add.emitError("Triton-to-LiteRT Bridge Input only permits scalar f32 "
                      "arith.addf inside the candidate linalg.generic");
        return WalkResult::interrupt();
      }
    } else if (auto cast = dyn_cast<arith::IndexCastOp>(operation)) {
      if (failed(validateEntryBlockPlacement(cast, "launch arithmetic")))
        return WalkResult::interrupt();
      if (!cast.getIn().getType().isInteger(32) ||
          !cast.getOut().getType().isIndex()) {
        cast.emitError("Triton-to-LiteRT semantic narrowing is unsupported at "
                       "Bridge Input ('arith.index_cast')");
        return WalkResult::interrupt();
      }
    } else if (isa<arith::ConstantOp, arith::MulIOp>(operation)) {
      if (failed(validateEntryBlockPlacement(operation, "launch arithmetic")))
        return WalkResult::interrupt();
    }

    auto rejectIllegalType = [&](std::optional<Type> illegalType) {
      if (!illegalType)
        return false;
      operation->emitError()
          << "Triton-to-LiteRT Bridge Input contains categorically illegal "
             "type "
          << *illegalType;
      return true;
    };

    for (Type type : operation->getOperandTypes()) {
      if (rejectIllegalType(findIllegalType(type)))
        return WalkResult::interrupt();
    }
    for (Type type : operation->getResultTypes()) {
      if (rejectIllegalType(findIllegalType(type)))
        return WalkResult::interrupt();
    }
    for (NamedAttribute attribute : operation->getAttrs()) {
      if (rejectIllegalType(findIllegalType(attribute.getValue())))
        return WalkResult::interrupt();
    }
    for (Region &region : operation->getRegions()) {
      for (Block &block : region) {
        for (BlockArgument argument : block.getArguments()) {
          if (rejectIllegalType(findIllegalType(argument.getType())))
            return WalkResult::interrupt();
        }
      }
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();
  if (failed(validateFunctionSignature(functions.front())) ||
      failed(validateMilestoneProgram(functions.front())) ||
      failed(analyzeMilestoneLaunchContractImpl(functions.front())))
    return failure();
  return success();
}

FailureOr<SmallVector<tts::MakeTensorPtrOp>>
analyzeMilestoneLaunchContract(func::FuncOp function) {
  return analyzeMilestoneLaunchContractImpl(function);
}

bool hasMilestoneModelABI(ModuleOp module) {
  auto roles = module->getAttrOfType<ArrayAttr>(kBufferRolesAttrName);
  if (!roles || roles.size() != kModelBufferCount ||
      !module->hasAttrOfType<UnitAttr>(kBuffersDistinctAttrName))
    return false;

  constexpr std::array<StringLiteral, kModelBufferCount> expectedRoles = {
      "input", "input", "output"};
  return llvm::all_of(llvm::zip_equal(roles, expectedRoles), [](auto pair) {
    auto role = dyn_cast<StringAttr>(std::get<0>(pair));
    return role && role.getValue() == std::get<1>(pair);
  });
}

LogicalResult
validateMilestoneOperationAllowlist(ModuleOp module,
                                    MilestoneOperationStage stage) {
  return validateOperationAllowlistImpl(module, stage);
}

std::unique_ptr<Pass> createVerifyBridgeInputPass() {
  return std::make_unique<VerifyBridgeInputPass>();
}

void registerTritonToLiteRTBridgePasses() {
  static PassRegistration<VerifyBridgeInputPass> registration;
  registerNormalizeLaunchMetadataPass();
  registerExtractStructuredInputSemanticsPass();
  registerFunctionalizeStructuredInputsPass();
  registerClassifyVectorAddLinalgPass();
  registerFunctionalizeStructuredOutputPass();
  registerVerifyBridgeIRV1Pass();
}

} // namespace mlir::triton_to_litert
