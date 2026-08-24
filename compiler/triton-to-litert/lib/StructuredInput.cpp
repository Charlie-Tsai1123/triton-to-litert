//===- StructuredInput.cpp - Structured input semantic analysis ----------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TritonToLiteRT/Bridge.h"

#include "BridgeInput.h"
#include "StructuredInput.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <array>

namespace mlir::triton_to_litert {
namespace {

bool hasMilestoneShape(RankedTensorType type) {
  return type && type.hasStaticShape() && type.getRank() == 1 &&
         type.getShape() == ArrayRef<int64_t>{kMilestoneExtent};
}

bool isConstantInteger(Value value, int64_t expected) {
  APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) &&
         constant.getSExtValue() == expected;
}

bool isCanonicalTileOffset(Value offset, func::FuncOp function) {
  if (function.getNumArguments() <= kProgramIdXArgument)
    return false;

  auto indexCast = offset.getDefiningOp<arith::IndexCastOp>();
  if (!indexCast || !indexCast.getIn().getType().isInteger(32) ||
      !indexCast.getOut().getType().isIndex())
    return false;

  auto multiply = indexCast.getIn().getDefiningOp<arith::MulIOp>();
  if (!multiply)
    return false;

  Value programIdX = function.getArgument(kProgramIdXArgument);
  return (multiply.getLhs() == programIdX &&
          isConstantInteger(multiply.getRhs(), kMilestoneExtent)) ||
         (multiply.getRhs() == programIdX &&
          isConstantInteger(multiply.getLhs(), kMilestoneExtent));
}

StringRef getStructuredPointerStage(tts::MakeTensorPtrOp pointer) {
  auto function = pointer->getParentOfType<func::FuncOp>();
  if (function && function.getNumArguments() == kModelBufferCount)
    return "normalized bridge preparation";
  return "Bridge Input";
}

FailureOr<StructuredPointerSemantics>
emitStructuredLayoutError(tts::MakeTensorPtrOp pointer, const Twine &property) {
  pointer.emitError() << "Triton-to-LiteRT "
                      << getStructuredPointerStage(pointer)
                      << " unsupported structured pointer layout: " << property;
  return failure();
}

LogicalResult emitStructuredInputError(Operation *operation,
                                       const Twine &property) {
  operation->emitError()
      << "Triton-to-LiteRT structured input semantics cannot be extracted: "
      << property;
  return failure();
}

template <typename T>
void appendArray(InFlightDiagnostic &diagnostic,
                 const SmallVectorImpl<T> &values) {
  diagnostic << '[';
  llvm::interleaveComma(values, diagnostic);
  diagnostic << ']';
}

void emitDescriptor(StructuredInputSemantics &input) {
  std::string elementType;
  llvm::raw_string_ostream(elementType) << input.pointer.elementType;
  std::string logicalTensorType;
  llvm::raw_string_ostream(logicalTensorType) << input.logicalTensorType;

  InFlightDiagnostic diagnostic = input.load.emitRemark();
  diagnostic << "Triton-to-LiteRT structured input semantics: abi_input="
             << input.abiArgumentIndex << ", source_buffer=arg"
             << input.abiArgumentIndex << ", tensor_shape=";
  appendArray(diagnostic, input.pointer.tensorShape);
  diagnostic << ", sizes=";
  appendArray(diagnostic, input.pointer.sizes);
  diagnostic << ", strides=";
  appendArray(diagnostic, input.pointer.strides);
  diagnostic << ", offsets=";
  appendArray(diagnostic, input.pointer.offsets);
  diagnostic << ", element_type=" << elementType
             << ", load_result=" << logicalTensorType << ", shape_state=";
  appendArray(diagnostic, input.pointer.wrapShape);
  diagnostic << ", order=";
  appendArray(diagnostic, input.pointer.order);
  diagnostic << ", full_range=" << (input.pointer.fullRange ? "true" : "false")
             << ", wrap=" << (input.pointer.wraps ? "true" : "false")
             << ", broadcast=" << (input.pointer.broadcasts ? "true" : "false")
             << ", mask=" << (input.masked ? "present" : "none")
             << ", other=" << (input.hasOther ? "present" : "none")
             << ", boundary_check="
             << (input.hasBoundaryCheck ? "present" : "none");
}

class ExtractStructuredInputSemanticsPass final
    : public PassWrapper<ExtractStructuredInputSemanticsPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ExtractStructuredInputSemanticsPass)

  ExtractStructuredInputSemanticsPass() = default;
  ExtractStructuredInputSemanticsPass(
      const ExtractStructuredInputSemanticsPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const final {
    return "extract-triton-to-litert-structured-input-semantics";
  }

  StringRef getDescription() const final {
    return "Extract milestone-1 structured input memory semantics";
  }

  void runOnOperation() final {
    FailureOr<SmallVector<StructuredInputSemantics, 2>> inputs =
        analyzeMilestoneStructuredInputs(getOperation());
    if (failed(inputs)) {
      signalPassFailure();
      return;
    }

    if (failed(validateMilestoneOperationAllowlist(
            getOperation(),
            MilestoneOperationStage::NormalizedBridgePreparation))) {
      signalPassFailure();
      return;
    }

    if (emitDescriptors)
      for (StructuredInputSemantics &input : *inputs)
        emitDescriptor(input);
  }

private:
  Option<bool> emitDescriptors{
      *this, "emit-descriptors",
      llvm::cl::desc("Emit the extracted structured input descriptors as "
                     "diagnostic remarks"),
      llvm::cl::init(false)};
};

} // namespace

bool isMilestoneTensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return hasMilestoneShape(tensor) && tensor.getElementType().isF32();
}

FailureOr<StructuredPointerSemantics>
analyzeMilestoneStructuredPointer(tts::MakeTensorPtrOp pointer,
                                  StructuredPointerOffsetForm offsetForm) {
  DictionaryAttr discardableAttributes =
      pointer->getDiscardableAttrDictionary();
  if (!discardableAttributes.empty()) {
    pointer.emitError() << "Triton-to-LiteRT "
                        << getStructuredPointerStage(pointer)
                        << " unsupported structured pointer semantic "
                           "attribute '"
                        << discardableAttributes.begin()->getName().strref()
                        << "'";
    return failure();
  }

  auto resultType = dyn_cast<RankedTensorType>(pointer.getResult().getType());
  if (!resultType || resultType.getRank() != 1)
    return emitStructuredLayoutError(
        pointer, "result must be rank-1 tensor<1024x!tt.ptr<f32>>");

  auto resultPointerType =
      dyn_cast<triton::PointerType>(resultType.getElementType());
  auto basePointerType =
      dyn_cast<triton::PointerType>(pointer.getBase().getType());
  if (!resultPointerType || !basePointerType ||
      !resultPointerType.getPointeeType().isF32() ||
      !basePointerType.getPointeeType().isF32())
    return emitStructuredLayoutError(pointer, "element type must be f32");

  if (pointer.getSizes() != ArrayRef<int64_t>{kMilestoneExtent})
    return emitStructuredLayoutError(pointer,
                                     "size must be static extent 1024");
  if (!hasMilestoneShape(resultType))
    return emitStructuredLayoutError(
        pointer, "result must be rank-1 tensor<1024x!tt.ptr<f32>>");
  if (!pointer.getStrides().empty() ||
      pointer.getStaticStrides() != ArrayRef<int64_t>{1})
    return emitStructuredLayoutError(
        pointer, "stride must be static positive unit stride");
  if (!pointer.getShape().empty() ||
      pointer.getStaticShape() != ArrayRef<int64_t>{0})
    return emitStructuredLayoutError(
        pointer, "shape must disable wraparound and broadcasting");
  if (!pointer.getOrder().empty())
    return emitStructuredLayoutError(pointer, "order must be empty");

  if (offsetForm == StructuredPointerOffsetForm::CanonicalLaunchExpression) {
    auto function = pointer->getParentOfType<func::FuncOp>();
    if (pointer.getOffsets().size() != 1 ||
        pointer.getStaticOffsets() != ArrayRef<int64_t>{ShapedType::kDynamic} ||
        !function ||
        !isCanonicalTileOffset(pointer.getOffsets().front(), function))
      return emitStructuredLayoutError(
          pointer, "offset must be the canonical program_id.x tile expression");
  } else if (!pointer.getOffsets().empty() ||
             pointer.getStaticOffsets() != ArrayRef<int64_t>{0}) {
    return emitStructuredLayoutError(pointer,
                                     "offset must be static zero after launch "
                                     "normalization");
  }

  return StructuredPointerSemantics{
      pointer,
      pointer.getBase(),
      resultType,
      basePointerType.getPointeeType(),
      SmallVector<int64_t>(resultType.getShape()),
      SmallVector<int64_t>(pointer.getSizes()),
      SmallVector<int64_t>(pointer.getStaticStrides()),
      SmallVector<int64_t>{0},
      SmallVector<int64_t>(pointer.getStaticShape()),
      SmallVector<int32_t>(pointer.getOrder()),
      /*fullRange=*/true,
      /*wraps=*/false,
      /*broadcasts=*/false};
}

FailureOr<StructuredInputSemantics>
analyzeMilestoneStructuredInputLoad(tts::LoadOp load,
                                    StructuredPointerOffsetForm offsetForm) {
  if (load.hasMask() || load.getOther())
    return emitStructuredInputError(
        load, "structured access must be unmasked and cover the full tensor");

  DictionaryAttr discardableAttributes = load->getDiscardableAttrDictionary();
  if (!discardableAttributes.empty()) {
    InFlightDiagnostic diagnostic = load.emitError();
    diagnostic << "Triton-to-LiteRT ";
    if (offsetForm == StructuredPointerOffsetForm::CanonicalLaunchExpression)
      diagnostic << "Bridge Input";
    else
      diagnostic << "normalized bridge preparation";
    diagnostic
        << " structured access must be unmasked and cover the full tensor: "
           "unsupported load semantic attribute '"
        << discardableAttributes.begin()->getName().strref() << "'";
    return failure();
  }

  auto logicalType = dyn_cast<RankedTensorType>(load.getResult().getType());
  if (!isMilestoneTensor(logicalType))
    return emitStructuredInputError(
        load, "rank-1 f32 tensors must have static extent 1024");

  auto pointer = load.getPtr().getDefiningOp<tts::MakeTensorPtrOp>();
  if (!pointer)
    return emitStructuredInputError(
        load, "model ABI cannot be functionalized: loads must use a declared "
              "read-only input buffer");

  FailureOr<StructuredPointerSemantics> pointerSemantics =
      analyzeMilestoneStructuredPointer(pointer, offsetForm);
  if (failed(pointerSemantics))
    return failure();

  auto function = load->getParentOfType<func::FuncOp>();
  auto base = dyn_cast<BlockArgument>(pointer.getBase());
  if (!function || !base || base.getOwner() != &function.getBody().front())
    return emitStructuredInputError(
        load, "model ABI cannot be functionalized: loads must use a declared "
              "read-only input buffer");

  unsigned abiArgumentIndex = base.getArgNumber();
  if (abiArgumentIndex == kOutputBufferArgument)
    return emitStructuredInputError(
        load, "the write-only output buffer cannot be used as an input load");
  if (abiArgumentIndex >= kInputBufferCount)
    return emitStructuredInputError(
        load, "model ABI cannot be functionalized: loads must use a declared "
              "read-only input buffer");

  if (offsetForm == StructuredPointerOffsetForm::StaticZero &&
      (!base.hasOneUse() || !pointer.getResult().hasOneUse() ||
       *pointer.getResult().user_begin() != load.getOperation()))
    return emitStructuredInputError(
        pointer, "input pointer must have exactly one load interpretation and "
                 "must not escape");

  return StructuredInputSemantics{*pointerSemantics,         load,
                                  load.getResult(),          logicalType,
                                  abiArgumentIndex,
                                  /*masked=*/false,
                                  /*hasOther=*/false,
                                  /*hasBoundaryCheck=*/false};
}

FailureOr<SmallVector<StructuredInputSemantics, 2>>
analyzeMilestoneStructuredInputs(ModuleOp module) {
  SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
  if (functions.size() != 1)
    return emitStructuredInputError(
        module, "expected exactly one normalized entry function");

  func::FuncOp function = functions.front();
  if (!hasMilestoneModelABI(module))
    return emitStructuredInputError(
        function,
        "model ABI cannot be functionalized: expected distinct buffer roles "
        "[input, input, output]");

  if (module->hasAttr(kLaunchGridAttrName) ||
      function.getNumArguments() != kModelBufferCount ||
      function.getNumResults() != 0 || !function.getBody().hasOneBlock())
    return emitStructuredInputError(
        function,
        "expected a normalized three-buffer ABI with no launch metadata");

  for (Type argumentType : function.getArgumentTypes()) {
    auto pointerType = dyn_cast<triton::PointerType>(argumentType);
    if (!pointerType || !pointerType.getPointeeType().isF32())
      return emitStructuredInputError(
          function, "normalized model buffers must have type !tt.ptr<f32>");
  }

  SmallVector<tts::LoadOp> loads;
  function.walk([&](tts::LoadOp load) { loads.push_back(load); });
  if (loads.size() != kInputBufferCount)
    return emitStructuredInputError(
        function, "expected exactly two structured input loads");

  std::array<bool, kInputBufferCount> seenInputs = {false, false};
  SmallVector<StructuredInputSemantics, 2> inputs;
  inputs.reserve(kInputBufferCount);
  for (tts::LoadOp load : loads) {
    FailureOr<StructuredInputSemantics> input =
        analyzeMilestoneStructuredInputLoad(
            load, StructuredPointerOffsetForm::StaticZero);
    if (failed(input))
      return failure();
    if (seenInputs[input->abiArgumentIndex])
      return emitStructuredInputError(
          load, "each declared input buffer must have exactly one load");
    seenInputs[input->abiArgumentIndex] = true;
    inputs.push_back(*input);
  }

  if (!llvm::all_of(seenInputs, [](bool seen) { return seen; }))
    return emitStructuredInputError(
        function, "each declared input buffer must have exactly one load");

  llvm::sort(inputs, [](const StructuredInputSemantics &lhs,
                        const StructuredInputSemantics &rhs) {
    return lhs.abiArgumentIndex < rhs.abiArgumentIndex;
  });
  return inputs;
}

std::unique_ptr<Pass> createExtractStructuredInputSemanticsPass() {
  return std::make_unique<ExtractStructuredInputSemanticsPass>();
}

void registerExtractStructuredInputSemanticsPass() {
  static PassRegistration<ExtractStructuredInputSemanticsPass> registration;
}

} // namespace mlir::triton_to_litert
