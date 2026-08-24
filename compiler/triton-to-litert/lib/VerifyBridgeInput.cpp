//===- VerifyBridgeInput.cpp - Structural Bridge Input gate --------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TritonToLiteRT/Bridge.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/StringSwitch.h"

#include <optional>

namespace mlir::triton_to_litert {
namespace {

bool isAllowedOperationNamespace(StringRef dialectNamespace) {
  return llvm::StringSwitch<bool>(dialectNamespace)
      .Cases({"builtin", "func", "arith", "math"}, true)
      .Cases({"tensor", "linalg", "tts"}, true)
      .Default(false);
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
    return "Verify the structural Triton-to-LiteRT Bridge Input boundary";
  }

  void runOnOperation() final {
    WalkResult result = getOperation().walk([&](Operation *operation) {
      StringRef dialectNamespace = operation->getName().getDialectNamespace();
      if (!isAllowedOperationNamespace(dialectNamespace)) {
        operation->emitError()
            << "Triton-to-LiteRT Bridge Input contains categorically illegal "
               "operation "
            << operation->getName();
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
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createVerifyBridgeInputPass() {
  return std::make_unique<VerifyBridgeInputPass>();
}

void registerTritonToLiteRTBridgePasses() {
  static PassRegistration<VerifyBridgeInputPass> registration;
}

} // namespace mlir::triton_to_litert
