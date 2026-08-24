add_triton_object(TritonToLiteRTBridge
  Pipelines.cpp
  FunctionalizeStructuredInputs.cpp
  NormalizeLaunchMetadata.cpp
  StructuredInput.cpp
  VerifyBridgeIR.cpp
  VerifyBridgeInput.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  TritonArithToLinalg
  TritonToStructured
  TritonToUnstructured
)

target_compile_options(
  TritonToLiteRTBridge
  PRIVATE
  ${TRITON_DISABLE_EH_RTTI_FLAGS}
)

target_include_directories(TritonToLiteRTBridge PUBLIC
  "${TRITON_TO_LITERT_SOURCE_DIR}/include"
)
