get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_llvm_executable(
  triton-to-litert-opt
  triton-to-litert-opt.cpp
  PARTIAL_SOURCES_INTENDED
)

llvm_update_compile_flags(triton-to-litert-opt)

target_include_directories(triton-to-litert-opt PRIVATE
  "${TRITON_TO_LITERT_TRITON_SHARED_SOURCE_DIR}/tools/triton-shared-opt"
)

target_link_libraries(triton-to-litert-opt PRIVATE
  TritonToLiteRTBridge
  TritonTransforms
  TritonSharedAnalysis
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  MLIROptLib
  MLIRPass
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-to-litert-opt)
