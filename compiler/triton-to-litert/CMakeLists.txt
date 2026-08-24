set(TRITON_TO_LITERT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_TO_LITERT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

if(NOT TARGET TritonToStructured)
  message(FATAL_ERROR
    "The Triton-Shared plugin must precede triton-to-litert in TRITON_PLUGIN_DIRS"
  )
endif()

# Triton currently uses TRITON_PLUGIN_DIRS as its only external CMake hook and
# assumes every entry is also a Python runtime backend.  This project only adds
# a sibling compiler tool, so keep it out of the generated backend tuple.
list(REMOVE_ITEM TRITON_PLUGIN_NAMES triton_to_litert)
set(TRITON_PLUGIN_NAMES "${TRITON_PLUGIN_NAMES}" PARENT_SCOPE)

get_filename_component(
  TRITON_TO_LITERT_REPOSITORY_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  ABSOLUTE
)

set(
  TRITON_TO_LITERT_TRITON_SHARED_SOURCE_DIR
  "${TRITON_TO_LITERT_REPOSITORY_ROOT}/third_party/triton-shared"
)
set(
  TRITON_TO_LITERT_TRITON_SHARED_BINARY_DIR
  "${CMAKE_BINARY_DIR}/third_party/triton_shared"
)

include_directories("${TRITON_TO_LITERT_SOURCE_DIR}/include")
include_directories("${TRITON_TO_LITERT_TRITON_SHARED_SOURCE_DIR}/include")
include_directories("${TRITON_TO_LITERT_TRITON_SHARED_BINARY_DIR}/include")

add_subdirectory(lib)
add_subdirectory(tools/triton-to-litert-opt)
add_subdirectory(test)
