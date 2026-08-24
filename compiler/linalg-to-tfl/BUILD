package(default_visibility = ["//visibility:public"])

# The LiteRT-pinned build compiles these tracked sibling sources through the
# //litert/linalg_to_tfl_bridge overlay package. That label is intentionally
# inside the visibility boundary of the pinned TFL dialect library.
filegroup(
    name = "bridge_pass_source",
    srcs = ["lib/Transforms/CategoryToTFL.cpp"],
)

filegroup(
    name = "bridge_pass_header",
    srcs = ["include/LinalgToTFL/Passes.h"],
)

filegroup(
    name = "bridge_tool_source",
    srcs = ["tools/linalg-to-tfl-opt.cpp"],
)
