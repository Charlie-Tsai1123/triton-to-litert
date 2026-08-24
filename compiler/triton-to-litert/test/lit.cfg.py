# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst


config.name = "TRITON-TO-LITERT"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.triton_to_litert_obj_root, "test")

tool_dirs = [
    config.triton_to_litert_tools_dir,
    config.triton_shared_tools_dir,
    config.llvm_tools_dir,
]

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])
llvm_config.with_environment("PATH", tool_dirs, append_path=True)
llvm_config.add_tool_substitutions(
    [
        "triton-shared-opt",
        "triton-to-litert-opt",
        ToolSubst("FileCheck", unresolved="fatal"),
    ],
    tool_dirs,
)

if os.path.isfile(config.triton_to_litert_litert_opt) and os.access(
    config.triton_to_litert_litert_opt, os.X_OK
):
    config.available_features.add("litert-side-bridge")
    llvm_config.add_tool_substitutions(
        [
            ToolSubst(
                "linalg-to-tfl-opt",
                command=config.triton_to_litert_litert_opt,
                unresolved="fatal",
            )
        ],
        [],
    )
