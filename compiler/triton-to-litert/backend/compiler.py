"""Build-system adapter for the Triton-to-LiteRT compiler-only plugin."""

from triton.backends.compiler import BaseBackend


class TritonToLiteRTBackend(BaseBackend):
    """Non-selectable backend required by Triton's external-plugin loader."""

    @staticmethod
    def supports_target(target):
        return False

    def hash(self):
        raise RuntimeError("triton-to-litert is a compiler pipeline, not a runtime backend")

    def parse_options(self, options):
        raise RuntimeError("triton-to-litert is a compiler pipeline, not a runtime backend")

    def add_stages(self, stages, options):
        raise RuntimeError("triton-to-litert is a compiler pipeline, not a runtime backend")

    def load_dialects(self, context):
        raise RuntimeError("triton-to-litert is a compiler pipeline, not a runtime backend")

    def get_module_map(self):
        raise RuntimeError("triton-to-litert is a compiler pipeline, not a runtime backend")
