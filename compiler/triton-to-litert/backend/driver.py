"""Build-system adapter for the Triton-to-LiteRT compiler-only plugin."""

from triton.backends.driver import DriverBase


class TritonToLiteRTDriver(DriverBase):
    """Inactive driver required by Triton's external-plugin loader."""

    @classmethod
    def is_active(cls):
        return False

    def map_python_to_cpp_type(self, ty):
        raise RuntimeError("triton-to-litert does not provide a runtime driver")

    def get_current_target(self):
        raise RuntimeError("triton-to-litert does not provide a runtime driver")

    def get_active_torch_device(self):
        raise RuntimeError("triton-to-litert does not provide a runtime driver")

    def get_benchmarker(self):
        raise RuntimeError("triton-to-litert does not provide a runtime driver")
