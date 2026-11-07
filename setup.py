"""Optional C extension: the native clip solver (src/frdv/_native.c), prebuilt into the wheel.

It is a plain shared library loaded with ctypes (frdv/_native.py), not a CPython module, so it has
no PyInit function. If it cannot be built (no compiler), the install still succeeds: frdv then
compiles the same source on first use, or falls back to numba / numpy.

Flags are portable (no -march=native) so the wheel runs on any CPU of its platform.
"""

import re
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

SRC = "src/frdv/_native.c"
# every non-static frdv_* function is called from Python
EXPORTS = sorted(set(re.findall(r"^\w[\w\s\*]*?\b(frdv_\w+)\(", Path(SRC).read_text(), re.M)))


class BuildLib(build_ext):
    def get_export_symbols(self, ext):
        return EXPORTS   # MSVC: export these instead of the PyInit_* setuptools assumes

    def build_extension(self, ext):
        if self.compiler.compiler_type == "msvc":
            ext.extra_compile_args = ["/O2", "/fp:fast"]
        else:
            ext.extra_compile_args = ["-O3", "-ffast-math", "-fno-finite-math-only"]
            ext.libraries = [] if sys.platform == "darwin" else ["m"]
        super().build_extension(ext)


setup(
    ext_modules=[Extension("frdv._libfrdv", [SRC], optional=True)],
    cmdclass={"build_ext": BuildLib},
)
