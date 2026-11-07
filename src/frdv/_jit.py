"""Optional numba kernels (``pip install frdv[fast]``), imported lazily.

``AVAILABLE`` only checks that numba is installed; numba itself (~0.3 s to import, plus its type
registries on first use) is imported the first time a kernel is used, so code paths that use the
native build (``frdv._native``) never pay for it. Disable with ``FRDV_NO_JIT=1``. The kernels live in
``_jit_kernels.py``.
"""

from __future__ import annotations

import importlib.util
import os

AVAILABLE = (os.environ.get("FRDV_NO_JIT", "0") not in ("1", "true", "True")
             and importlib.util.find_spec("numba") is not None)


def __getattr__(name):
    if not AVAILABLE or name.startswith("__"):
        raise AttributeError(name)
    from . import _jit_kernels
    return getattr(_jit_kernels, name)
