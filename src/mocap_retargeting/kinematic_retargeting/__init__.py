"""Constrained kinematic retargeting and trajectory refinement."""

import os


def configure_numerical_threads():
    """Preserve the existing single-thread limits inside each process."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


# Set limits before importing numerical libraries, including in spawned workers.
configure_numerical_threads()
