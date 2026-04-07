"""Model evaluation utilities - reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def set_single_thread_env() -> None:
    """
    Pin all BLAS/OpenMP thread counts to 1 before importing sklearn models.
    Without this, multi-threaded libraries can cause latency measurements to vary significantly between runs on the
    same hardware.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
