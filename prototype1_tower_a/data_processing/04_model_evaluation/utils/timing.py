"""Model evaluation utilities - latency measurement."""

from __future__ import annotations

import time
from typing import Callable


def time_ms(fn: Callable[[], None], runs: int = 3, warmup: int = 1) -> float:
    """
    Return total wall-clock milliseconds for `runs` executions of fn().
    The warmup calls are discarded to avoid measuring JIT compilation or cold-start overhead that would not occur in
    a production serving context.
    """
    for _ in range(max(0, warmup)):
        fn()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - start) * 1000.0

def ms_per_item(total_ms: float, n_items: int) -> float:
    return total_ms / float(n_items) if n_items > 0 else 0.0
