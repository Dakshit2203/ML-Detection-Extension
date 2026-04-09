"""
Tower B - Prototype 2 - backend/decision_policy.py

Threshold-based decision policies for converting a p_b score into a risk level and action (allow / warn / block).

Two policies are implemented:

  FixedThresholdPolicy
    Uses constant threshold values from config.py. This is the default for Prototype 2 because Tower B operates
    standalone here - it is not yet fused with Tower A. Fixed thresholds are safe and predictable.

  AdaptiveQuantilePolicy
    The same rolling-buffer approach used in Tower A. Included for completeness and so that decision_policy.py can be
    reused in Prototype 3 without modification, but not the default for standalone Tower B.

The make_policy() factory function reads settings.decision_mode and returns the appropriate policy. All other modules
use make_policy() rather than instantiating policies directly.

Threshold values
threshold_red = 0.70 (scores above this -> danger / block)
threshold_orange = 0.55 (scores above this -> suspicious / warn)

These were selected based on the Tower B standalone evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np

from .config import settings

# Fixed threshold policy
class FixedThresholdPolicy:
    """
    Applies constant thresholds from config.py to produce a decision.

    It is deterministic, requires no warm-up period, and produces the same decision for the same score on every call.
    """

    def decide(self, score: float) -> Dict[str, Any]:
        """
        Maps a p_b score to a risk level and decision.

        Parameters
        score : float p_b in [0, 1] from the Tower B HGB model.

        Returns
        dict with keys: decision, risk_level, thresholds
        """
        s = float(score)
        t_red = float(settings.threshold_red)
        t_orange = float(settings.threshold_orange)

        if s >= t_red:
            risk, decision = "danger", "block"
        elif s >= t_orange:
            risk, decision = "suspicious", "warn"
        else:
            risk, decision = "low", "allow"

        return {
            "decision": decision,
            "risk_level": risk,
            "thresholds": {
                "mode": "fixed",
                "threshold_red": t_red,
                "threshold_orange": t_orange,
            },
        }

# Adaptive quantile policy (mirrors Tower A implementation)
@dataclass
class _AdaptiveState:
    """Internal state for the adaptive policy (not exposed externally)."""
    learning_mode: bool = True
    buffer_size: int = 0
    threshold_red: float = settings.threshold_red
    threshold_orange: float = settings.threshold_orange
    n_updates: int = 0


class AdaptiveQuantilePolicy:
    """
    Rolling-buffer adaptive threshold policy.

    Thresholds are set as quantiles of recently observed scores, targeting a specified false positive rate. The same
    logic as Tower A's AdaptiveThreshold class, adapted as a decision policy rather than a standalone class.

    Not the default for Prototype 2 standalone - provided for completeness and reuse in Prototype 3.
    """

    def __init__(self):
        self._scores: List[float] = []
        self._state = _AdaptiveState()

    def update(self, score: float) -> None:
        """Adds a new score to the rolling buffer and recomputes thresholds."""
        self._scores.append(float(score))
        if len(self._scores) > settings.buffer_max:
            self._scores = self._scores[-settings.buffer_max:]

        self._state.buffer_size = len(self._scores)

        if len(self._scores) < settings.min_samples:
            self._state.learning_mode = True
            return

        self._state.learning_mode = False
        arr = np.asarray(self._scores, dtype=np.float64)
        self._state.threshold_red = float(np.quantile(arr, 1.0 - settings.target_fpr_red))
        self._state.threshold_orange = float(np.quantile(arr, 1.0 - settings.target_fpr_orange))
        self._state.n_updates += 1

    def decide(self, score: float) -> Dict[str, Any]:
        """Maps a p_b score to a decision using the current adaptive thresholds."""
        s = float(score)
        t_red = float(self._state.threshold_red)
        t_orange = float(self._state.threshold_orange)

        if s >= t_red:
            risk, decision = "danger", "block"
        elif s >= t_orange:
            risk, decision = "suspicious", "warn"
        else:
            risk, decision = "low", "allow"

        # Do not block during the learning phase or before the buffer is stable
        if decision == "block" and (
            self._state.learning_mode
            or self._state.buffer_size < settings.min_scores_to_block
        ):
            decision = "warn"

        return {
            "decision": decision,
            "risk_level": risk,
            "thresholds": {
                "mode": "adaptive_quantile",
                "learning_mode": self._state.learning_mode,
                "buffer_size": self._state.buffer_size,
                "threshold_red": t_red,
                "threshold_orange": t_orange,
                "n_updates": self._state.n_updates,
            },
        }

# Factory
def make_policy() -> FixedThresholdPolicy | AdaptiveQuantilePolicy:
    """
    Returns the appropriate policy instance based on settings.decision_mode.

    Called once at backend startup. The returned object is then reused for every request, which is important for 
    AdaptiveQuantilePolicy since its rolling buffer must persist across requests.
    """
    if settings.decision_mode == "adaptive_quantile":
        return AdaptiveQuantilePolicy()
    return FixedThresholdPolicy()
