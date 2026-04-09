"""
Prototype 3 - Fused System - backend/fusion.py

Weighted late-fusion of Tower A and Tower B scores, and the three-phase adaptive decision policy applied to the
resulting fused probability.

Fusion formula
    p_fused = WEIGHT_A * p_A + WEIGHT_B * p_B
            = 0.75 * p_A + 0.25 * p_B

Late fusion is preferred over feature-level fusion because it preserves modularity: each tower's score is independently
inspectable, XAI attributions remain meaningful within their own feature spaces, and either tower can be retrained
without affecting the other.

If Tower B returns None (probe failure), the fused score falls back to p_A alone. The response flags
tower_b_available=False so the popup can display a reduced-confidence note.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .config import WEIGHT_A, WEIGHT_B

def fuse(p_a: float, p_b: Optional[float]) -> Tuple[float, bool]:
    """ 
    Combines Tower A and Tower B probabilities into a single fused score. 
    If p_b is None, falls back to p_A alone with tower_b_available=False.
    """
    if p_b is None:
        return float(p_a), False

    p_fused = WEIGHT_A * float(p_a) + WEIGHT_B * float(p_b)
    return float(p_fused), True


def apply_decision_policy(
    p_fused: float,
    learning_mode: bool,
    buffer_size: int,
    t_red: float,
    t_orange: float,
    min_to_block: int,
    learning_warn_at: float,
) -> Dict[str, Any]:
    """
    Maps a fused probability to a risk level and enforcement decision.

    The three-phase logic mirrors Prototype 1's adaptive_threshold.py:

      Phase 1 (learning) - no blocking; conservative fallbacks.
      Phase 2 (adapting) - calibrated thresholds but block → warn.
      Phase 3 (stable)   - full enforcement.
    """
    if learning_mode:
        if p_fused >= learning_warn_at:
            risk, decision = "suspicious", "warn"
        elif p_fused >= 0.50:
            risk, decision = "low", "allow"
        else:
            risk, decision = "safe", "allow"
        phase = "learning"

    elif buffer_size < min_to_block:
        if p_fused >= t_red:
            risk, decision = "danger", "warn"  # block downgraded
        elif p_fused >= t_orange:
            risk, decision = "suspicious", "warn"
        elif p_fused >= 0.50:
            risk, decision = "low", "allow"
        else:
            risk, decision = "safe", "allow"
        phase = "adapting_no_block"

    else:
        if p_fused >= t_red:
            risk, decision = "danger", "block"
        elif p_fused >= t_orange:
            risk, decision = "suspicious", "warn"
        elif p_fused >= 0.50:
            risk, decision = "low", "allow"
        else:
            risk, decision = "safe", "allow"
        phase = "stable_block_enabled"

    return {"risk_level": risk, "decision": decision, "phase": phase}
