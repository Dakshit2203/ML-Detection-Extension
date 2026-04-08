"""
Tower A - Prototype 1 - backend/adaptive_threshold.py

Purpose
Implements the adaptive quantile-based decision threshold that was introduced as an architectural response to the
Phase 6 external validation finding.

Three-phase enforcement
To avoid a cold-start problem (FPR = 1.0 from the first scan), the threshold goes through three phases based on the
number of scores in the buffer:

  Phase 1 - Learning (buffer < MIN_SCORES_TO_ADAPT)
    No adaptive threshold yet. Conservative fallback thresholds (0.99 / 0.95) mean almost nothing is blocked. Only
    p ≥ 0.9999 triggers even a warning.

  Phase 2 - Adapting (MIN_SCORES_TO_ADAPT ≤ buffer < MIN_SCORES_TO_BLOCK)
    Quantile threshold computed from real browsing data but block decisions are downgraded to warn - the sample is not
    yet large enough to trust.

  Phase 3 - Stable (buffer ≥ MIN_SCORES_TO_BLOCK)
    Full enforcement: danger -> block, suspicious -> warn, low/safe -> allow.

Persistence
The buffer and its metadata are persisted to a JSON file (adaptive_state.json) in the artifacts' directory. This
ensures the adaptive state survives backend restarts - the buffer is not reset every time uvicorn is restarted.

Privacy note
The buffer stores only floating-point scores (not URLs). No browsing history or personal data is recorded. This is
consistent with the GDPR privacy-by-design constraint stated in the project requirements (NFR9).

Key assumption
The assumption underlying adaptive quantile thresholding is that the majority of URLs encountered during real browsing
are benign. This is reasonable for a general-purpose browser extension where benign traffic dominates. Under heavy
adversarial use (a user deliberately visiting many phishing pages), the threshold would drift upward.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class AdaptiveState:
    # Serialisable state for the adaptive threshold buffer.
    # Stored as JSON in artifacts/adaptive_state.json.
    scores: list = field(default_factory=list)
    n_updates: int = 0


class AdaptiveThreshold:
    # Maintains a rolling buffer of p_phish scores and computes quantile-based decision thresholds from it.
    def __init__(self, state_path: Path, maxlen: int = 500, min_samples: int = 50):
        self.state_path = Path(state_path)
        self.maxlen = maxlen
        self.min_samples = min_samples
        self.state = self._load()

    def _load(self) -> AdaptiveState:
        if self.state_path.exists():
            try:
                raw = json.loads(self.state_path.read_text(encoding="utf-8"))
                return AdaptiveState(
                    scores=list(raw.get("scores", [])),
                    n_updates=int(raw.get("n_updates", 0)),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                # Corrupt file - start fresh rather than crash
                pass
        return AdaptiveState()

    def _save(self) -> None:
        # Persists the current state to disk as JSON.
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(asdict(self.state), indent=2),
            encoding="utf-8"
        )


    def update(self, p: float) -> None:
        # Adds a new p_phish score to the rolling buffer and saves state. Scores are appended to the right. When the buffer 
        # exceeds maxlen, the oldest score (leftmost) is removed.
        self.state.scores.append(float(p))
        if len(self.state.scores) > self.maxlen:
            self.state.scores = self.state.scores[-self.maxlen:]
        self.state.n_updates += 1
        self._save()
    
    def reset(self) -> None:
        # Clears the buffer entirely. Useful for testing. Resets n_updates to zero and saves.
        self.state = AdaptiveState()
        self._save()

    def quantile_threshold(self, target_fpr: float) -> Optional[float]:
        # Computes a single threshold as the (1 − target_fpr) quantile of the buffer. Returns None if the buffer has 
        # fewer than min_samples entries.
        if len(self.state.scores) < self.min_samples:
            return None
        q = float(max(0.0, min(1.0, 1.0 - float(target_fpr))))
        arr = np.asarray(self.state.scores, dtype=np.float64)
        return float(np.quantile(arr, q))

    def get_thresholds(
            self,
            target_fpr_red: float,
            target_fpr_orange: float,
            fallback_red: float,
            fallback_orange: float,
    ) -> Dict[str, Any]:
        # Returns a complete threshold state dictionary for use in the /predict response. This dictionary is passed to 
        # the popup via the JSON response so the adaptive phase is visible in the UI.
        buf = len(self.state.scores)
        learning = buf < self.min_samples

        t_red = self.quantile_threshold(target_fpr_red) if not learning else None
        t_orange = self.quantile_threshold(target_fpr_orange) if not learning else None

        red = float(t_red) if t_red is not None else float(fallback_red)
        orange = float(t_orange) if t_orange is not None else float(fallback_orange)

        # Invariant: orange threshold must not exceed red threshold
        if orange > red:
            orange = red

        return {
            "mode": "adaptive_quantile",
            "learning_mode": learning,
            "min_samples": self.min_samples,
            "buffer_size": buf,
            "target_fpr_red": float(target_fpr_red),
            "target_fpr_orange": float(target_fpr_orange),
            "threshold_red": red,
            "threshold_orange": orange,
            "n_updates": int(self.state.n_updates),
        }
