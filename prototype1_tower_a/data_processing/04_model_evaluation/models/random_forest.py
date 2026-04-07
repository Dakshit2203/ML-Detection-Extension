"""
Tower A - Phase 4 Models - Random Forest

Random forest classifier for the handcrafted AE feature regime.

Included in Phase 4 as the primary ensemble baseline against logistic regression. Random forests are robust to feature
scale differences (no normalisation needed) and naturally capture non-linear feature interactions. However, on this
dataset they show a larger generalisation gap than LR under the eTLD+1 split (0.025 vs 0.021 PR-AUC drop), indicating
they overfit domain-specific patterns more than the linear model.

n_estimators=300 balances evaluation stability with runtime. n_jobs=1 is deliberately set to a single thread so that
latency measurements in evaluate.py reflect single-request inference cost, consistent with the browser extension
deployment context where one URL is classified at a time.

Called by evaluate.py via:
    from models.random_forest import build_model
    model = build_model(seed=42)
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def build_model(*, seed: int = 42) -> RandomForestClassifier:
    """
    Return an unfitted RandomForestClassifier with the locked evaluation configuration.

    Parameters
    seed - Random state for reproducibility. Must match the seed passed to set_seeds() in evaluate.py so that results
    are deterministic across runs.
    """
    return RandomForestClassifier(
        n_estimators=400,
        n_jobs=1,           # single-threaded for consistent latency measurement
        random_state=seed,
    )