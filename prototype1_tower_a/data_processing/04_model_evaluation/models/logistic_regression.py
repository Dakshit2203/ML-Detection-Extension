"""
Tower A - Phase 4 Models - Logistic Regression

Logistic regression classifier for the handcrafted AE feature regime.

This is the model selected for final deployment based on Phase 4 evaluation:
  - Highest PR-AUC on the eTLD+1 split (0.9733) with the smallest generalisation gap
  - Lowest FPR at the 2% policy target (0.0197)
  - Analytically exact XAI: feature contribution = coef_i × value_i (exact SHAP for linear models)
  - Fastest inference (0.00089 ms/URL) and smallest serialised size (0.001 MB)

lbfgs is used over liblinear because the feature matrix is dense and lbfgs handles multi-class well if the label set
is ever extended. max_iter=4000 is set high to ensure convergence on the full training set without early stopping.

Called by evaluate.py via:
    from models.logistic_regression import build_model
    model = build_model(seed=42)
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def build_model(*, seed: int = 42) -> LogisticRegression:
    """
    Return an unfitted LogisticRegression with the locked evaluation configuration.

    Parameters
    seed - Random state for reproducibility. Must match the seed passed to set_seeds() in evaluate.py so that results
    are deterministic across runs.
    """
    return LogisticRegression(
        solver="lbfgs",
        max_iter=5000,
        random_state=seed,
    )