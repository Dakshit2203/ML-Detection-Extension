"""
Tower A - Phase 4 Models - HistGradientBoosting

Histogram-based gradient boosting classifier for the handcrafted AE feature regime.

HistGradientBoosting is the scikit-learn implementation of the LightGBM-style histogram algorithm. It was chosen over
GradientBoostingClassifier for two reasons:
  1. Native handling of missing values without imputation - num_subdomains uses a -1 sentinel for missing eTLD+1,
  and HistGradientBoosting treats any value outside the learned bin boundaries as missing, effectively isolating it.
  2. Substantially faster training on tabular data due to the histogram binning strategy, which is important for the
  Phase 4 grid across multiple splits.

In Phase 4 it is evaluated on the handcrafted AE feature set to provide a fair cross-model comparison.

max_iter=500 is sufficient for convergence on 35 features. early_stopping is disabled to ensure all runs train for the
same number of iterations regardless of validation set composition.

Called by evaluate.py via:
    from models.hist_gradient_boosting import build_model
    model = build_model(seed=42)
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier


def build_model(*, seed: int = 42) -> HistGradientBoostingClassifier:
    """
    Return an unfitted HistGradientBoostingClassifier with the locked evaluation configuration.

    Parameters
    seed - Random state for reproducibility. Must match the seed passed to set_seeds() in evaluate.py so that results
    are deterministic across runs.
    """
    return HistGradientBoostingClassifier(
        max_iter=500,
        early_stopping=False,   # ensures identical iteration count across all runs
        random_state=seed,
    )