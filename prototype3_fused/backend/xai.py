"""
Prototype 3 - Fused System - backend/xai.py

Per-request explainability for both towers.

Tower A - analytical attribution
    contribution_i = coef_i * feature_value_i

This is exact for logistic regression - mathematically equivalent to SHAP LinearExplainer but requires no external
library and adds no measurable latency. Positive values push the score towards phishing; negative values push towards benign.

Tower B - global feature importance
HGB exposes feature_importances_ (mean decrease in impurity). This is a global surrogate rather than a true per-instance
explanation. It is reported as such in the popup. Full per-instance attribution would require TreeSHAP, which is out of
scope given the browser-latency constraints of this prototype.

Both functions return a list of dicts sorted by descending magnitude, truncated to top_n, for direct rendering in popup.js.
"""

from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def explain_tower_a(
    model: Any,
    feature_values: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Returns the top_n Tower A features by |coef_i * feature_value_i|.
    """
    if not hasattr(model, "coef_"):
        return []

    coef = model.coef_[0]
    values = feature_values.iloc[0].to_numpy(dtype=np.float64)

    attributions = [
        {
            "feature": name,
            "value": round(float(values[i]), 6),
            "contribution": round(float(coef[i]) * float(values[i]), 6),
            "direction": "phishing" if coef[i] * values[i] > 0 else "benign",
        }
        for i, name in enumerate(feature_names)
    ]

    attributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return attributions[:top_n]


def explain_tower_b(
    model: Any,
    feature_names: List[str],
    meta: Dict[str, Any],
    importances: List[float] | None = None,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Returns the top_n Tower B features by permutation importance alongside
    the observed metadata values for the scored domain.

    Importances are pre-computed during training (04_train_eval.py) because
    HistGradientBoostingClassifier does not expose feature_importances_
    as a direct attribute.
    """
    if importances is None:
        return []

    attributions = [
        {
            "feature": name,
            "value": meta.get(name, None),
            "importance": round(float(importances[i]), 6),
            "direction": "informative",
        }
        for i, name in enumerate(feature_names)
    ]

    attributions.sort(key=lambda x: x["importance"], reverse=True)
    return attributions[:top_n]