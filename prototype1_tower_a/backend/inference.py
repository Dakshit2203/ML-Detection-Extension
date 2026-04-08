"""
Tower A - Prototype 1 - backend/inference.py

Purpose
Loads the trained logistic regression model and feature extraction modules, and exposes a single predict_proba(url_norm) 
method that returns a p_phish score in [0, 1] for any HTTP/HTTPS URL. 

The model is stored as a bare LogisticRegression object so that the exact analytical XAI formula can be applied:
    contribution_i = coef_i × feature_value_i
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import load as joblib_load
import tldextract

# Reads the canonical 35-column feature order from feature_spec.json.
# Any mismatch between the feature columns used during training and those produced here would cause silent score corruption.
def _load_feature_columns(spec_path: Path) -> List[str]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    cols = spec.get("feature_columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError(
            f"feature_spec.json at {spec_path} is missing 'feature_columns' "
            "or it is empty. This file is produced by Phase 5 (train_final_lr.py)."
        )
    return [str(c) for c in cols]

# Extracts the effective top-level domain + 1 (eTLD+1) from a URL.
def _compute_etld1(url: str) -> str | float:
    ext = tldextract.extract(str(url))
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return np.nan

# Loads the Tower A model and feature modules; produces p_phish for a URL.
class TowerAInference:
    
    def __init__(self, model_path: Path, feature_spec_path: Path, features_dir: Path):
        # Load model 
        self.model = joblib_load(model_path)

        # Load feature spec (column order contract)
        self.feature_cols = _load_feature_columns(feature_spec_path)

        # Import feature extraction modules
        # The features/ directory must be on sys.path for the group_* imports to resolve.
        features_dir_str = str(features_dir.resolve())
        if features_dir_str not in sys.path:
            sys.path.insert(0, features_dir_str)

        import group_a_structure 
        import group_b_chars
        import group_c_entropy
        import group_d_tokens
        import group_e_brand

        # Store module references for use in extract_features()
        self._group_a = group_a_structure
        self._group_b = group_b_chars
        self._group_c = group_c_entropy
        self._group_d = group_d_tokens
        self._group_e = group_e_brand

    # Extracts all 35 features from a normalised URL string.
    def extract_features(self, url_norm: str) -> pd.DataFrame:
        # Single-row DataFrame: both url_norm and etld1 are needed by the modules
        df = pd.DataFrame({"url_norm": [url_norm]})
        df["etld1"] = df["url_norm"].map(_compute_etld1)

        # Run all five feature group extractors
        A = self._group_a.extract_group_a(df, url_col="url_norm", etld1_col="etld1")
        B = self._group_b.extract_group_b(df, url_col="url_norm")
        C = self._group_c.extract_group_c(df, url_col="url_norm")
        D = self._group_d.extract_group_d(df, url_col="url_norm")
        E = self._group_e.extract_group_e(df, url_col="url_norm", etld1_col="etld1")

        # Concatenate all groups horizontally (35 columns total)
        feats = pd.concat([A, B, C, D, E], axis=1)

        # Validate that all expected columns are present
        missing = [c for c in self.feature_cols if c not in feats.columns]
        if missing:
            raise RuntimeError(
                f"Feature extraction produced missing columns: {missing}. "
            )
        # Enforce canonical column order (critical: must match training)
        return feats[self.feature_cols]

    # Returns the phishing probability for a normalised URL.
    def predict_proba(self, url_norm: str) -> float:
        # Extract features -> (1, 35) float32 array
        X_df = self.extract_features(url_norm)
        X = X_df.to_numpy(dtype=np.float32)

        # LR predict_proba returns shape (1, 2); take column 1 (phishing class)
        p = float(self.model.predict_proba(X)[0, 1])
        return p
