"""
Model evaluation utilities - feature matrix loading and split path resolution.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def split_paths(root: Path, split_kind: str) -> Tuple[Path, Path, Path]:
    """
    Return the (train, val, test) CSV paths for the given split type. root is the data_processing/ directory,
    resolved by project_root().
    """
    if split_kind == "etld1":
        base = root / "03_data_splitting" / "outputs" / "etld1"
        return base / "train.csv", base / "val.csv", base / "test.csv"
    if split_kind == "random":
        base = root / "03_data_splitting" / "outputs" / "random_baseline"
        return base / "train.csv", base / "val.csv", base / "test.csv"
    raise ValueError(f"split_kind must be 'etld1' or 'random', got: {split_kind!r}")

def load_split_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"url_norm", "label"}.issubset(df.columns):
        raise RuntimeError(f"{path} missing required columns: url_norm, label")
    df["url_norm"] = df["url_norm"].astype(str)
    df["label"] = df["label"].astype(int)
    return df

def resolve_features_ae_path(root: Path) -> Path:
    """Locate the precomputed AE feature matrix written by Phase 2."""
    p = root / "02_feature_engineering" / "outputs" / "features_handcrafted_AE.csv"
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Feature matrix not found at: {p}\n"
        f"Run 02_feature_engineering/extract_features.py first."
    )

def build_ae_xy(
    features_path: Path,
    split_df: pd.DataFrame,
    *,
    drop_cols: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Join split_df (url_norm, label) to the precomputed AE feature matrix and return (X, y, feature_cols).

    The join is on url_norm + label rather than url_norm alone so that any row count mismatch caused by duplicate
    or missing URLs is caught explicitly.
    """
    feat_df = pd.read_csv(features_path)
    if not {"url_norm", "label"}.issubset(feat_df.columns):
        raise RuntimeError(f"{features_path} missing required columns: url_norm, label")

    feat_df["url_norm"] = feat_df["url_norm"].astype(str)
    feat_df["label"] = feat_df["label"].astype(int)

    merged = split_df.merge(feat_df, on=["url_norm", "label"], how="left")
    if merged.shape[0] != split_df.shape[0]:
        raise RuntimeError(
            f"Merge changed row count ({split_df.shape[0]} -> {merged.shape[0]})."
            f"Check for duplicate url_norm values."
        )

    exclude = {"url_norm", "label", "source", "etld1", "split"}
    if drop_cols:
        exclude |= set(drop_cols)

    feature_cols = [
        c for c in merged.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(merged[c])
    ]

    if not feature_cols:
        raise RuntimeError("No numeric AE feature columns found after merge.")

    X = merged[feature_cols].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy(dtype=np.int32)
    return X, y, feature_cols

# The preprocessor must be a top-level function, not a lambda, for joblib to pickle it correctly when the vectoriser
# is serialised as part of the pipeline.
_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)


def _strip_scheme(s: str) -> str:
    """Remove http:// or https:// before character n-gram extraction."""
    return _SCHEME_RE.sub("", s)


def build_tfidf_vectorizer(max_features: int = 50_000,
                           min_df: int = 2) -> TfidfVectorizer:
    """
    Character 3–5-gram TF-IDF vectoriser used by the n-gram SGD baseline.
    Scheme is stripped so that http:// and https:// prefixes do not dominate the most frequent n-grams.
    """
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        preprocessor=_strip_scheme,
        lowercase=False,
        max_features=max_features,
        min_df=min_df,
    )