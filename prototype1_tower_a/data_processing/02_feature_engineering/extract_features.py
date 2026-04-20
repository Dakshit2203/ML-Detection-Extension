"""
Tower A - Phase 2: Feature Extraction

Extracts all 35 handcrafted URL features (Groups A–E) and writes the feature matrix used by all downstream model
evaluation and training phases.

Inputs:  01_preprocessing/outputs/towerA_dataset.csv
Outputs: 02_feature_engineering/outputs/features_handcrafted_AE.csv

"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PROC = BASE_DIR.parent
P1_ROOT = DATA_PROC.parent
FEATURES_DIR = P1_ROOT / "features"

if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))

from group_a_structure import extract_group_a
from group_b_chars import extract_group_b
from group_c_entropy import extract_group_c
from group_d_tokens import extract_group_d
from group_e_brand import extract_group_e

DATASET_PATH = DATA_PROC / "01_preprocessing" / "outputs" / "towerA_dataset.csv"
OUTPUT_DIR   = BASE_DIR / "outputs"
OUTPUT_PATH  = OUTPUT_DIR / "features_handcrafted_AE.csv"

def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            f"Run 01_preprocessing/preprocess.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"  Rows: {len(df):,}")

    print("Extracting features...")
    A = extract_group_a(df, url_col="url_norm", etld1_col="etld1")
    B = extract_group_b(df, url_col="url_norm")
    C = extract_group_c(df, url_col="url_norm")
    D = extract_group_d(df, url_col="url_norm")
    E = extract_group_e(df, url_col="url_norm", etld1_col="etld1")

    X = pd.concat([A, B, C, D, E], axis=1)
    out = pd.concat([df[["url_norm", "label", "source", "etld1"]], X], axis=1)

    feature_cols = [c for c in out.columns if c not in {"url_norm", "label", "source", "etld1"}]
    assert len(feature_cols) == 35, (
        f"Expected 35 features, got {len(feature_cols)}: {feature_cols}"
    )

    out.to_csv(OUTPUT_PATH, index=False)
    print(f" Features extracted: {len(feature_cols)} columns")
    print(f" Output written to: {OUTPUT_PATH}")

    nan_counts = out[feature_cols].isna().sum()
    if nan_counts.any():
        print("WARNING: NaN values detected in features:")
        for col, cnt in nan_counts[nan_counts > 0].items():
            print(f" {col}: {cnt} NaN values")
    else:
        print(" NaN check: OK (no missing feature values)")

    print("\nFeature groups:")
    print(f" Group A (structural): {len(A.columns)} features")
    print(f" Group B (characters): {len(B.columns)} features")
    print(f" Group C (entropy): {len(C.columns)} features")
    print(f" Group D (tokens): {len(D.columns)} features")
    print(f" Group E (brand): {len(E.columns)} features")

if __name__ == "__main__":
    main()