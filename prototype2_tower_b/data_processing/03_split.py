"""
Tower B - Data Processing - 03_split.py

Creates the Tower B train/validation/test splits by reusing the eTLD+1 domain assignment mapping from the Prototype 1 (Tower A) split.

Why the split must be shared with Prototype 1
The Prototype 3 fusion evaluation computes p_fused = 0.75·pA + 0.25·pB for each URL in the test set and then measures
the fused PR-AUC. This is only valid if pA and pB are produced from the same set of test URLs - i.e., the Tower A and
Tower B test sets must be identical.

If Tower B were split independently (e.g., by running a fresh eTLD+1 split on the metadata CSV), the test sets would be
different and the fusion scores could not be directly compared or combined.

Approach: reuse the eTLD+1 assignment file from Prototype 1

The Prototype 1 Phase 3 split script (split_etld1.py) writes three files:
  data_processing/03_data_splitting/outputs/etld1/
    train_etld1.csv   - 71,715 rows
    val_etld1.csv     - 11,035 rows
    test_etld1.csv    -  9,916 rows

Each file contains a 'url_norm' column. This script joins the Tower B metadata CSV against these URL lists to create
the Tower B split files.

The join is on 'url_norm'. URLs in the metadata CSV that are not found in a given split file are silently dropped - this
can happen if a small number of URLs failed metadata collection entirely and were not written to the CSV.
The drop count is reported so it can be noted in the dissertation.

Outputs
-------
  outputs/splits/train_B.csv  - Tower B training set
  outputs/splits/val_B.csv    - Tower B validation set
  outputs/splits/test_B.csv   - Tower B test set 
"""


from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Paths
# Tower B metadata: the full dataset from collection
DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_CSV = DATA_DIR / "towerB_metadata_master.csv"

# Prototype 1 eTLD+1 split files - source of truth for domain assignments
P1_SPLIT_DIR = (
    Path(__file__).resolve().parents[2]
    / "prototype1_tower_a" / "data_processing"
    / "03_data_splitting" / "outputs" / "etld1"
)

# Output directory for Tower B split files
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "splits"

# Feature spec: used to validate that the split files contain the right columns
SPEC_PATH = Path(__file__).resolve().parent / "outputs" / "feature_spec_B.json"

def load_url_set(split_file: Path) -> set:
    """
    Reads the url_norm column from a Prototype 1 split file and returns a set of normalised URL strings for fast
    membership testing.
    """
    if not split_file.exists():
        print(f"ERROR: Split file not found: {split_file}")
        print("Ensure Prototype 1 Phase 3 (split_etld1.py) has been run.")
        sys.exit(1)
    df = pd.read_csv(split_file, usecols=["url_norm"])
    return set(df["url_norm"].dropna().str.strip())

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load feature spec
    if not SPEC_PATH.exists():
        print(f"ERROR: Feature spec not found: {SPEC_PATH}")
        print("Run 02_feature_spec.py first.")
        sys.exit(1)

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    features = spec["features"]
    print(f"Feature spec loaded: {len(features)} features")

    # Load metadata CSV
    print(f"Reading metadata CSV: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        print(f"ERROR: Metadata CSV not found: {INPUT_CSV}")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded: {len(df):,} rows")

    if "url_norm" not in df.columns:
        print("ERROR: 'url_norm' column not found in metadata CSV.")
        sys.exit(1)

    # Normalise whitespace in url_norm for consistent join
    df["url_norm"] = df["url_norm"].str.strip()

    # Load Prototype 1 split URL sets
    print(f"\nLoading Prototype 1 split files from: {P1_SPLIT_DIR}")
    train_urls = load_url_set(P1_SPLIT_DIR / "train.csv")
    val_urls = load_url_set(P1_SPLIT_DIR / "val.csv")
    test_urls = load_url_set(P1_SPLIT_DIR / "test.csv")

    print(f" Tower A train set: {len(train_urls):,} URLs")
    print(f" Tower A val set: {len(val_urls):,} URLs")
    print(f" Tower A test set: {len(test_urls):,} URLs")

    # Assign rows to splits
    # Each row in the metadata CSV is assigned to the split that contains its url_norm. Rows not present in any
    # split (rare, due to metadata collection failures) are dropped and the count is reported.
    df_train = df[df["url_norm"].isin(train_urls)].copy()
    df_val = df[df["url_norm"].isin(val_urls)].copy()
    df_test = df[df["url_norm"].isin(test_urls)].copy()

    total_assigned = len(df_train) + len(df_val) + len(df_test)
    dropped = len(df) - total_assigned

    print(f"\nSplit assignment results:")
    print(f" train: {len(df_train):,} rows")
    print(f" val: {len(df_val):,} rows")
    print(f" test: {len(df_test):,} rows")
    if dropped > 0:
        print(f" dropped (not in any split): {dropped:,} rows")

    # Validate label balance
    print("\nLabel distribution per split:")
    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        if "label" in split_df.columns:
            vc = split_df["label"].value_counts().sort_index()
            print(f" {name}: {dict(vc)}")

    # Validate feature columns
    # Confirm that all 18 feature columns in the spec are present in each split.
    # If any are missing, the training script would fail silently or produce incorrect results.
    print("\nValidating feature columns against feature spec...")
    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        missing_cols = [f for f in features if f not in split_df.columns]
        if missing_cols:
            print(f" WARNING [{name}]: {len(missing_cols)} feature columns missing: {missing_cols}")
        else:
            print(f" [{name}]: all {len(features)} feature columns present ✓")

    # Write split files
    # Save full rows (not just feature columns) so that url_norm and label are available for evaluation and diagnosis.
    df_train.to_csv(OUTPUT_DIR / "train_B.csv", index=False)
    df_val.to_csv( OUTPUT_DIR / "val_B.csv", index=False)
    df_test.to_csv( OUTPUT_DIR / "test_B.csv", index=False)

    print(f"\nSplit files written to: {OUTPUT_DIR}")
    print(" train_B.csv")
    print(" val_B.csv")
    print(" test_B.csv")
    print("\nTest set shares the same domain-level split as Tower A.")
    print("This ensures Prototype 3 fusion evaluation is valid.")


if __name__ == "__main__":
    main()
