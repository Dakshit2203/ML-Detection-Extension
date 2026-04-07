"""
Tower A - Phase 3: Random URL-Level Split (Baseline Comparison Only)

Produces a stratified random URL-level split retained solely to quantify performance inflation relative to the 
domain-level split. Because the same eTLD+1 can appear in both train and test, the model may partly memorise domain 
reputations, inflating reported metrics.

Comparing PR-AUC on this split against the eTLD+1 split (split_etld1.py) provides the generalisation gap figure used 
in the dissertation evaluation.

Inputs:  01_preprocessing/outputs/towerA_dataset.csv
Outputs: 03_data_splitting/outputs/random_baseline/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PROC = BASE_DIR.parent
DATASET_PATH = DATA_PROC / "01_preprocessing" / "outputs" / "towerA_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "random_baseline"

def _pct(x: int, total: int) -> float:
    return 100.0 * float(x) / float(max(total, 1))


def main(seed: int = 42, train_frac: float = 0.70, val_frac: float = 0.15) -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            f"Run 01_preprocessing/preprocess.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f" Total rows: {len(df):,}")

    if "url_norm" not in df.columns or "label" not in df.columns:
        raise RuntimeError("Expected columns missing: url_norm and/or label.")

    rng = np.random.default_rng(seed)

    # Stratify by label to preserve the 50/50 class ratio in each split.
    idx0 = df.index[df["label"] == 0].to_numpy().copy()
    idx1 = df.index[df["label"] == 1].to_numpy().copy()
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    
    def assign(idx: np.ndarray) -> tuple:
        n = len(idx)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    tr0, va0, te0 = assign(idx0)
    tr1, va1, te1 = assign(idx1)

    split = pd.Series(index=df.index, dtype="object")
    for idx, label in [(tr0, "train"), (va0, "val"), (te0, "test"),
                       (tr1, "train"), (va1, "val"), (te1, "test")]:
        split.loc[idx] = label

    if split.isna().any():
        raise RuntimeError("Unassigned rows detected - split logic error.")

    mapping = pd.DataFrame({
        "row_id": df.index.values,
        "split": split.values,
        "label": df["label"].values,
    })
    mapping.to_csv(OUTPUT_DIR / "splits_manifest.csv", index=False)

    ml = df[["url_norm", "label"]].copy()
    ml["split"] = split.values
    ml.loc[ml["split"] == "train", ["url_norm", "label"]].to_csv(OUTPUT_DIR / "train.csv", index=False)
    ml.loc[ml["split"] == "val", ["url_norm", "label"]].to_csv(OUTPUT_DIR / "val.csv", index=False)
    ml.loc[ml["split"] == "test", ["url_norm", "label"]].to_csv(OUTPUT_DIR / "test.csv", index=False)

    total_rows = len(df)
    row_counts = mapping.groupby(["split", "label"]).size().unstack(fill_value=0)

    summary = [
        "=== Tower A Random Split Summary (URL-level baseline) ===",
        f"Seed: {seed}",
        f"Target fractions: train={train_frac:.2f}  val={val_frac:.2f}  "
        f"test={1 - train_frac - val_frac:.2f}",
        "",
        "NOTE: Random URL-level split - the same eTLD+1 may appear in train and test.",
        "Metrics on this split are inflated by domain memorisation.",
        "",
        f"Total rows: {total_rows:,}",
        "",
        "[Row counts by split x label]",
        row_counts.to_string(),
        "",
        "[Row percentages by split]",
    ]
    for s in ["train", "val", "test"]:
        n_s = int((mapping["split"] == s).sum())
        summary.append(f" {s}: {n_s:,} ({_pct(n_s, total_rows):.2f}%)")
    summary += [
        "",
        "[Files written]",
        str(OUTPUT_DIR / "splits_manifest.csv"),
        str(OUTPUT_DIR / "train.csv"),
        str(OUTPUT_DIR / "val.csv"),
        str(OUTPUT_DIR / "test.csv"),
    ]
    (OUTPUT_DIR / "split_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print(f"\nSplit complete (random URL-level - for baseline comparison only):")
    for s in ["train", "val", "test"]:
        n_s = int((mapping["split"] == s).sum())
        print(f"  {s}: {n_s:,} rows ({_pct(n_s, total_rows):.1f}%)")
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(seed=42, train_frac=0.70, val_frac=0.15)