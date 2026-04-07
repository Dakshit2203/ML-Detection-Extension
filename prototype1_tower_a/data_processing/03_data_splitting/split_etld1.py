"""
Tower A - Phase 3:  eTLD+1 Domain-Level Split (Primary Evaluation Split)

Partitions towerA_dataset.csv by grouping all URLs sharing the same eTLD+1 into the same split. This prevents a URL 
from a domain seen during training from appearing in the test set under a different path, which would allow the model 
to memorise domain reputations rather than learn transferable URL structure patterns. All reported evaluation metrics 
use this split.

Split ratios: train=70%, val=15%, test=15% (applied at the domain group level)
Stratification is by domain-level label - the majority label across all URLs for that eTLD+1 - to maintain class 
balance across splits.

URLs with missing eTLD+1 are assigned to train only. Allowing them into val or test would be a form of leakage because 
a URL with no resolvable suffix is structurally atypical and should not influence threshold selection or reporting.

The random URL-level split (split_random.py) is kept for comparison only.

Inputs:  01_preprocessing/outputs/towerA_dataset.csv
Outputs: 03_data_splitting/outputs/etld1/
  splits_manifest.csv   row-level split assignment with eTLD+1 metadata
  train.csv, val.csv, test.csv
  split_summary.txt     leakage check and row/group counts

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PROC = BASE_DIR.parent
DATASET_PATH = DATA_PROC / "01_preprocessing" / "outputs" / "towerA_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "etld1"

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

    if "etld1" not in df.columns:
        raise RuntimeError("etld1 column missing. Run preprocess.py first.")

    etld1 = df["etld1"]
    etld1_missing = etld1.isna() | (etld1.astype(str).str.strip() == "")
    df["_etld1_missing"] = etld1_missing
    miss_rows = int(etld1_missing.sum())
    print(f" Missing eTLD+1: {miss_rows} rows (assigned to train only)")

    # Build one row per eTLD+1 group, labelled by majority class.
    df_non = df.loc[~df["_etld1_missing"], ["etld1", "label"]].copy()
    grp    = (
        df_non.groupby("etld1")["label"]
        .agg(lambda x: int(x.mode().iloc[0]))
        .reset_index()
    )

    rng = np.random.default_rng(seed)

    def split_groups(values: np.ndarray) -> tuple[set, set, set]:
        values = values.copy()
        rng.shuffle(values)
        n = len(values)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        return (
            set(values[:n_train]),
            set(values[n_train:n_train + n_val]),
            set(values[n_train + n_val:]),
        )

    # Stratify separately by class so the test set reflects the true class ratio.
    g0 = grp.loc[grp["label"] == 0, "etld1"].to_numpy()
    g1 = grp.loc[grp["label"] == 1, "etld1"].to_numpy()

    train0, val0, test0 = split_groups(g0)
    train1, val1, test1 = split_groups(g1)

    train_g = train0 | train1
    val_g = val0 | val1
    test_g = test0 | test1

    split = pd.Series(index=df.index, dtype="object")
    split.loc[df["_etld1_missing"]] = "train"

    idx_non = df.index[~df["_etld1_missing"]]
    et = df.loc[idx_non, "etld1"].astype(str)
    split.loc[idx_non[et.isin(list(train_g))]] = "train"
    split.loc[idx_non[et.isin(list(val_g))]] = "val"
    split.loc[idx_non[et.isin(list(test_g))]] = "test"

    if split.isna().any():
        raise RuntimeError("Unassigned rows detected - split logic error.")

    mapping = pd.DataFrame({
        "row_id": df.index.values,
        "split": split.values,
        "label": df["label"].values,
        "etld1": df["etld1"].values,
        "etld1_missing": df["_etld1_missing"].astype(int).values,
    })
    mapping.to_csv(OUTPUT_DIR / "splits_manifest.csv", index=False)

    ml = df[["url_norm", "label"]].copy()
    ml["split"] = split.values

    ml.loc[ml["split"] == "train", ["url_norm", "label"]].to_csv(OUTPUT_DIR / "train.csv", index=False)
    ml.loc[ml["split"] == "val", ["url_norm", "label"]].to_csv(OUTPUT_DIR / "val.csv", index=False)
    ml.loc[ml["split"] == "test", ["url_norm", "label"]].to_csv(OUTPUT_DIR / "test.csv", index=False)

    total_rows = len(df)
    row_counts = mapping.groupby(["split", "label"]).size().unstack(fill_value=0)
    non_missing = mapping.loc[mapping["etld1_missing"] == 0].copy()
    uniq = non_missing.drop_duplicates(subset=["etld1"])
    group_counts = uniq.groupby(["split", "label"]).size().unstack(fill_value=0)

    train_et = set(uniq.loc[uniq["split"] == "train", "etld1"].astype(str))
    val_et = set(uniq.loc[uniq["split"] == "val", "etld1"].astype(str))
    test_et = set(uniq.loc[uniq["split"] == "test", "etld1"].astype(str))

    summary = [
        "=== Tower A Primary Split Summary (Domain-level by eTLD+1) ===",
        f"Seed: {seed}",
        f"Target fractions: train={train_frac:.2f}  val={val_frac:.2f}  "
        f"test={1 - train_frac - val_frac:.2f}",
        "",
        f"Total rows: {total_rows:,}",
        f"Rows with missing eTLD+1: {miss_rows} ({_pct(miss_rows, total_rows):.2f}%)",
        "Missing eTLD+1 policy: assigned to TRAIN only",
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
        "[Unique eTLD+1 groups by split x label (non-missing only)]",
        group_counts.to_string(),
        "",
        "[Leakage check: eTLD+1 intersections across splits]",
        f" train ∩ val: {len(train_et & val_et)} (expected 0)",
        f" train ∩ test: {len(train_et & test_et)} (expected 0)",
        f" val ∩ test: {len(val_et & test_et)} (expected 0)",
        "",
        "[Files written]",
        str(OUTPUT_DIR / "splits_manifest.csv"),
        str(OUTPUT_DIR / "train.csv"),
        str(OUTPUT_DIR / "val.csv"),
        str(OUTPUT_DIR / "test.csv"),
    ]
    (OUTPUT_DIR / "split_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print(f"\nSplit complete (eTLD+1 domain-level):")
    for s in ["train", "val", "test"]:
        n_s = int((mapping["split"] == s).sum())
        print(f"  {s}: {n_s:,} rows ({_pct(n_s, total_rows):.1f}%)")
    print(f"\nLeakage check (all must be 0):")
    print(f" train ∩ val: {len(train_et & val_et)}")
    print(f" train ∩ test: {len(train_et & test_et)}")
    print(f" val ∩ test: {len(val_et & test_et)}")
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(seed=42, train_frac=0.70, val_frac=0.15)
