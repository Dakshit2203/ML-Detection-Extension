"""
Tower A - Phase 2: Feature Separability Analysis

Computes univariate statistics for each of the 35 handcrafted features to confirm that each group contributes
discriminative signal before committing to model training. Analysis only - no data is modified.

Three outputs:
  descriptive_feature_metrics.csv  per-feature mean, std, Cohen's d, KS, AUC
  group_separation_summary.csv     group-level aggregation of the above
  stability_feature_summary.csv    AUC and KS variance across 5 random subsamples

The stability check is included to confirm that separability metrics are consistent across the dataset rather than
driven by a small cluster of URLs.

Inputs:  02_feature_engineering/outputs/features_handcrafted_AE.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, rankdata

BASE_DIR   = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "outputs" / "features_handcrafted_AE.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

def cohens_d(x0: np.ndarray, x1: np.ndarray) -> float:
    """Pooled-SD Cohen's d between the benign (x0) and phishing (x1) distributions."""
    x0, x1 = x0.astype(float), x1.astype(float)
    n0, n1  = len(x0), len(x1)
    if n0 < 2 or n1 < 2:
        return np.nan
    s0 = np.nanstd(x0, ddof=1)
    s1 = np.nanstd(x1, ddof=1)
    sp = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / max(n0+n1-2, 1))
    if sp == 0 or np.isnan(sp):
        return 0.0
    return float((np.nanmean(x1) - np.nanmean(x0)) / sp)

def single_feature_auc(scores: np.ndarray, y: np.ndarray) -> float:
    """
    Rank-based univariate AUC - a model-free separability measure.

    Computed via the Mann-Whitney U statistic rather than fitting a classifier, so it reflects the feature's intrinsic
    discriminative power without overfitting.
    """
    scores = scores.astype(float)
    y      = y.astype(int)
    mask   = ~np.isnan(scores)
    scores, y = scores[mask], y[mask]
    if len(np.unique(y)) < 2:
        return np.nan
    r     = rankdata(scores)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    R_pos = np.sum(r[y == 1])
    U     = R_pos - n_pos*(n_pos+1)/2
    return float(U / (n_pos * n_neg))

def feature_group(name: str) -> str:
    group_a = {"url_len","host_len","path_len","query_len","num_dots","num_slashes",
                "num_subdomains","etld1_missing","has_query","has_fragment","has_port","is_ip_host"}
    group_b = {"digit_ratio","alpha_ratio","symbol_ratio","upper_ratio_pq",
                "count_at","count_dash","count_underscore","count_percent","count_equal","count_question"}
    group_c = {"entropy_full","entropy_host","entropy_path"}
    group_d = {"num_tokens","avg_token_len","max_token_len",
                "brand_keyword_present","auth_token_present","action_token_present"}
    group_e = {"brand_in_subdomain","brand_in_path","brand_mismatch","deep_subdomain"}
    if name in group_a: return "A"
    if name in group_b: return "B"
    if name in group_c: return "C"
    if name in group_d: return "D"
    if name in group_e: return "E"
    return "UNKNOWN"

def compute_metrics(df: pd.DataFrame, label_col: str,
                    feature_cols: list[str]) -> pd.DataFrame:
    y    = df[label_col].values
    rows = []
    for col in feature_cols:
        x  = df[col].values
        x0 = x[y == 0]
        x1 = x[y == 1]
        try:
            ks = ks_2samp(x0[~np.isnan(x0)], x1[~np.isnan(x1)]).statistic
        except Exception:
            ks = np.nan
        rows.append({
            "feature": col,
            "group": feature_group(col),
            "mean_benign": float(np.nanmean(x0)),
            "median_benign": float(np.nanmedian(x0)),
            "std_benign": float(np.nanstd(x0)),
            "iqr_benign": float(np.nanpercentile(x0,75)-np.nanpercentile(x0,25)),
            "mean_phish": float(np.nanmean(x1)),
            "median_phish": float(np.nanmedian(x1)),
            "std_phish": float(np.nanstd(x1)),
            "iqr_phish": float(np.nanpercentile(x1,75)-np.nanpercentile(x1,25)),
            "cohens_d": cohens_d(x0, x1),
            "ks_stat": float(ks),
            "single_feature_auc": single_feature_auc(x, y),
        })
    return pd.DataFrame(rows)

def compute_group_summary(metrics: pd.DataFrame, auc_thresh: float = 0.55) -> pd.DataFrame:
    g   = metrics.groupby("group", dropna=False)
    out = g.agg(
        n_features = ("feature", "count"),
        auc_mean = ("single_feature_auc", "mean"),
        auc_median = ("single_feature_auc", "median"),
        auc_max = ("single_feature_auc", "max"),
        ks_mean = ("ks_stat", "mean"),
        ks_median = ("ks_stat", "median"),
        d_mean = ("cohens_d", "mean"),
    ).reset_index()
    pct_above = (
        metrics.assign(_flag=metrics["single_feature_auc"] >= auc_thresh)
        .groupby("group")["_flag"]
        .mean()
        .reset_index(name=f"pct_auc_ge_{auc_thresh}")
    )
    return out.merge(pct_above, on="group", how="left").sort_values("auc_mean", ascending=False)

def stability_check(df: pd.DataFrame, label_col: str, feature_cols: list[str],
                    frac: float = 0.8, repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Recompute AUC and KS on 5 random 80% subsamples and report mean and std.

    High std would suggest the metric is driven by a small number of influential URLs rather than a genuine
    population-level pattern.
    """
    aucs, kss = [], []
    rng = np.random.default_rng(seed)
    n = len(df)
    for i in range(repeats):
        idx = rng.choice(n, size=int(frac*n), replace=False)
        sub = df.iloc[idx].copy()
        m = compute_metrics(sub, label_col, feature_cols)[["feature","ks_stat","single_feature_auc"]]
        m = m.rename(columns={"ks_stat": f"ks_{i}", "single_feature_auc": f"auc_{i}"}
                       ).set_index("feature")
        aucs.append(m[[f"auc_{i}"]])
        kss.append(m[[f"ks_{i}"]])
    auc_mat = pd.concat(aucs, axis=1)
    ks_mat = pd.concat(kss, axis=1)
    out = pd.DataFrame(index=auc_mat.index)
    out["group"] = [feature_group(f) for f in out.index]
    out["auc_mean"] = auc_mat.mean(axis=1)
    out["auc_std"] = auc_mat.std(axis=1)
    out["ks_mean"] = ks_mat.mean(axis=1)
    out["ks_std"] = ks_mat.std(axis=1)
    return out.reset_index().rename(columns={"index":"feature"}).sort_values("auc_mean", ascending=False)

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature matrix not found: {INPUT_PATH}\n"
            f"Run extract_features.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading features from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    label_col = "label"
    feature_cols = [c for c in df.columns if c not in {"url_norm","label","source","etld1"}]
    print(f" Rows: {len(df):,} | Features: {len(feature_cols)}")

    metrics = compute_metrics(df, label_col, feature_cols).sort_values("single_feature_auc", ascending=False)
    metrics.to_csv(OUTPUT_DIR / "descriptive_feature_metrics.csv", index=False)
    print(f"  Written: {OUTPUT_DIR / 'descriptive_feature_metrics.csv'}")

    group_sum = compute_group_summary(metrics, auc_thresh=0.55)
    group_sum.to_csv(OUTPUT_DIR / "group_separation_summary.csv", index=False)
    print(f" Written: {OUTPUT_DIR / 'group_separation_summary.csv'}")

    stab = stability_check(df, label_col, feature_cols, frac=0.8, repeats=5, seed=42)
    stab.to_csv(OUTPUT_DIR / "stability_feature_summary.csv", index=False)
    print(f" Written: {OUTPUT_DIR / 'stability_feature_summary.csv'}")

    print("\nTop 5 features by single_feature_auc:")
    for _, row in metrics.head(5).iterrows():
        print(f"  {row['feature']:<30} AUC={row['single_feature_auc']:.4f}  "
              f"KS={row['ks_stat']:.4f}  d={row['cohens_d']:.4f}")

if __name__ == "__main__":
    main()