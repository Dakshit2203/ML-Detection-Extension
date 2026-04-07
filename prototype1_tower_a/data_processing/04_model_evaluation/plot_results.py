"""
Tower A - Phase 4: Generate Evaluation Figures

Produces publication-ready PNG figures from summary_results.csv and the
per-run JSON files written by evaluate.py.

Figures 1–5 are generated from summary_results.csv (threshold-free metrics
and efficiency). Figures 6–7 require the individual JSON files because they
use policy metrics (Recall@FPR≤2%) not stored in the summary CSV.

Note: evaluate.py writes policy metrics under the key "test_policy_metrics".
plot_results.py reads that same key. Do not rename either end independently.

Outputs: 04_model_evaluation/outputs/figures/ (300 DPI PNG)

Usage:
  cd data_processing/
  python 04_model_evaluation/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SUMMARY_CSV = HERE / "outputs" / "summary_results.csv"
JSON_DIR = HERE / "outputs" / "json"
OUT_DIR = HERE / "outputs" / "figures"

SEED = 42

MODEL_ORDER = ["lr", "rf", "hgb", "sgd"]
MODEL_LABELS = {
    "lr": "Logistic Regression (A–E)",
    "rf": "Random Forest (A–E)",
    "hgb": "Gradient Boosting (A–E)",
    "sgd": "SGD (n-grams)",
}
SPLIT_ORDER = ["random", "etld1"]
SPLIT_LABELS = {"random": "Random", "etld1": "eTLD+1"}

FINAL_MODEL_KEY = "lr"
FINAL_SPLIT_KEY = "etld1"

plt.rcParams.update({
    "savefig.dpi": 300,
    "figure.dpi": 120,
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
})

def save_fig(fig, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)

def load_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing: {SUMMARY_CSV}\nRun run_all.py first.")
    df = pd.read_csv(SUMMARY_CSV)
    if "seed" in df.columns:
        df = df[df["seed"] == SEED].copy()
    df = df[df["split"].isin(SPLIT_ORDER) & df["model"].isin(MODEL_ORDER)].copy()
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["split_label"] = df["split"].map(SPLIT_LABELS)
    if "vectorizer_size_mb" in df.columns:
        df["vectorizer_size_mb"] = df["vectorizer_size_mb"].fillna(0.0)
    return df

def load_policy_from_json(json_dir: Path) -> pd.DataFrame:
    """
    Read per-run policy metrics from the JSON files produced by evaluate.py.
    The key "test_policy_metrics" must match exactly what evaluate.py writes.
    """
    rows = []
    if not json_dir.exists():
        return pd.DataFrame(rows)
    for fp in sorted(json_dir.glob("*.json")):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = obj.get("meta", {})
        split = meta.get("split")
        model = meta.get("model")
        seed = meta.get("seed")
        if split not in SPLIT_ORDER or model not in MODEL_ORDER:
            continue
        if seed is not None and int(seed) != SEED:
            continue
        policy = obj.get("test_policy_metrics", {})
        rec_block = policy.get("recall_at_fpr_cap", None)
        rows.append({
            "split": split,
            "model": model,
            "seed": seed,
            "recall_at_fpr_cap": rec_block.get("recall") if isinstance(rec_block, dict) else None,
        })
    return pd.DataFrame(rows)

def pivot_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    p = df.pivot_table(index="model", columns="split", values=metric_col, aggfunc="first")
    return p.reindex(MODEL_ORDER)


def annotate_bars(ax, fmt: str = "{:.4f}", pad: int = 3) -> None:
    for bar in ax.patches:
        h = bar.get_height()
        if h is None:
            continue
        ax.annotate(fmt.format(h),
                    (bar.get_x() + bar.get_width() / 2.0, h),
                    ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, pad), fontsize=10)

def fig1_pr_auc_by_split(df: pd.DataFrame) -> None:
    pivot = pivot_metric(df, "pr_auc")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x, w = list(range(len(pivot.index))), 0.38
    b1 = ax.bar([i - w/2 for i in x], pivot["random"], width=w, label="Random")
    b2 = ax.bar([i + w/2 for i in x], pivot["etld1"], width=w, label="eTLD+1")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in pivot.index], rotation=15, ha="right")
    ax.set_ylabel("PR-AUC")
    ax.set_title("PR-AUC by Model: Random vs eTLD+1 Split")
    ax.set_ylim(max(0.80, min(pivot.min()) - 0.03), 1.00)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.4f}",
                        (b.get_x() + b.get_width()/2, b.get_height()),
                        ha="center", va="bottom",
                        textcoords="offset points", xytext=(0, 3), fontsize=10)
    save_fig(fig, "fig1_pr_auc_random_vs_etld1")
    
def fig2_pr_auc_drop(df: pd.DataFrame) -> None:
    """Generalisation gap: PR-AUC drop from random to domain-separated split."""
    pivot = pivot_metric(df, "pr_auc")
    drop = pivot["random"] - pivot["etld1"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(drop.index)), drop.values)
    ax.set_xticks(range(len(drop.index)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in drop.index], rotation=15, ha="right")
    ax.set_ylabel("Δ PR-AUC (Random − eTLD+1)")
    ax.set_title("Generalisation Drop Under Domain-Separated Split")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    annotate_bars(ax, fmt="{:.4f}")
    save_fig(fig, "fig2_pr_auc_drop")

def fig3_recall_vs_fpr_etld1(df: pd.DataFrame) -> None:
    d = df[df["split"] == "etld1"].copy().set_index("model").loc[MODEL_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.scatter(d["fpr"], d["recall"], s=80)
    for _, row in d.iterrows():
        ax.annotate(MODEL_LABELS[row["model"]],
                    (row["fpr"], row["recall"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=11)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("Recall")
    ax.set_title("Recall vs FPR (eTLD+1 Split)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(0.0, max(d["fpr"].max() * 1.15, 0.10))
    ax.set_ylim(min(d["recall"].min() - 0.03, 0.80), 0.93)
    save_fig(fig, "fig3_recall_vs_fpr_etld1")

def fig4_perf_vs_latency_etld1(df: pd.DataFrame) -> None:
    d = df[df["split"] == "etld1"].copy().set_index("model").loc[MODEL_ORDER].reset_index()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.scatter(d["end_to_end_ms_per_url"], d["pr_auc"], s=80)
    for _, row in d.iterrows():
        ax.annotate(MODEL_LABELS[row["model"]],
                    (row["end_to_end_ms_per_url"], row["pr_auc"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=11)
    ax.set_xlabel("End-to-end latency (ms / URL)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Performance vs Deployment Cost (eTLD+1 Split)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlim(0, d["end_to_end_ms_per_url"].max() * 1.20)
    ax.set_ylim(min(d["pr_auc"].min() - 0.02, 0.85), 0.99)
    save_fig(fig, "fig4_prauc_vs_latency_etld1")

def fig5_memory_footprint_log(df: pd.DataFrame) -> None:
    d = df[df["split"] == "etld1"].copy().set_index("model").loc[MODEL_ORDER].reset_index()
    d["vectorizer_size_mb"] = d["vectorizer_size_mb"].fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x, w = list(range(len(d))), 0.38
    b1 = ax.bar([i - w/2 for i in x], d["model_size_mb"], width=w, label="Model (MB)")
    b2 = ax.bar([i + w/2 for i in x], d["vectorizer_size_mb"], width=w, label="Vectoriser (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in d["model"]], rotation=15, ha="right")
    ax.set_ylabel("Size (MB) [log scale]")
    ax.set_title("Model / Vectoriser Memory Footprint")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    # Log scale prevents the Random Forest bar from visually flattening LR and SGD.
    ax.set_yscale("log")
    
    def fmt_mb(v: float) -> str:
        if v >= 10: return f"{v:.1f}"
        if v >= 1: return f"{v:.2f}"
        if v >= 0.01: return f"{v:.3f}"
        return f"{v:.4f}"

    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if h and h > 0:
                ax.annotate(fmt_mb(h),
                            (b.get_x() + b.get_width()/2, h),
                            ha="center", va="bottom",
                            textcoords="offset points", xytext=(0, 3), fontsize=10)
    save_fig(fig, "fig5_size_mb_logscale")

def fig6_confusion_matrix_final(df: pd.DataFrame) -> None:
    d = df[(df["split"] == FINAL_SPLIT_KEY) & (df["model"] == FINAL_MODEL_KEY)].copy()
    if d.empty:
        print("Skipping fig6: final model row not found in summary_results.csv")
        return
    r = d.iloc[0]
    cm = [[int(r["tn"]), int(r["fp"])],
          [int(r["fn"]), int(r["tp"])]]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm)
    ax.set_title(f"Confusion Matrix (eTLD+1) - {MODEL_LABELS[FINAL_MODEL_KEY]}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign (0)", "Phish (1)"])
    ax.set_yticklabels(["Benign (0)", "Phish (1)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, "fig6_confusion_matrix_final_etld1")

def fig7_recall_at_fpr_cap_etld1(df_policy: pd.DataFrame) -> None:
    """Recall@FPR≤2% for each model under the eTLD+1 split."""
    if df_policy.empty or "recall_at_fpr_cap" not in df_policy.columns:
        print("Skipping fig7: no policy JSON data found")
        return
    d = (df_policy[df_policy["split"] == "etld1"]
         .dropna(subset=["recall_at_fpr_cap"])
         .set_index("model").reindex(MODEL_ORDER).reset_index()
         .dropna(subset=["recall_at_fpr_cap"]))
    if d.empty:
        print("Skipping fig7: recall_at_fpr_cap missing for etld1 split")
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(range(len(d)), d["recall_at_fpr_cap"].astype(float).values)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in d["model"]], rotation=15, ha="right")
    ax.set_ylabel("Recall (test)")
    ax.set_title("Recall at Fixed FPR ≤ 2% - eTLD+1 Split")
    ax.set_ylim(max(0.0, d["recall_at_fpr_cap"].min() - 0.10), 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    annotate_bars(ax, fmt="{:.4f}")
    save_fig(fig, "fig7_recall_at_fpr_cap_etld1")

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_summary()
    df_policy = load_policy_from_json(JSON_DIR)

    print(f"Loaded {len(df)} summary rows:")
    print(df[["split","model","regime","pr_auc","roc_auc"]].sort_values(["split","model"]).to_string(index=False))
    print(f"\nLoaded {len(df_policy)} policy JSON rows")

    fig1_pr_auc_by_split(df)
    fig2_pr_auc_drop(df)
    fig3_recall_vs_fpr_etld1(df)
    fig4_perf_vs_latency_etld1(df)
    fig5_memory_footprint_log(df)
    fig6_confusion_matrix_final(df)
    fig7_recall_at_fpr_cap_etld1(df_policy)

    print(f"\nFigures saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
