"""
Prototype 3 - Fused System - evaluation/evaluate_fused_system.py

Produces the three-way comparative evaluation of Tower A, Tower B, and the fused system on the shared eTLD+1 held-out
test set (9,916 URLs).

This script extends the prototype evaluation work. Thresholds, metrics, and weights are read directly from committed 
artifacts


Models are re-scored because per-URL scores were not stored during training. After re-scoring, aggregate metrics are
verified against the stored reports. The script exits rather than producing inconsistent figures if they diverge.


All outputs are written to prototype3_fused/evaluation/outputs/:
  evaluation_report.txt         dissertation-ready metrics table
  predictions_test.csv          per-URL p_A, p_B, p_fused
  fig1_pr_curves.png            PR curves: Tower A vs Tower B vs Fused
  fig2_roc_curves.png           ROC curves: Tower A vs Tower B vs Fused
  fig3_score_distributions.png  Score distributions by label
  fig4_disagreement_scatter.png p_A vs p_B scatter coloured by label
  fig5_tower_a_features.png     Tower A top-15 feature attributions
  fig6_tower_b_importance.png   Tower B permutation importances
  fig7_confusion_matrices.png   Confusion matrices at stored thresholds
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")


# Paths - all relative to the repository root
# parents[0] = evaluation/  parents[1] = prototype3_fused/  parents[2] = repo root

REPO_ROOT = Path(__file__).resolve().parents[2]
P3_ROOT = Path(__file__).resolve().parents[1]
P1_ROOT = REPO_ROOT / "prototype1_tower_a"
P2_ROOT = REPO_ROOT / "prototype2_tower_b"

MODEL_A_PATH = P1_ROOT / "backend" / "artifacts" / "model_lr.joblib"
SPEC_A_PATH = P1_ROOT / "backend" / "artifacts" / "feature_spec.json"
REPORT_A_PATH = P1_ROOT / "data_processing" / "05_final_model" / "outputs" / "final_model_report.json"
FEATURES_AE_CSV = P1_ROOT / "data_processing" / "02_feature_engineering" / "outputs" / "features_handcrafted_AE.csv"
TEST_A_CSV = P1_ROOT / "data_processing" / "03_data_splitting" / "outputs" / "etld1" / "test.csv"
VAL_A_CSV = P1_ROOT / "data_processing" / "03_data_splitting" / "outputs" / "etld1" / "val.csv"

MODEL_B_PATH = P2_ROOT / "artifacts" / "towerB" / "hgb.joblib"
SPEC_B_PATH = P2_ROOT / "artifacts" / "towerB" / "feature_spec_B.json"
IMP_B_PATH = P2_ROOT / "artifacts" / "towerB" / "feature_importances_B.json"
REPORT_B_PATH = P2_ROOT / "data_processing" / "outputs" / "04_model_report.json"
TEST_B_CSV = P2_ROOT / "data_processing" / "outputs" / "splits" / "test_B.csv"
VAL_B_CSV = P2_ROOT / "data_processing" / "outputs" / "splits" / "val_B.csv"

OUT_DIR = Path(__file__).resolve().parent / "outputs"

# Fusion weights from Prototype 3 config
try:
    import importlib.util as _ilu
    _s = _ilu.spec_from_file_location("p3_config", P3_ROOT / "backend" / "config.py")
    _m = _ilu.module_from_spec(_s)
    _s.loader.exec_module(_m)
    WEIGHT_A, WEIGHT_B = _m.WEIGHT_A, _m.WEIGHT_B
except Exception as _e:
    print(f"Warning: could not read config.py ({_e}). Falling back to 0.75/0.25.")
    WEIGHT_A, WEIGHT_B = 0.75, 0.25


# Figure style
STYLE = {
    "tower_a": {"color": "#1f77b4", "label": "Tower A (URL Lexical, LR)"},
    "tower_b": {"color": "#ff7f0e", "label": "Tower B (Metadata, HGB)"},
    "fused": {"color": "#2ca02c",
              "label": f"Fused ({WEIGHT_A}\u00b7pA + {WEIGHT_B}\u00b7pB)"},
}
DPI = 150
plt.rcParams.update({
    "font.family": "Arial", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
})


# Data loading
# split_csv is a parameter so both test and val splits can use the same functions

def load_tower_a(split_csv: Path, feature_cols: list) -> tuple:
    """
    Joins features_handcrafted_AE.csv against a split CSV on url_norm.
    Mirrors what train_final_lr.py does during training.

    features_handcrafted_AE.csv also contains a 'label' column, so the merge produces label_x (from the split) and
    label_y (from the features CSV). Both are validated to be identical then collapsed to 'label'.
    """
    split = pd.read_csv(split_csv)
    feats = pd.read_csv(FEATURES_AE_CSV)
    merged = split[["url_norm", "label"]].merge(feats, on="url_norm", how="inner")

    # Resolve label duplication from the merge
    if "label_x" in merged.columns and "label_y" in merged.columns:
        if not (merged["label_x"] == merged["label_y"]).all():
            raise RuntimeError(
                "label_x and label_y disagree after merge - data integrity problem."
            )
        merged = merged.rename(columns={"label_x": "label"}).drop(columns=["label_y"])

    missing = [c for c in feature_cols if c not in merged.columns]
    if missing:
        raise RuntimeError(f"Missing Tower A columns after join: {missing}")
    return (
        merged[feature_cols].to_numpy(dtype=np.float32),
        merged["label"].to_numpy(dtype=np.int32),
        merged["url_norm"].tolist(),
    )


def load_tower_b(split_csv: Path, feature_cols: list) -> tuple:
    """Loads Tower B features from a split CSV (metadata already included)."""
    df = pd.read_csv(split_csv)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing Tower B columns: {missing}")
    return (
        df[feature_cols].to_numpy(dtype=np.float64),
        df["label"].to_numpy(dtype=np.int32),
        df["url_norm"].tolist(),
    )


def align_splits(
    X_a: np.ndarray, y_a: np.ndarray, urls_a: list,
    X_b: np.ndarray, y_b: np.ndarray, urls_b: list,
) -> tuple:
    """
    Aligns two splits to their shared URL intersection.
    Returns (X_a_aligned, X_b_aligned, y_aligned, shared_urls).
    Both splits share the same eTLD+1 mapping so the intersection should be close to the full split size.
    Any unexpected drop is immediately visible in the printed count.
    """
    shared = sorted(set(urls_a) & set(urls_b))
    idx_a = {u: i for i, u in enumerate(urls_a)}
    idx_b = {u: i for i, u in enumerate(urls_b)}
    ia = [idx_a[u] for u in shared]
    ib = [idx_b[u] for u in shared]
    return X_a[ia], X_b[ib], y_a[ia], shared


# Consistency verification

def verify_consistency(
    label: str,
    y: np.ndarray,
    p: np.ndarray,
    stored_prauc: float,
    stored_rocauc: float,
    tolerance: float = 1e-3,
) -> None:
    """
    Verifies that re-scoring a model reproduces the aggregate metrics from the stored prototype report.
    Exits rather than producing inconsistent figures.
    """
    got_pr = float(average_precision_score(y, p))
    got_roc = float(roc_auc_score(y, p))
    diff_pr = abs(got_pr - stored_prauc)
    diff_roc = abs(got_roc - stored_rocauc)
    print(f" {label} PR-AUC: stored={stored_prauc:.6f} recomputed={got_pr:.6f} diff={diff_pr:.2e}")
    print(f" {label} ROC-AUC: stored={stored_rocauc:.6f} recomputed={got_roc:.6f} diff={diff_roc:.2e}")
    if diff_pr > tolerance or diff_roc > tolerance:
        raise RuntimeError(
            f"{label} consistency FAILED - recomputed metrics do not match the stored "
            f"prototype report (PR diff={diff_pr:.4f}, ROC diff={diff_roc:.4f}). "
            f"Ensure model artifacts match those used during prototype evaluation."
        )
    print(f" {label} consistency: PASS")


# Threshold selection (fused only - A and B thresholds are read from reports)
def pick_threshold_best_f1(y_val: np.ndarray, p_val: np.ndarray) -> float:
    best_t, best_f1_val = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 300):
        f = f1_score(y_val, (p_val >= t).astype(int), zero_division=0)
        if f > best_f1_val:
            best_f1_val, best_t = f, float(t)
    return best_t


def pick_threshold_fpr_cap(y_val, p_val, fpr_cap=0.02):
    best_t, best_recall = 0.99, 0.0
    for t in np.linspace(0.01, 0.99, 500):
        preds = (p_val >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if fpr <= fpr_cap and rec > best_recall:
            best_recall = rec
            best_t = float(t)
    return best_t


# Metrics at a threshold

def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    preds = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_val = (2 * prec * recall) / (prec + recall) if (prec + recall) > 0 else 0.0
    return {
        "threshold": round(float(threshold), 6),
        "pr_auc": round(float(average_precision_score(y, p)), 6),
        "roc_auc": round(float(roc_auc_score(y, p)), 6),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1_val), 6),
        "fpr": round(fp / (tn + fp) if (tn + fp) > 0 else 0.0, 6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# FIGURES

def _save(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved: {out_path.name}")


def fig_curves(y, p_a, p_b, p_fused, rep_a, rep_b, mode: str, out_path: Path) -> None:
    """
    PR curve (mode='pr') or ROC curve (mode='roc').
    Tower A and B AUC values are read from stored prototype reports so figure labels match the dissertation metrics
    tables exactly.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    if mode == "pr":
        aucs       = (rep_a["test_metrics"]["best_f1"]["pr_auc"],
                      rep_b["metrics"]["test_pr_auc"],
                      float(average_precision_score(y, p_fused)))
        curve_data = [(precision_recall_curve(y, p)[1],  # x = recall
                       precision_recall_curve(y, p)[0])  # y = precision
                      for p in (p_a, p_b, p_fused)]
        ax.axhline(y.mean(), color="grey", linestyle="--", linewidth=1,
                   label=f"Random baseline ({y.mean():.4f})")
        ax.set(xlabel="Recall", ylabel="Precision",
               title="Precision-Recall Curves - Tower A vs Tower B vs Fused",
               xlim=[0, 1], ylim=[0, 1.05])
        legend_loc = "upper right"
        auc_label = "PR-AUC"
    else:
        aucs = (rep_a["test_metrics"]["best_f1"]["roc_auc"],
                rep_b["metrics"]["test_roc_auc"],
                float(roc_auc_score(y, p_fused)))
        curve_data = [(roc_curve(y, p)[0], roc_curve(y, p)[1])
                      for p in (p_a, p_b, p_fused)]
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1,
                label="Random baseline (0.5000)")
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
               title="ROC Curves - Tower A vs Tower B vs Fused",
               xlim=[0, 1], ylim=[0, 1.05])
        legend_loc = "lower right"
        auc_label  = "ROC-AUC"

    # Plot all three curves - legend() is called after this loop so all labels are captured correctly.
    for (x, y_vals), key, auc in zip(curve_data, STYLE, aucs):
        ax.plot(x, y_vals, color=STYLE[key]["color"], linewidth=2,
                label=f"{STYLE[key]['label']} ({auc_label} = {auc:.4f})")

    ax.legend(loc=legend_loc, framealpha=0.9)
    ax.grid(alpha=0.3)
    _save(fig, out_path)



def fig_score_distributions(y, p_a, p_fused, out_path: Path) -> None:
    """
    Score distributions for Tower A and the fused system on the internal test set. On this in-distribution set, benign
    and phishing scores are well-separated for Tower A. The fused panel shows how Tower B's near- random scores
    (mean ~0.22 for both classes) pull the fused distributions closer together, reducing the separation seen in Tower A alone.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, p, title in [
        (axes[0], p_a, "Tower A (p_A)"),
        (axes[1], p_fused, f"Fused ({WEIGHT_A}\u00b7pA + {WEIGHT_B}\u00b7pB)"),
    ]:
        ax.hist(p[y == 0], bins=50, alpha=0.6, color="#1f77b4",
                label=f"Benign (n={int((y==0).sum()):,})", density=True)
        ax.hist(p[y == 1], bins=50, alpha=0.6, color="#d62728",
                label=f"Phishing (n={int((y==1).sum()):,})", density=True)
        ax.set(xlabel="Phishing Score", ylabel="Density", title=title)
        ax.legend(framealpha=0.9)
        ax.grid(alpha=0.3)
    fig.suptitle("Score Distributions - eTLD+1 Test Set (9,916 URLs)",
                 fontsize=13, fontweight="bold")
    _save(fig, out_path)


def fig_disagreement_scatter(y, p_a, p_b, out_path: Path) -> None:
    """
    The bottom-right quadrant (high p_A, low p_B) is the correction zone where Tower B attenuates Tower A false
    positives on benign domains that have clean infrastructure signals.
    """
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y), size=min(2000, len(y)), replace=False)
    y_s, pa_s, pb_s = y[idx], p_a[idx], p_b[idx]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pa_s[y_s == 0], pb_s[y_s == 0], c="#1f77b4",
               alpha=0.4, s=12, label="Benign")
    ax.scatter(pa_s[y_s == 1], pb_s[y_s == 1], c="#d62728",
               alpha=0.4, s=12, label="Phishing")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)

    disagree = int(np.sum(np.abs(p_a - p_b) > 0.3))
    correction = int(np.sum((p_a > 0.7) & (p_b < 0.3)))
    ax.text(0.62, 0.04, f"Correction zone\n(pA>0.7, pB<0.3): {correction}",
            transform=ax.transAxes, fontsize=8, color="darkorange")
    ax.set(xlabel="Tower A score (p_A)", ylabel="Tower B score (p_B)",
           title=(f"Tower Agreement - 2,000-URL subsample\n"
                  f"Disagreements |pA\u2212pB|>0.3: {disagree}"),
           xlim=[-0.02, 1.02], ylim=[-0.02, 1.02])
    ax.legend(loc="center right", framealpha=0.9)
    ax.grid(alpha=0.2)
    _save(fig, out_path)


def fig_tower_a_features(model_a, feature_cols: list, X_test: np.ndarray,
                          out_path: Path) -> None:
    coef = model_a.coef_[0]
    attrs = coef * np.abs(X_test).mean(axis=0)
    order = np.argsort(np.abs(attrs))[::-1][:15]
    names = [feature_cols[i].replace("_", " ") for i in order]
    vals = attrs[order]

    # url_len dominates the scale. Clip at 150% of the second-largest value so that remaining features are visible, and
    # annotate the clipped bar.
    second_max = sorted(np.abs(vals), reverse=True)[1] if len(vals) > 1 else np.abs(vals).max()
    x_clip = second_max * 1.5
    clipped = {i: v for i, v in enumerate(vals) if abs(v) > x_clip}
    vals_plot = np.clip(vals, -x_clip, x_clip)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names)), vals_plot,
            color=["#1f77b4" if v > 0 else "#2ca02c" for v in vals],
            edgecolor="white", height=0.7)

    # Annotate any clipped bars with their actual value
    for i, actual in clipped.items():
        xpos = x_clip if actual > 0 else -x_clip
        ax.text(xpos * 0.98, i, f" {actual:.0f}",
                ha="right" if actual > 0 else "left",
                va="center", fontsize=8, color="white", fontweight="bold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Mean contribution (coef × |feature value|)",
           title="Tower A - Top 15 URL Features by Mean Attribution\n"
                 "(Blue = phishing signal, Green = benign signal)")
    if clipped:
        ax.set_xlim(-x_clip * 1.08, x_clip * 1.08)
        ax.annotate("† bar clipped - actual value shown inside",
                    xy=(0.99, 0.01), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=7, color="grey")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[
        mpatches.Patch(color="#1f77b4", label="Phishing signal"),
        mpatches.Patch(color="#2ca02c", label="Benign signal"),
    ], loc="lower right")
    _save(fig, out_path)

def fig_tower_b_importance(out_path: Path) -> None:
    """Reads directly from the committed feature_importances_B.json artefact."""
    imp_map = json.loads(IMP_B_PATH.read_text(encoding="utf-8"))["importances"]
    items = sorted(imp_map.items(), key=lambda x: abs(x[1]), reverse=True)
    names = [k.replace("_", " ") for k, v in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(names)), vals,
            color=["#ff7f0e" if v > 0 else "#9467bd" for v in vals],
            edgecolor="white", height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Permutation Importance (mean PR-AUC drop, n_repeats=10)",
           title="Tower B - Feature Importances\n"
                 "(feature_importances_B.json, scoring=average_precision)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[
        mpatches.Patch(color="#ff7f0e", label="Positive importance"),
        mpatches.Patch(color="#9467bd", label="Negative (noisy)"),
    ], loc="lower right")
    _save(fig, out_path)


def fig_confusion_matrices(y, p_a, p_b, p_fused,
                            t_a, t_b, t_fused, out_path: Path) -> None:
    configs = [
        ("Tower A\n(stored best-F1)", p_a, t_a),
        ("Tower B\n(stored best-F1)", p_b, t_b),
        ("Fused System\n(computed best-F1)", p_fused, t_fused),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, p, t) in zip(axes, configs):
        preds = (p >= t).astype(int)
        cm = confusion_matrix(y, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        ax.imshow(cm_norm, interpolation="nearest",
                  cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="True")
        ax.set_xticklabels(["Benign", "Phishing"])
        ax.set_yticklabels(["Benign", "Phishing"])
        prec_v = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_v = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr_v = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        ax.set_title(f"{title}\nt={t:.4f} Prec={prec_v:.3f} "
                     f"Rec={rec_v:.3f} FPR={fpr_v:.3f}", fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}\n({cm_norm[i,j]:.1%})",
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.suptitle("Confusion Matrices - eTLD+1 Test Set (9,916 URLs)",
                 fontsize=13, fontweight="bold", y=1.02)
    _save(fig, out_path)


# Text report

def write_report(rep_a, rep_b, m_fused, m_fused_fpr2,
                 y, p_a, p_b, p_fused, out_path: Path) -> None:

    pr_a = rep_a["test_metrics"]["best_f1"]["pr_auc"]
    pr_b = rep_b["metrics"]["test_pr_auc"]
    pr_fused = float(average_precision_score(y, p_fused))
    roc_a = rep_a["test_metrics"]["best_f1"]["roc_auc"]
    roc_b = rep_b["metrics"]["test_roc_auc"]
    roc_fused = float(roc_auc_score(y, p_fused))

    sa = rep_a["test_metrics"]["best_f1"]
    sa2 = rep_a["test_metrics"].get("fpr_cap_0.02", {})
    sb = rep_b["thresholds"]["best_f1"]

    mask = (p_a > 0.7) & (p_b < 0.3)
    benign_corr = int((y[mask] == 0).sum()) if mask.sum() > 0 else 0
    n = len(y)

    L = []
    L.append("=" * 70)
    L.append("FUSED SYSTEM EVALUATION")
    L.append(f"Test set: {n:,} URLs (eTLD+1) |"
             f"Fusion: {WEIGHT_A}*pA + {WEIGHT_B}*pB (from config.py)")
    L.append("=" * 70)

    # AUC table - Tower A/B values from stored reports, Fused computed here
    L.append(f"\n{'System':<30} {'PR-AUC':>8} {'ROC-AUC':>9} Source")
    L.append("-" * 62)
    L.append(f"{'Tower A (LR, 35 URL features)':<30} {pr_a:>8.4f} {roc_a:>9.4f} final_model_report.json")
    L.append(f"{'Tower B (HGB, 12 meta features)':<30} {pr_b:>8.4f} {roc_b:>9.4f} 04_model_report.json")
    L.append(f"{'Fused System':<30} {pr_fused:>8.4f} {roc_fused:>9.4f} computed (this script)")
    L.append(f"\nFusion gain over Tower A: "
             f"PR-AUC {pr_fused-pr_a:+.4f} ROC-AUC {roc_fused-roc_a:+.4f}")

    # Operating point table
    L.append(f"\n{'System / Threshold':<34} {'Prec':>7} {'Rec':>7}"
             f"{'F1':>7} {'FPR':>7} {'TP':>6} {'FP':>5} {'FN':>5}")
    L.append("-" * 80)
    L.append(f"Tower A t={sa['threshold']:.4f} best-F1 (stored)"
             f"{sa['precision']:>7.4f} {sa['recall']:>7.4f} {sa['f1']:>7.4f}"
             f"{sa['fpr']:>7.4f} {sa['tp']:>6} {sa['fp']:>5} {sa['fn']:>5}")
    if sa2:
        L.append(f"Tower A t={sa2.get('threshold',0):.4f} FPR<=2% (stored)"
                 f"{sa2.get('precision',0):>7.4f} {sa2.get('recall',0):>7.4f}"
                 f"{sa2.get('f1',0):>7.4f} {sa2.get('fpr',0):>7.4f}")
    L.append(f"Tower B t={sb['threshold']:.4f} best-F1 (stored)"
             f"{sb['precision']:>7.4f} {sb['recall']:>7.4f} {sb['f1']:>7.4f}"
             f"{sb['fpr']:>7.4f} {sb['tp']:>6} {sb['fp']:>5} {sb['fn']:>5}")
    L.append(f"Fused t={m_fused['threshold']:.4f} best-F1 (computed)"
             f"{m_fused['precision']:>7.4f} {m_fused['recall']:>7.4f}"
             f"{m_fused['f1']:>7.4f} {m_fused['fpr']:>7.4f}"
             f"{m_fused['tp']:>6} {m_fused['fp']:>5} {m_fused['fn']:>5}")
    L.append(f"Fused t={m_fused_fpr2['threshold']:.4f} FPR<=2% (computed)"
             f"{m_fused_fpr2['precision']:>7.4f} {m_fused_fpr2['recall']:>7.4f} "
             f"{m_fused_fpr2['f1']:>7.4f} {m_fused_fpr2['fpr']:>7.4f}")

    # Disagreement analysis
    disagree = int(np.sum(np.abs(p_a - p_b) > 0.3))
    correction = int(np.sum((p_a > 0.7) & (p_b < 0.3)))
    corr = float(np.corrcoef(p_a, p_b)[0, 1])
    L.append("\n--- Tower Disagreement ---")
    L.append(f"|pA - pB| > 0.3: {disagree:,} ({disagree/n*100:.1f}%)")
    L.append(f"Correction zone (pA>0.7, pB<0.3): {correction:,} - "
             f"{benign_corr} true benign (Tower B attenuates Tower A FP)")
    L.append(f"Pearson r(pA, pB): {corr:.4f}"
             f"(low = complementary signals)")

    # Score statistics
    L.append("\n--- Score Statistics ---")
    for lbl, p in [("Tower A", p_a), ("Tower B", p_b), ("Fused", p_fused)]:
        L.append(f"{lbl:<8} benign mean={p[y==0].mean():.4f}"
                 f"p95={np.percentile(p[y==0], 95):.4f}"
                 f"phishing mean={p[y==1].mean():.4f}"
                 f"p05={np.percentile(p[y==1], 5):.4f}")

    L.append("\n" + "=" * 70)
    report = "\n".join(L)
    print(report)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n  Written: {out_path}")



# Main

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("Prototype 3 - Fused System Evaluation")
    print(f"Weights: WEIGHT_A={WEIGHT_A}, WEIGHT_B={WEIGHT_B} (from config.py)")
    print("=" * 65)

    # Verify all required files exist before doing any work
    required = {
        "Tower A model": MODEL_A_PATH,
        "Tower A feature spec": SPEC_A_PATH,
        "Tower A features CSV": FEATURES_AE_CSV,
        "Tower A test split": TEST_A_CSV,
        "Tower A val split": VAL_A_CSV,
        "Tower A report": REPORT_A_PATH,
        "Tower B model": MODEL_B_PATH,
        "Tower B feature spec": SPEC_B_PATH,
        "Tower B importances": IMP_B_PATH,
        "Tower B test split": TEST_B_CSV,
        "Tower B val split": VAL_B_CSV,
        "Tower B report": REPORT_B_PATH,
    }
    missing = {k: v for k, v in required.items() if not v.exists()}
    if missing:
        print("\nERROR - required files not found:")
        for k, v in missing.items():
            print(f" {k}: {v}")
        sys.exit(1)

    # [1] Load stored prototype reports
    print("\n[1] Loading stored prototype reports...")
    rep_a = json.loads(REPORT_A_PATH.read_text(encoding="utf-8"))
    rep_b = json.loads(REPORT_B_PATH.read_text(encoding="utf-8"))
    print(f" Tower A stored PR-AUC: {rep_a['test_metrics']['best_f1']['pr_auc']}")
    print(f" Tower B stored PR-AUC: {rep_b['metrics']['test_pr_auc']}")

    # [2] Load feature specifications
    print("\n[2] Loading feature specifications...")
    cols_a = json.loads(SPEC_A_PATH.read_text(encoding="utf-8"))["feature_columns"]
    cols_b = json.loads(SPEC_B_PATH.read_text(encoding="utf-8"))["features"]
    print(f" Tower A: {len(cols_a)} features | Tower B: {len(cols_b)} features")

    # [3] Load models from prototype artefacts
    print("\n[3] Loading models...")
    model_a = joblib.load(MODEL_A_PATH)
    model_b = joblib.load(MODEL_B_PATH)

    # [4] Load and align test splits
    print("\n[4] Loading test data...")
    X_a, y_a, urls_a = load_tower_a(TEST_A_CSV, cols_a)
    X_b, y_b, urls_b = load_tower_b(TEST_B_CSV, cols_b)
    X_a, X_b, y, shared = align_splits(X_a, y_a, urls_a, X_b, y_b, urls_b)
    print(f"Shared test URLs: {len(y):,}"
          f"(benign={int((y==0).sum()):,}, phishing={int((y==1).sum()):,})")

    # [5] Score both towers on the test set
    print("\n[5] Scoring test set...")
    p_a = model_a.predict_proba(X_a)[:, 1].astype(np.float64)
    p_b = model_b.predict_proba(X_b)[:, 1].astype(np.float64)

    # [6] Verify recomputed metrics match stored prototype reports
    print("\n[6] Consistency verification...")
    verify_consistency("Tower A", y, p_a,
                       rep_a["test_metrics"]["best_f1"]["pr_auc"],
                       rep_a["test_metrics"]["best_f1"]["roc_auc"])
    verify_consistency("Tower B", y, p_b,
                       rep_b["metrics"]["test_pr_auc"],
                       rep_b["metrics"]["test_roc_auc"])

    # [7] Compute fused scores using weights from config.py
    print(f"\n[7] Computing fused scores...")
    p_fused = WEIGHT_A * p_a + WEIGHT_B * p_b
    print(f" Fused PR-AUC: {average_precision_score(y, p_fused):.4f}")

    # [8] Select fused thresholds on the validation set.
    #   Tower A and B thresholds come from stored reports - only the fused threshold needs computing since no prior
    #   fused evaluation exists.
    print("\n[8] Selecting fused thresholds on validation set...")
    X_av, y_av, urls_av = load_tower_a(VAL_A_CSV, cols_a)
    X_bv, y_bv, urls_bv = load_tower_b(VAL_B_CSV, cols_b)
    X_av, X_bv, y_v, _ = align_splits(X_av, y_av, urls_av, X_bv, y_bv, urls_bv)
    p_fus_v = (WEIGHT_A * model_a.predict_proba(X_av)[:, 1].astype(np.float64) +
               WEIGHT_B * model_b.predict_proba(X_bv)[:, 1].astype(np.float64))

    t_fused_f1 = pick_threshold_best_f1(y_v, p_fus_v)
    t_fused_fpr2 = pick_threshold_fpr_cap(y_v, p_fus_v, fpr_cap=0.02)
    print(f" Fused best-F1 threshold: {t_fused_f1:.4f}")
    print(f" Fused FPR<=2% threshold: {t_fused_fpr2:.4f}")

    # [9] Compute fused operating-point metrics on the test set
    m_fused = metrics_at(y, p_fused, t_fused_f1)
    m_fused_fpr2 = metrics_at(y, p_fused, t_fused_fpr2)

    # Stored thresholds for Tower A and B (used in confusion matrix figure)
    t_a = rep_a["thresholds"]["best_f1"]
    t_b = rep_b["thresholds"]["best_f1"]["threshold"]

    # Save per-URL predictions
    pd.DataFrame({
        "url_norm": shared, "label": y,
        "p_a": p_a.round(6), "p_b": p_b.round(6), "p_fused": p_fused.round(6),
    }).to_csv(OUT_DIR / "predictions_test.csv", index=False)
    print(f"\n Predictions saved: {len(shared):,} rows")

    # [10] Generate dissertation figures
    print("\n[10] Generating figures...")
    fig_curves(y, p_a, p_b, p_fused, rep_a, rep_b, "pr",
               OUT_DIR / "fig1_pr_curves.png")
    fig_curves(y, p_a, p_b, p_fused, rep_a, rep_b, "roc",
               OUT_DIR / "fig2_roc_curves.png")
    fig_score_distributions(y, p_a, p_fused,
                            OUT_DIR / "fig3_score_distributions.png")
    fig_disagreement_scatter(y, p_a, p_b,
                             OUT_DIR / "fig4_disagreement_scatter.png")
    fig_tower_a_features(model_a, cols_a, X_a,
                         OUT_DIR / "fig5_tower_a_features.png")
    fig_tower_b_importance(OUT_DIR / "fig6_tower_b_importance.png")
    fig_confusion_matrices(y, p_a, p_b, p_fused, t_a, t_b, t_fused_f1,
                           OUT_DIR / "fig7_confusion_matrices.png")

    # [11] Write evaluation report
    print("\n[11] Writing evaluation report...")
    write_report(rep_a, rep_b, m_fused, m_fused_fpr2,
                 y, p_a, p_b, p_fused,
                 OUT_DIR / "evaluation_report.txt")

    print("\n" + "=" * 65)
    print(f"Evaluation complete. Outputs: {OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()