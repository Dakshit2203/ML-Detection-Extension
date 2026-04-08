"""
Tower A - Phase 6: Compare Validation Scenarios and Generate Plots

Reads the prediction CSVs and evaluation JSONs from step 03 and produces a set of diagnostic plots for both scenarios.

Figures:
  score distribution histograms with threshold overlays
  empirical CDFs by class (shows separation and overlap)
  PR and ROC curves (with PR baseline at class prevalence)
  calibration plots (mean predicted vs empirical phishing rate per bin)
  recall vs FPR operating curve (shows achievable recall at each FPR target)

The calibration plot bins are annotated with sample counts to make it clear which calibration regions are 
well-supported by data.

The recall vs FPR curve uses benign-quantile thresholds - the same mechanism as the adaptive threshold in the 
backend - and marks the 2% FPR operating point.

Outputs: outputs/<latest_run>/plots/ and /reports/comparison_*.csv/.txt

Run AFTER 03_evaluate.py.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

BASE = Path(__file__).resolve().parent
OUT_ROOT = BASE / "outputs"

plt.rcParams.update({
    "savefig.dpi": 300,
    "figure.dpi": 100,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

def latest_run_dir() -> Path:
    runs = sorted(OUT_ROOT.glob("external_validation__*"), key=lambda p: p.name)
    if not runs:
        raise FileNotFoundError(
            f"No external_validation__* found under {OUT_ROOT}.\n"
            f"Run scripts 01-03 first."
        )
    return runs[-1]

def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _threshold_lines(rep: Dict[str, Any]) -> Dict[str, float]:
    """Extract all threshold values from an evaluation JSON for use as vertical lines."""
    lines: Dict[str, float] = {}
    for k, m in rep.get("stored_threshold_metrics", {}).items():
        lines[f"stored:{k}"] = float(m["threshold"])
    for k, m in rep.get("external_fpr_threshold_metrics", {}).items():
        lines[f"ext:{k}"] = float(m["threshold"])
    return lines

def _select_annotated(lines: Dict[str, float]) -> Dict[str, float]:
    """Limit annotations to the most informative thresholds to avoid label clutter."""
    keep = ["stored:best_f1", "stored:fpr_cap_0.02", "ext:fpr_0.010", "ext:fpr_0.020"]
    return {k: lines[k] for k in keep if k in lines}

def _draw_vlines(ax, all_lines: Dict[str,float],
                 annotated: Dict[str,float], label_y: float) -> None:
    for _, x in all_lines.items():
        ax.axvline(x, linestyle="--", linewidth=0.8, alpha=0.6)
    for name, x in annotated.items():
        ax.axvline(x, linestyle="--", linewidth=1.2)
        ax.text(x, label_y, name, rotation=90, va="top", ha="right", fontsize=7)

def plot_hist(df: pd.DataFrame, rep: Dict[str,Any], title: str, out: Path) -> None:
    y, p = df["label"].to_numpy(), df["p_phish"].to_numpy()
    fig, ax = plt.subplots()
    ax.hist(p[y==0], bins=60, alpha=0.7, label="benign")
    ax.hist(p[y==1], bins=60, alpha=0.7, label="phish")
    ax.set_xlabel("p_phish"); ax.set_ylabel("count")
    ax.set_title(title); ax.legend()
    all_lines = _threshold_lines(rep)
    _draw_vlines(ax, all_lines, _select_annotated(all_lines), ax.get_ylim()[1])
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

def plot_cdf(df: pd.DataFrame, rep: Dict[str,Any], title: str, out: Path) -> None:
    fig, ax = plt.subplots()
    for lab, name in [(0,"benign"),(1,"phish")]:
        x = np.sort(df.loc[df["label"]==lab, "p_phish"].to_numpy())
        if x.size:
            ax.plot(x, np.arange(1, x.size+1)/x.size, label=name)
    ax.set_xlabel("p_phish"); ax.set_ylabel("CDF")
    ax.set_title(title); ax.legend()
    all_lines = _threshold_lines(rep)
    _draw_vlines(ax, all_lines, _select_annotated(all_lines), ax.get_ylim()[1])
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

def plot_pr(df: pd.DataFrame, prefix: str, out: Path) -> float:
    y, p = df["label"].to_numpy(), df["p_phish"].to_numpy()
    ap = float(average_precision_score(y, p)) if len(np.unique(y))==2 else 0.5
    prec, rec, _ = precision_recall_curve(y, p)
    # PR baseline = class prevalence; a useful model must exceed this at all recall levels.
    prev = float((y==1).mean()) if y.size else 0.0
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label="PR curve")
    ax.axhline(prev, linestyle="--", linewidth=1, label=f"baseline (prev={prev:.3f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(f"{prefix} PR curve (AP={ap:.4f})"); ax.legend()
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return ap

def plot_roc(df: pd.DataFrame, prefix: str, out: Path) -> float:
    y, p = df["label"].to_numpy(), df["p_phish"].to_numpy()
    auc = float(roc_auc_score(y, p)) if len(np.unique(y))==2 else 0.5
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1, label="random baseline")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"{prefix} ROC curve (AUC={auc:.4f})"); ax.legend()
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return auc

def plot_calibration(df: pd.DataFrame, title: str, out: Path, n_bins: int = 15) -> None:
    """
    Plot mean predicted probability against empirical phishing rate per score bin.
    Bins with fewer than 50 samples are excluded to avoid misleading calibration estimates.
    Sample counts are annotated on each point to show which regions are well-supported.
    """
    y, p = df["label"].to_numpy(), df["p_phish"].to_numpy()
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.clip(np.digitize(p, bins)-1, 0, n_bins-1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() < 50:
            continue
        xs.append(float(p[mask].mean()))
        ys.append(float(y[mask].mean()))
        ns.append(int(mask.sum()))
    fig, ax = plt.subplots()
    ax.plot([0,1],[0,1], linestyle="--", label="ideal")
    if xs:
        ax.plot(xs, ys, marker="o", label="observed")
        for x, yv, n in zip(xs, ys, ns):
            ax.text(x, yv, str(n), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("mean predicted p_phish"); ax.set_ylabel("empirical phishing rate")
    ax.set_title(title + " (bin counts annotated)"); ax.legend()
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

def plot_fpr_recall_curve(df: pd.DataFrame, title: str, out: Path,
                           mark_fpr: float = 0.02) -> None:
    """
    Show recall as a function of FPR target, using benign-quantile threshold calibration.
    This directly illustrates the operating envelope of the adaptive threshold policy and highlights the 2% FPR
    deployment target.
    """
    y, p = df["label"].to_numpy(), df["p_phish"].to_numpy()
    benign, phish = p[y==0], p[y==1]
    if benign.size == 0 or phish.size == 0:
        return
    fprs = [0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.10]
    recalls: List[float] = [
        float((phish >= float(np.quantile(benign, 1.0-fpr))).mean())
        for fpr in fprs
    ]
    fig, ax = plt.subplots()
    ax.plot(fprs, recalls, marker="o")
    ax.set_xlabel("False Positive Rate (benign-quantile threshold)")
    ax.set_ylabel("Recall (TPR)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if mark_fpr in fprs:
        i = fprs.index(mark_fpr)
        ax.scatter([fprs[i]], [recalls[i]])
        ax.text(fprs[i], recalls[i],
                f" FPR={mark_fpr:.0%}\n recall={recalls[i]:.3f}",
                fontsize=8, va="bottom")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

def main() -> None:
    run_dir = latest_run_dir()
    rep_dir = run_dir / "reports"
    pred_dir = run_dir / "predictions"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir: {run_dir}")

    for path in (rep_dir/"evaluation_aggressive.json", rep_dir/"evaluation_neutral.json",
                 pred_dir/"predictions_aggressive.csv", pred_dir/"predictions_neutral.csv"):
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}\nRun 03_evaluate.py first.")

    A = load_json(rep_dir / "evaluation_aggressive.json")
    B = load_json(rep_dir / "evaluation_neutral.json")
    dfA = pd.read_csv(pred_dir / "predictions_aggressive.csv")
    dfB = pd.read_csv(pred_dir / "predictions_neutral.csv")

    plot_hist(dfA, A, "Aggressive paths: score distribution + thresholds",
              plots_dir/"aggressive_hist_with_thresholds.png")
    plot_hist(dfB, B, "Neutral paths: score distribution + thresholds",
              plots_dir/"neutral_hist_with_thresholds.png")

    plot_cdf(dfA, A, "Aggressive paths: CDF by class + thresholds",
             plots_dir/"aggressive_cdf.png")
    plot_cdf(dfB, B, "Neutral paths: CDF by class + thresholds",
             plots_dir/"neutral_cdf.png")

    ap_ag = plot_pr( dfA, "Aggressive", plots_dir/"aggressive_pr.png")
    auc_ag = plot_roc(dfA, "Aggressive", plots_dir/"aggressive_roc.png")
    ap_ne = plot_pr( dfB, "Neutral", plots_dir/"neutral_pr.png")
    auc_ne = plot_roc(dfB, "Neutral", plots_dir/"neutral_roc.png")

    plot_calibration(dfA, "Aggressive: calibration", plots_dir/"aggressive_calibration.png")
    plot_calibration(dfB, "Neutral: calibration", plots_dir/"neutral_calibration.png")

    plot_fpr_recall_curve(dfA, "Aggressive: recall vs FPR (quantile threshold, 2% marked)",
                          plots_dir/"aggressive_fpr_recall_curve.png", mark_fpr=0.02)
    plot_fpr_recall_curve(dfB, "Neutral: recall vs FPR (quantile threshold, 2% marked)",
                          plots_dir/"neutral_fpr_recall_curve.png", mark_fpr=0.02)

    rows = []
    for name, rep in [("aggressive", A), ("neutral", B)]:
        for k, m in rep.get("stored_threshold_metrics", {}).items():
            rows.append({"scenario": name, "threshold_type": f"stored:{k}",
                         "threshold": m["threshold"], "fpr": m["fpr"],
                         "recall": m["recall"], "precision": m["precision"], "f1": m["f1"]})
        for k, m in rep.get("external_fpr_threshold_metrics", {}).items():
            rows.append({"scenario": name, "threshold_type": f"ext:{k}",
                         "threshold": m["threshold"], "fpr": m["fpr"],
                         "recall": m["recall"], "precision": m["precision"], "f1": m["f1"]})
    pd.DataFrame(rows).to_csv(rep_dir/"comparison_metrics.csv", index=False)

    summary = [
        "=== External Validation Comparison Summary ===",
        f"Generated: {datetime.now().isoformat()}",
        f"Run dir: {run_dir}", "",
        f"{'Scenario':<15} {'PR-AUC':<10} {'ROC-AUC':<10}",
        f"{'aggressive':<15} {ap_ag:<10.4f} {auc_ag:<10.4f}",
        f"{'neutral':<15} {ap_ne:<10.4f} {auc_ne:<10.4f}",
    ]
    (rep_dir/"comparison_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    print(f"\nPlots: {plots_dir}")
    print(f"Reports: {rep_dir}")

if __name__ == "__main__":
    main()
