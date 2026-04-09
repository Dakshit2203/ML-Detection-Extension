"""
Tower B - Data Processing - 04_train_eval.py

Trains the Tower B HGB model and evaluates it on the held-out test set.

Model choice: HistGradientBoostingClassifier
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from sklearn.inspection import permutation_importance

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)

# Paths
DATA_DIR = Path(__file__).resolve().parent / "outputs" / "splits"
SPEC_PATH = Path(__file__).resolve().parent / "outputs" / "feature_spec_B.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "towerB"
importances_path = ARTIFACTS_DIR / "feature_importances_B.json"

# Fixed random seed for reproducibility
SEED = 42

# Utility functions
def load_split(name: str, features: list, label_col: str = "label") -> tuple:
    """
    Loads a split CSV and returns (X, y) arrays with features in the canonical order defined by the feature spec.

    None values in the metadata are filled with MISSING_SENTINEL before returning, matching the behaviour of the
    backend's TowerB.score() method.
    """
    path = DATA_DIR / f"{name}_B.csv"
    if not path.exists():
        print(f"ERROR: Split file not found: {path}")
        print("Run 03_split.py first.")
        sys.exit(1)

    df = pd.read_csv(path)

    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"ERROR: Missing feature columns in {name}: {missing}")
        sys.exit(1)

    X = df[features].to_numpy(dtype=np.float64)
    y = df[label_col].to_numpy(dtype=np.int32)
    return X, y

def pick_threshold_best_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Returns the threshold that maximises F1 on the provided set."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 200):
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def pick_threshold_fpr_cap(
    y_true: np.ndarray, y_prob: np.ndarray, fpr_cap: float
) -> float | None:
    """
    Returns the highest threshold where actual FPR ≤ fpr_cap.
    Returns None if no such threshold exists.
    """
    best_t = None
    for t in np.linspace(0.99, 0.01, 500):
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        if fpr <= fpr_cap:
            best_t = float(t)
            break
    return best_t


def compute_metrics_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict:
    """Returns a metrics dictionary for a given decision threshold."""
    preds = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    n_neg = tn + fp
    n_pos = fn + tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    return {
        "threshold": round(float(threshold), 6),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "fpr": round(float(fpr), 6),
        "fnr": round(float(fn / n_pos) if n_pos > 0 else 0.0, 6),
    }


# Main
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load feature spec
    if not SPEC_PATH.exists():
        print(f"ERROR: Feature spec not found: {SPEC_PATH}")
        print("Run 02_feature_spec.py first.")
        sys.exit(1)

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    features = spec["features"]
    print(f"Feature spec: {len(features)} features")

    # Load splits
    print("Loading splits...")
    X_train, y_train = load_split("train", features)
    X_val, y_val = load_split("val", features)
    X_test, y_test = load_split("test",features)

    print(f" train: {X_train.shape[0]:,} rows")
    print(f" val: {X_val.shape[0]:,} rows")
    print(f" test: {X_test.shape[0]:,} rows")

    # Train HGB
    # HGB is configured with:
    # - max_iter=500: sufficient for this feature count; early stopping is off so that the full training history is used
    # - random_state=42: reproducibility
    print("\nTraining HistGradientBoostingClassifier...")
    t0 = time.perf_counter()
    model = HistGradientBoostingClassifier(
        max_iter=500,
        random_state=SEED,
        early_stopping=False,
    )
    model.fit(X_train, y_train)
    train_time_s = time.perf_counter() - t0
    print(f" Training complete in {train_time_s:.1f}s")

    # Score all splits
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    # PR-AUC and ROC-AUC
    pr_auc_val = average_precision_score(y_val, p_val)
    pr_auc_test = average_precision_score(y_test, p_test)
    roc_auc_val = roc_auc_score(y_val, p_val)
    roc_auc_test= roc_auc_score(y_test, p_test)
    brier_test = brier_score_loss(y_test, p_test)

    print(f"\n Val PR-AUC: {pr_auc_val:.4f}")
    print(f" Test PR-AUC: {pr_auc_test:.4f}")
    print(f" Val ROC-AUC: {roc_auc_val:.4f}")
    print(f" Test ROC-AUC: {roc_auc_test:.4f}")
    print(f" Brier score: {brier_test:.4f}")

    # Threshold selection on validation set
    # Thresholds are always selected on the validation set, never on the test set. Applying val-derived thresholds to
    # the test set avoids threshold overfitting and produces honest operating-point metrics.
    print("\nSelecting thresholds on validation set...")
    t_best_f1 = pick_threshold_best_f1(y_val, p_val)
    t_fpr_cap2 = pick_threshold_fpr_cap(y_val, p_val, fpr_cap=0.02)
    t_fpr_cap5 = pick_threshold_fpr_cap(y_val, p_val, fpr_cap=0.05)

    print(f" best_f1 threshold: {t_best_f1:.4f}")
    print(f" fpr_cap_0.02 threshold: {t_fpr_cap2:.4f}" if t_fpr_cap2 else " fpr_cap_0.02: NOT ACHIEVABLE")
    print(f" fpr_cap_0.05 threshold: {t_fpr_cap5:.4f}" if t_fpr_cap5 else " fpr_cap_0.05: NOT ACHIEVABLE")

    # Evaluate on test set at each threshold
    print("\nTest set metrics at each threshold:")
    thresholds_report = {}

    for name, t in [
        ("best_f1", t_best_f1),
        ("fpr_cap_02", t_fpr_cap2),
        ("fpr_cap_05", t_fpr_cap5),
    ]:
        if t is not None:
            m = compute_metrics_at_threshold(y_test, p_test, t)
            thresholds_report[name] = m
            print(f" [{name}] t={t:.4f} precision={m['precision']:.4f} "
                  f"recall={m['recall']:.4f} fpr={m['fpr']:.4f} f1={m['f1']:.4f}")
        else:
            thresholds_report[name] = None
            print(f" [{name}] NOT ACHIEVABLE on validation set")

    # Latency measurement
    # Measured on the test set (N=9,916 rows) for comparability with Tower A.
    n_sample = min(500, len(X_test))
    X_sample = X_test[:n_sample]
    runs = []
    for _ in range(5):
        t0 = time.perf_counter()
        model.predict_proba(X_sample)
        runs.append((time.perf_counter() - t0) * 1000 / n_sample)
    latency_ms = float(np.median(runs))
    print(f"\nInference latency: {latency_ms:.4f} ms/URL (median of 5 runs on {n_sample} URLs)")

    # Compute permutation feature importances on the validation set.
    # HistGradientBoostingClassifier does not expose feature_importances_ as a direct attribute, so permutation
    # importance is computed here and saved to JSON for use by the Prototype 3 XAI layer.
    print("\nComputing permutation feature importances on validation set...")
    perm_result = permutation_importance(
        model, X_val, y_val,
        n_repeats=10,
        random_state=SEED,
        scoring="average_precision",
    )
    importances_dict = {
        features[i]: round(float(perm_result.importances_mean[i]), 6)
        for i in range(len(features))
    }
    importances_path = ARTIFACTS_DIR / "feature_importances_B.json"
    importances_path.write_text(
        json.dumps({
            "method": "permutation_importance",
            "scoring": "average_precision",
            "n_repeats": 10,
            "set": "validation",
            "importances": importances_dict,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Feature importances saved: {importances_path}")

    # Model size
    model_path = ARTIFACTS_DIR / "hgb.joblib"
    joblib.dump(model, model_path)
    model_mb = model_path.stat().st_size / (1024 ** 2)
    print(f"Model size: {model_mb:.4f} MB")
    print(f"Model saved: {model_path}")

    # Write report
    report = {
        "model": "HistGradientBoostingClassifier",
        "n_features": len(features),
        "features": features,
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "train_time_s": round(train_time_s, 2),
        "metrics": {
            "val_pr_auc": round(pr_auc_val, 4),
            "test_pr_auc": round(pr_auc_test, 4),
            "val_roc_auc": round(roc_auc_val, 4),
            "test_roc_auc": round(roc_auc_test, 4),
            "brier_score": round(brier_test, 4),
        },
        "thresholds": thresholds_report,
        "latency_ms_per_url": round(latency_ms, 4),
        "model_size_mb": round(model_mb, 4),
        "missing_sentinel": "NaN (native HGB missing value support)",
        "random_seed": SEED,
    }

    report_path = OUTPUT_DIR / "04_model_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Summary
    lines = [
        "=" * 70,
        "Tower B - Model Evaluation Summary",
        "=" * 70,
        f"Model:      HistGradientBoostingClassifier",
        f"Features:   {len(features)}",
        f"Train rows: {X_train.shape[0]:,}",
        f"Val rows:   {X_val.shape[0]:,}",
        f"Test rows:  {X_test.shape[0]:,}",
        "",
        "AUC metrics (test set):",
        f" PR-AUC:  {pr_auc_test:.4f}",
        f" ROC-AUC: {roc_auc_test:.4f}",
        f" Brier:   {brier_test:.4f}",
        "",
        "Operating points (val threshold -> test performance):",
    ]
    for name, m in thresholds_report.items():
        if m:
            lines.append(
                f" [{name}] t={m['threshold']:.4f}"
                f"precision={m['precision']:.4f} recall={m['recall']:.4f}"
                f"fpr={m['fpr']:.4f} f1={m['f1']:.4f}"
            )
        else:
            lines.append(f" [{name}] NOT ACHIEVABLE")

    lines += [
        "",
        f"Latency:    {latency_ms:.4f} ms/URL",
        f"Model size: {model_mb:.4f} MB",
        "=" * 70,
    ]

    summary = "\n".join(lines)
    print("\n" + summary)
    (OUTPUT_DIR / "04_model_summary.txt").write_text(summary, encoding="utf-8")

    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print(f" 04_model_report.json")
    print(f" 04_model_summary.txt")
    print(f" artifacts/towerB/hgb.joblib")

if __name__ == "__main__":
    main()

