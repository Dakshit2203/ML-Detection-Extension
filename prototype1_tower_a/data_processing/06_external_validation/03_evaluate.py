"""
Tower A - Phase 6: External Evaluation

Applies the trained LR model to both external scenarios and evaluates it under two threshold strategies:
  - stored training thresholds (best_f1, fpr_cap_0.02, recall_target_0.95)
  - externally calibrated thresholds derived from the benign score distribution at a set of target FPR values (the quantile approach)

The stored training thresholds produce FPR=1.0 on external benign URLs. This is the key finding that motivates adaptive 
thresholding in the live backend. Root cause: training benign URLs (Kaggle) have rich path structure, so the model 
learns to associate a short/missing path with phishing. Tranco top-domain URLs are typically domain-only, so they score 
high under training-distribution thresholds.

The externally calibrated thresholds demonstrate that the model's discriminative ability (ROC-AUC, PR-AUC) is largely 
intact - only the threshold needs adjusting.

Inputs:  outputs/<latest_run>/datasets/ and /features/
         prototype1_tower_a/backend/artifacts/model_lr.joblib
         prototype1_tower_a/backend/artifacts/thresholds.json

Outputs: outputs/<latest_run>/predictions/ and /reports/

Run AFTER 02_extract_features.py.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import load as joblib_load
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(__file__).resolve().parent
OUT_ROOT = BASE / "outputs"
P1_ROOT = BASE.parents[1]

ARTIFACTS_DIR = P1_ROOT / "backend" / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_lr.joblib"
THRESHOLDS_PATH = ARTIFACTS_DIR / "thresholds.json"

SCENARIOS = ["aggressive", "neutral"]
FPR_TARGETS = [0.005, 0.010, 0.020, 0.050]

def latest_run_dir() -> Path:
    runs = sorted(OUT_ROOT.glob("external_validation__*"), key=lambda p: p.name)
    if not runs:
        raise FileNotFoundError(
            f"No external_validation__* found under {OUT_ROOT}.\n"
            f"Run scripts 01 and 02 first."
        )
    return runs[-1]

def confusion_counts(y: np.ndarray, yhat: np.ndarray) -> Tuple[int,int,int,int]:
    tp = int(((y==1)&(yhat==1)).sum()); fp = int(((y==0)&(yhat==1)).sum())
    tn = int(((y==0)&(yhat==0)).sum()); fn = int(((y==1)&(yhat==0)).sum())
    return tp, fp, tn, fn

def metrics_at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> Dict[str,Any]:
    yhat = (p >= t).astype(np.int32)
    tp,fp,tn,fn = confusion_counts(y, yhat)
    prec = tp/(tp+fp) if (tp+fp) else 0.0
    rec = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    total= tp+tn+fp+fn
    acc = (tp+tn)/total if total else 0.0
    spec = tn/(tn+fp) if (tn+fp) else 0.0
    fpr = fp/(fp+tn) if (fp+tn) else 0.0
    fnr = fn/(fn+tp) if (fn+tp) else 0.0
    return {
        "threshold": float(t), "accuracy": float(acc),
        "balanced_accuracy": float(0.5*(rec+spec)),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "specificity": float(spec), "fpr": float(fpr), "fnr": float(fnr),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

def pick_threshold_for_fpr(y: np.ndarray, p: np.ndarray,
                            fpr_target: float) -> Optional[float]:
    """
    Set the threshold at the (1 - fpr_target) quantile of the benign score distribution. This is the same 
    quantile-based calibration used by the adaptive threshold in the backend - it does not require labelled data at 
    inference time, only the benign score distribution.
    """
    benign = p[y == 0]
    if benign.size == 0:
        return None
    return float(np.quantile(benign, min(max(1.0 - float(fpr_target), 0.0), 1.0)))

def describe_scores(p: np.ndarray) -> Dict[str,float]:
    if p.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    d = {"mean": float(np.mean(p)), "std": float(np.std(p)),
         "min": float(np.min(p)), "max": float(np.max(p))}
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        d[f"p{int(q*100):02d}"] = float(np.quantile(p, q))
    return d


def load_thresholds() -> Dict[str,float]:
    obj = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    return {
        k: float(obj[k])
        for k in ("best_f1", "fpr_cap_0.02", "recall_target_0.95")
        if k in obj and obj[k] is not None
    }

def main() -> None:
    for path in (MODEL_PATH, THRESHOLDS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing artefact: {path}\n"
                f"Run 05_final_model/train_final_lr.py first."
            )

    run_dir = latest_run_dir()
    pred_dir = run_dir / "predictions"
    rep_dir = run_dir / "reports"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir: {run_dir}")
    model = joblib_load(MODEL_PATH)
    model_feature_cols: Optional[List[str]] = (
        list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    )
    thresholds = load_thresholds()

    summary_lines = [
        f"External evaluation - {datetime.now().isoformat()}",
        f"Run dir: {run_dir}",
        f"Model: {MODEL_PATH}",
    ]

    for scenario in SCENARIOS:
        ds_path = run_dir / "datasets" / f"external_dataset_{scenario}.csv"
        feat_path = run_dir / "features" / f"features_{scenario}.csv"
        for p in (ds_path, feat_path):
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}\nRun scripts 01-02 first.")

        ds = pd.read_csv(ds_path)
        feats = pd.read_csv(feat_path)
        y = ds["label"].astype(int).to_numpy()

        if "label" in feats.columns:
            feats = feats.drop(columns=["label"])

        if model_feature_cols is not None:
            missing = [c for c in model_feature_cols if c not in feats.columns]
            if missing:
                raise RuntimeError(f"Features missing required model columns: {missing}")
            X = feats[model_feature_cols].to_numpy(dtype=np.float32)
            used_cols = model_feature_cols
        else:
            X = feats.to_numpy(dtype=np.float32)
            used_cols = list(feats.columns)

        p_scores = model.predict_proba(X)[:, 1]
        roc = float(roc_auc_score(y, p_scores)) if len(np.unique(y)) == 2 else 0.5
        pr = float(average_precision_score(y, p_scores)) if len(np.unique(y)) == 2 else 0.5

        stored = {k: metrics_at_threshold(y, p_scores, t) for k, t in thresholds.items()}

        ext = {}
        for fpr_t in FPR_TARGETS:
            t = pick_threshold_for_fpr(y, p_scores, fpr_t)
            if t is not None:
                ext[f"fpr_{fpr_t:.3f}"] = metrics_at_threshold(y, p_scores, t)

        benign_stats = describe_scores(p_scores[y == 0])
        phish_stats = describe_scores(p_scores[y == 1])

        pred_df = pd.DataFrame({"label": y, "p_phish": p_scores})
        if "source" in ds.columns:
            pred_df["source"] = ds["source"].astype(str).values
        pred_csv = pred_dir / f"predictions_{scenario}.csv"
        pred_df.to_csv(pred_csv, index=False)

        report = {
            "created_at": datetime.now().isoformat(),
            "scenario": scenario,
            "model_path": str(MODEL_PATH),
            "n_rows": int(len(ds)),
            "label_counts": {str(k): int(v) for k, v in ds["label"].value_counts().to_dict().items()},
            "roc_auc": roc,
            "pr_auc": pr,
            "model_feature_count": int(len(used_cols)),
            "benign_score_stats": benign_stats,
            "phish_score_stats": phish_stats,
            "stored_threshold_metrics": stored,
            "external_fpr_threshold_metrics": ext,
        }
        (rep_dir / f"evaluation_{scenario}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

        summary_lines += [
            f"=== {scenario} ===",
            f"Rows: {len(ds):,} | Labels: {report['label_counts']}",
            f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f}",
            f"Benign scores - mean={benign_stats['mean']:.4f} p95={benign_stats.get('p95', float('nan')):.4f}",
            f"Phish scores - mean={phish_stats['mean']:.4f} p95={phish_stats.get('p95', float('nan')):.4f}",
            "",
            "Stored training thresholds:",
        ]
        for k, m in stored.items():
            summary_lines.append(
                f" {k}: t={m['threshold']:.6f} FPR={m['fpr']:.4f} "
                f"recall={m['recall']:.4f} prec={m['precision']:.4f}"
            )
        summary_lines.append("Externally calibrated thresholds (quantile):")
        for k, m in ext.items():
            summary_lines.append(
                f" {k}: t={m['threshold']:.6f} FPR={m['fpr']:.4f} "
                f"recall={m['recall']:.4f} prec={m['precision']:.4f}"
            )
        summary_lines.append("")

    summary_path = rep_dir / "external_evaluation_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"Reports: {rep_dir}")

if __name__ == "__main__":
    main()