"""
Tower A - Phase 5: Train and Export the Final Logistic Regression Model

Trains the deployment LR model on the eTLD+1 train split, applies three threshold policies on the validation set, and 
exports all artefacts to prototype1_tower_a/backend/artifacts/ for use by the live inference backend.

LR was selected based on Phase 4 evaluation results.

Inputs:
  - 02_feature_engineering/outputs/features_handcrafted_AE.csv
  - 03_data_splitting/outputs/etld1/train.csv, val.csv, test.csv
  - backend/artifacts/feature_spec.json (defines column order for inference)

Outputs written to prototype1_tower_a/backend/artifacts/:
  model_lr.joblib       serialised trained model
  thresholds.json       three threshold policies selected on val
  feature_spec.json     column order copied verbatim from input spec
  meta.json             training provenance and final test metrics
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

DATA_PROC = Path(__file__).resolve().parents[1]
P1_ROOT = DATA_PROC.parent

SPLIT_DIR = DATA_PROC / "03_data_splitting" / "outputs" / "etld1"
TRAIN_CSV = SPLIT_DIR / "train.csv"
VAL_CSV = SPLIT_DIR / "val.csv"
TEST_CSV = SPLIT_DIR / "test.csv"

AE_FEATURES_CSV = DATA_PROC / "02_feature_engineering" / "outputs" / "features_handcrafted_AE.csv"
ART_DIR = P1_ROOT / "backend" / "artifacts"
FEATURE_SPEC_JSON= ART_DIR / "feature_spec.json"
OUT_DIR = DATA_PROC / "05_final_model" / "outputs"

SEED = 42
FPR_CAP = 0.02
RECALL_TARGET = 0.95

LR_PARAMS = dict(solver="lbfgs", max_iter=4000, random_state=SEED)

@dataclass
class Metrics:
    threshold: float; roc_auc: float; pr_auc: float
    precision: float; recall: float; f1: float
    accuracy: float; balanced_accuracy: float; specificity: float
    fpr: float; fnr: float
    tp: int; fp: int; tn: int; fn: int


def _confusion(y: np.ndarray, yhat: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    tn = int(((y == 0) & (yhat == 0)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    return tp, fp, tn, fn

def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Metrics:
    yhat = (p >= threshold).astype(np.int32)
    tp, fp, tn, fn = _confusion(y, yhat)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    roc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else 0.5
    pr = average_precision_score(y, p) if len(np.unique(y)) == 2 else 0.5
    return Metrics(
        threshold=float(threshold), roc_auc=float(roc), pr_auc=float(pr),
        precision=float(prec), recall=float(rec), f1=float(f1),
        accuracy=float(acc), balanced_accuracy=float(0.5*(rec+spec)),
        specificity=float(spec), fpr=float(fpr), fnr=float(fnr),
        tp=tp, fp=fp, tn=tn, fn=fn,
    )

def pick_best_f1(y: np.ndarray, p: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(p):
        m = compute_metrics(y, p, float(t))
        if m.f1 > best_f1:
            best_f1, best_t = m.f1, float(t)
    return float(best_t)

def pick_recall_at_fpr(y: np.ndarray, p: np.ndarray, fpr_cap: float) -> Optional[float]:
    best_t, best_recall = None, -1.0
    for t in np.unique(p):
        m = compute_metrics(y, p, float(t))
        if m.fpr <= fpr_cap and m.recall > best_recall:
            best_recall, best_t = m.recall, float(t)
    return best_t

def pick_precision_at_recall(y: np.ndarray, p: np.ndarray,
                              recall_target: float) -> Optional[float]:
    best_t, best_prec = None, -1.0
    for t in np.unique(p):
        m = compute_metrics(y, p, float(t))
        if m.recall >= recall_target and m.precision > best_prec:
            best_prec, best_t = m.precision, float(t)
    return best_t


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("url_norm", "label"):
        if col not in df.columns:
            raise ValueError(f"{path} missing column: {col}")
    df["url_norm"] = df["url_norm"].astype(str)
    df["label"] = df["label"].astype(int)
    return df

def load_feature_spec(path: Path) -> list[str]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    cols = spec.get("feature_columns")
    if not isinstance(cols, list) or not cols:
        raise ValueError(f"feature_spec.json has no valid feature_columns list: {path}")
    return cols

def build_xy(features_csv: Path, split_df: pd.DataFrame,
             feature_cols: list[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Join split_df to the AE feature matrix on url_norm.
    Only the required feature columns are loaded to avoid a label column collision
    when the feature CSV also contains a label column.
    """
    feats = pd.read_csv(features_csv, usecols=["url_norm"] + feature_cols)
    merged = split_df[["url_norm", "label"]].merge(feats, on="url_norm", how="left")
    nan_rows = merged[feature_cols].isna().any(axis=1).sum()
    if nan_rows > 0:
        raise RuntimeError(
            f"{nan_rows} rows have missing features after merge. "
            f"Check for url_norm mismatches between split files and the feature matrix."
        )
    return merged[feature_cols].to_numpy(dtype=np.float32), merged["label"].to_numpy(dtype=np.int32)

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def main() -> None:
    for p in (TRAIN_CSV, VAL_CSV, TEST_CSV, AE_FEATURES_CSV, FEATURE_SPEC_JSON):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing required file: {p}\n"
                f"Ensure all earlier phases have been run."
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading splits...")
    train_df = load_split(TRAIN_CSV)
    val_df = load_split(VAL_CSV)
    test_df = load_split(TEST_CSV)
    print(f" Train: {len(train_df):,} Val: {len(val_df):,} Test: {len(test_df):,}")

    feature_cols = load_feature_spec(FEATURE_SPEC_JSON)
    print(f" Feature columns: {len(feature_cols)}")

    X_train, y_train = build_xy(AE_FEATURES_CSV, train_df, feature_cols)
    X_val, y_val = build_xy(AE_FEATURES_CSV, val_df, feature_cols)
    X_test, y_test = build_xy(AE_FEATURES_CSV, test_df, feature_cols)

    print("\nTraining Logistic Regression...")
    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train, y_train)

    # Confirm that the feature column order the model was trained on matches the spec used by the backend. A mismatch
    # here would cause silent errors at inference time with no runtime exception.
    if hasattr(model, "feature_names_in_"):
        trained = list(model.feature_names_in_)
        if trained != feature_cols:
            raise ValueError(
                f"Feature order mismatch.\n"
                f"Model trained on: {trained}\n"
                f"Spec contains: {feature_cols}"
            )

    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    # All three thresholds are derived from validation scores only.
    t_best_f1 = pick_best_f1(y_val, p_val)
    t_fpr_cap = pick_recall_at_fpr(y_val, p_val, fpr_cap=FPR_CAP)
    t_recall_target = pick_precision_at_recall(y_val, p_val, recall_target=RECALL_TARGET)

    thresholds = {
        "best_f1": float(t_best_f1),
        "fpr_cap_0.02": None if t_fpr_cap is None else float(t_fpr_cap),
        "recall_target_0.95": None if t_recall_target is None else float(t_recall_target),
        "params": {"fpr_cap": FPR_CAP, "recall_target": RECALL_TARGET},
        "selected_on": "val",
        "split_basis": "etld1",
    }

    test_best = compute_metrics(y_test, p_test, thresholds["best_f1"])
    test_fpr = (compute_metrics(y_test, p_test, thresholds["fpr_cap_0.02"])
                if thresholds["fpr_cap_0.02"] is not None else None)
    test_rec = (compute_metrics(y_test, p_test, thresholds["recall_target_0.95"])
                if thresholds["recall_target_0.95"] is not None else None)

    dump(model, ART_DIR / "model_lr.joblib")
    write_json(ART_DIR / "thresholds.json", thresholds)

    (ART_DIR / "feature_spec.json").write_text(
        FEATURE_SPEC_JSON.read_text(encoding="utf-8"), encoding="utf-8"
    )

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "logistic_regression",
        "feature_set_id": "AE_v1",
        "n_features": len(feature_cols),
        "seed": SEED,
        "split": "etld1",
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "lr_params": LR_PARAMS,
        "thresholds": thresholds,
        "test_metrics": {
            "best_f1": asdict(test_best),
            "fpr_cap_0.02": None if test_fpr is None else asdict(test_fpr),
            "recall_target_0.95": None if test_rec is None else asdict(test_rec),
        },
    }
    write_json(ART_DIR / "meta.json", meta)

    rows = []
    for name, m in [("best_f1", test_best), ("fpr_cap_0.02", test_fpr),
                    ("recall_target_0.95", test_rec)]:
        if m is None:
            continue
        d = asdict(m)
        d["policy"] = name
        rows.append(d)
    pd.DataFrame(rows).to_csv(OUT_DIR / "final_model_metrics.csv", index=False)
    write_json(OUT_DIR / "final_model_report.json", meta)

    print("\n=== FINAL MODEL EXPORT COMPLETE ===")
    print(f"\nTest metrics (best-F1 threshold = {thresholds['best_f1']:.6f}):")
    print(f" ROC-AUC: {test_best.roc_auc:.4f}")
    print(f" PR-AUC: {test_best.pr_auc:.4f}")
    print(f" Recall: {test_best.recall:.4f}")
    print(f" Precision: {test_best.precision:.4f}")
    print(f" FPR: {test_best.fpr:.4f}")
    if test_fpr:
        print(f"\nTest metrics (fpr_cap_0.02 threshold = {thresholds['fpr_cap_0.02']:.6f}):")
        print(f" Recall @ FPR≤2%: {test_fpr.recall:.4f} FPR: {test_fpr.fpr:.4f}")
    print(f"\nArtefacts written to: {ART_DIR}")

if __name__ == "__main__":
    main()
