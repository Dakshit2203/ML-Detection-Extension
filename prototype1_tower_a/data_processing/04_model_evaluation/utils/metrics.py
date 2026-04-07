"""Model evaluation utilities - metric computation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    fpr: float
    fnr: float
    tp: int
    fp: int
    tn: int
    fn: int

def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Metrics:
    y_hat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0, 1]).ravel()

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, y_hat, average="binary", zero_division=0
    )
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_acc = 0.5 * (rec + specificity)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    roc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else 0.5
    pr = average_precision_score(y, p) if len(np.unique(y)) == 2 else 0.5

    return Metrics(
        roc_auc=float(roc), pr_auc=float(pr),
        accuracy=float(acc), balanced_accuracy=float(balanced_acc),
        precision=float(prec), recall=float(rec), f1=float(f1),
        specificity=float(specificity), fpr=float(fpr), fnr=float(fnr),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )

def pick_threshold_best_f1(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Select the threshold that maximises F1 on the validation set."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(p_val):
        m = compute_metrics(y_val, p_val, float(t))
        if m.f1 > best_f1:
            best_f1, best_t = m.f1, float(t)
    return best_t

def pick_threshold_recall_at_fpr(y_val: np.ndarray, p_val: np.ndarray,
                                  fpr_cap: float) -> Optional[float]:
    """
    Select the threshold with the highest recall subject to FPR <= fpr_cap.
    Thresholds are scanned from highest to lowest to favour lower FPR candidates when recall is tied.
    """
    best_t, best_recall = None, -1.0
    for t in np.unique(p_val)[::-1]:
        m = compute_metrics(y_val, p_val, float(t))
        if m.fpr <= fpr_cap and m.recall > best_recall:
            best_recall, best_t = m.recall, float(t)
    return best_t

def pick_threshold_precision_at_recall(y_val: np.ndarray, p_val: np.ndarray,
                                        recall_target: float) -> Optional[float]:
    """
    Select the threshold with the highest precision subject to recall >= recall_target.
    """
    best_t, best_prec = None, -1.0
    for t in np.unique(p_val)[::-1]:
        m = compute_metrics(y_val, p_val, float(t))
        if m.recall >= recall_target and m.precision > best_prec:
            best_prec, best_t = m.precision, float(t)
    return best_t


def to_dict(m: Metrics) -> Dict:
    return asdict(m)
