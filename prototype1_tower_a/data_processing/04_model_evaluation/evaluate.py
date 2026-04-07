"""
Tower A - Phase 4: Single Model Evaluation

Trains and evaluates one (split, regime, model) configuration. Threshold policies are selected on the validation set
and reported on the held-out test set. Results are written as a JSON file and as a row in the shared summary CSV.

Called by run_all.py for each combination; can also be run directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from models.logistic_regression import build_model as build_lr
from models.random_forest import build_model as build_rf
from models.hist_gradient_boosting import build_model as build_hgb
from models.sgd_ngrams import build_model as build_sgd

from utils.features import (
    split_paths,
    load_split_csv,
    resolve_features_ae_path,
    build_ae_xy,
    build_tfidf_vectorizer,
)
from utils.io_utils import (
    project_root,
    ensure_dir,
    write_json,
    upsert_summary_row,
    serialized_size_mb,
)
from utils.metrics import (
    compute_metrics,
    pick_threshold_best_f1,
    pick_threshold_recall_at_fpr,
    pick_threshold_precision_at_recall,
    to_dict,
)
from utils.repro import set_single_thread_env, set_seeds
from utils.timing import time_ms, ms_per_item

MODEL_FACTORIES = {
    "lr": build_lr,
    "rf": build_rf,
    "hgb": build_hgb,
    "sgd": build_sgd,
}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "etld1"], required=True)
    ap.add_argument("--regime", choices=["AE", "ngrams"], required=True)
    ap.add_argument("--model", choices=["lr", "rf", "hgb", "sgd"], required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fpr_cap", type=float, default=0.02)
    ap.add_argument("--recall_target", type=float, default=0.95)
    ap.add_argument("--latency_sample", type=int, default=10_000)
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    set_single_thread_env()
    set_seeds(args.seed)

    root = project_root()
    out_dir = root / "04_model_evaluation" / "outputs"
    json_dir = out_dir / "json"
    ensure_dir(json_dir)

    train_path, val_path, test_path = split_paths(root, args.split)
    train_df = load_split_csv(train_path)
    val_df = load_split_csv(val_path)
    test_df = load_split_csv(test_path)

    if args.regime == "ngrams" and args.model != "sgd":
        raise RuntimeError("regime=ngrams requires model=sgd.")

    model = MODEL_FACTORIES[args.model](seed=args.seed)

    meta: Dict[str, Any] = {
        "split": args.split, "regime": args.regime, "model": args.model,
        "seed": args.seed, "train_path": str(train_path),
        "val_path": str(val_path), "test_path": str(test_path),
        "fpr_cap": args.fpr_cap, "recall_target": args.recall_target,
    }

    feature_extraction_ms_per_url: Optional[float] = None
    feature_extraction_note: Optional[str] = None

    if args.regime == "AE":
        feats_path = resolve_features_ae_path(root)
        meta["features_ae_path"] = str(feats_path)

        X_train, y_train, feature_cols = build_ae_xy(feats_path, train_df)
        X_val, y_val, _ = build_ae_xy(feats_path, val_df)
        X_test, y_test, _ = build_ae_xy(feats_path, test_df)

        meta["n_features"] = int(len(feature_cols))
        # AE features are precomputed in Phase 2; extraction time is not attributable to the model and is therefore
        # not measured here.
        feature_extraction_note = "AE features precomputed in Phase 2; extraction latency not measured."

        model.fit(X_train, y_train)
        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        n = min(args.latency_sample, X_test.shape[0])
        Xs = X_test[:n]
        total_ms = time_ms(lambda: model.predict_proba(Xs), runs=5, warmup=1)
        inference_ms_per_url = ms_per_item(total_ms, n)
        end_to_end_ms_per_url = inference_ms_per_url
        model_mb = serialized_size_mb(model)
        vectorizer_mb = None

    else:
        vectorizer = build_tfidf_vectorizer(max_features=50_000, min_df=2)
        meta.update({"tfidf_ngram_range": "3-5", "tfidf_max_features": 50_000, "tfidf_min_df": 2})

        X_train_text = train_df["url_norm"].tolist()
        X_val_text = val_df["url_norm"].tolist()
        X_test_text = test_df["url_norm"].tolist()

        y_train = train_df["label"].to_numpy(dtype=np.int32)
        y_val = val_df["label"].to_numpy(dtype=np.int32)
        y_test = test_df["label"].to_numpy(dtype=np.int32)

        X_train = vectorizer.fit_transform(X_train_text)
        X_val = vectorizer.transform(X_val_text)
        X_test = vectorizer.transform(X_test_text)

        meta["tfidf_vocab_size"] = int(len(vectorizer.vocabulary_))
        model.fit(X_train, y_train)
        p_val = model.predict_proba(X_val)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        # Measure vectorisation and inference separately to distinguish feature extraction cost from model scoring cost.
        n = min(args.latency_sample, len(X_test_text))
        test_s = X_test_text[:n]
        tx_ms = time_ms(lambda: vectorizer.transform(test_s), runs=3, warmup=1)
        Xt = vectorizer.transform(test_s)
        inf_ms = time_ms(lambda: model.predict_proba(Xt), runs=3, warmup=1)

        feature_extraction_ms_per_url = ms_per_item(tx_ms, n)
        inference_ms_per_url = ms_per_item(inf_ms, n)
        end_to_end_ms_per_url = feature_extraction_ms_per_url + inference_ms_per_url
        model_mb = serialized_size_mb(model)
        vectorizer_mb = serialized_size_mb(vectorizer)

    # All threshold policies are selected on val only; test set is never used to influence threshold choice.
    best_f1_t = pick_threshold_best_f1(y_val, p_val)
    t_fpr_cap = pick_threshold_recall_at_fpr(y_val, p_val, fpr_cap=args.fpr_cap)
    t_recall_target = pick_threshold_precision_at_recall(y_val, p_val, recall_target=args.recall_target)

    thresholds: Dict[str, Optional[float]] = {
        "best_f1": float(best_f1_t),
        "recall_at_fpr_cap": None if t_fpr_cap is None else float(t_fpr_cap),
        "precision_at_recall_target": None if t_recall_target is None else float(t_recall_target),
    }

    val_best = compute_metrics(y_val, p_val, best_f1_t)
    val_policy: Dict[str, Any] = {}
    if t_fpr_cap is not None:
        val_policy["recall_at_fpr_cap"] = to_dict(compute_metrics(y_val, p_val, t_fpr_cap))
    if t_recall_target is not None:
        val_policy["precision_at_recall_target"] = to_dict(compute_metrics(y_val, p_val, t_recall_target))

    test_best = compute_metrics(y_test, p_test, best_f1_t)
    test_policy: Dict[str, Any] = {}
    if t_fpr_cap is not None:
        test_policy["recall_at_fpr_cap"] = to_dict(compute_metrics(y_test, p_test, t_fpr_cap))
    if t_recall_target is not None:
        test_policy["precision_at_recall_target"] = to_dict(compute_metrics(y_test, p_test, t_recall_target))

    results: Dict[str, Any] = {
        "meta": meta,
        "thresholds": thresholds,
        "val_metrics_best_f1": to_dict(val_best),
        "val_policy_metrics": val_policy,
        "test_metrics_best_f1": to_dict(test_best),
        # "test_policy_metrics" is the key read by plot_results.py.
        "test_policy_metrics": test_policy,
        "efficiency": {
            "feature_extraction_ms_per_url": feature_extraction_ms_per_url,
            "feature_extraction_note": feature_extraction_note,
            "inference_ms_per_url": float(inference_ms_per_url),
            "end_to_end_ms_per_url": float(end_to_end_ms_per_url),
            "model_size_mb": float(model_mb),
            "vectorizer_size_mb": vectorizer_mb,
        },
    }

    json_name = f"{args.split}__{args.regime}__{args.model}__seed{args.seed}.json"
    write_json(json_dir / json_name, results)

    summary_csv = out_dir / "summary_results.csv"
    flat = {
        "split": args.split, "regime": args.regime, "model": args.model, "seed": args.seed,
        "threshold_best_f1": thresholds["best_f1"],
        "roc_auc": test_best.roc_auc, "pr_auc": test_best.pr_auc,
        "precision": test_best.precision, "recall": test_best.recall,
        "f1": test_best.f1, "accuracy": test_best.accuracy,
        "balanced_accuracy": test_best.balanced_accuracy,
        "specificity": test_best.specificity, "fpr": test_best.fpr,
        "fnr": test_best.fnr,
        "tp": test_best.tp, "fp": test_best.fp, "tn": test_best.tn, "fn": test_best.fn,
        "feature_extraction_ms_per_url": feature_extraction_ms_per_url,
        "inference_ms_per_url": results["efficiency"]["inference_ms_per_url"],
        "end_to_end_ms_per_url": results["efficiency"]["end_to_end_ms_per_url"],
        "model_size_mb": results["efficiency"]["model_size_mb"],
        "vectorizer_size_mb": results["efficiency"]["vectorizer_size_mb"],
    }
    upsert_summary_row(summary_csv, flat)

    print(f"  {args.split}/{args.regime}/{args.model}  "
          f"PR-AUC={test_best.pr_auc:.4f} ROC-AUC={test_best.roc_auc:.4f}  "
          f"FPR={test_best.fpr:.4f} latency={end_to_end_ms_per_url:.5f} ms/URL")
    print(f" JSON: {json_dir / json_name}")

if __name__ == "__main__":
    main()