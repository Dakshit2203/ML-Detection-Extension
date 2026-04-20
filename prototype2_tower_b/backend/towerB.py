"""
Tower B - Prototype 2 - backend/towerB.py

Loads the trained HGB model and feature specification, then produces a p_b phishing probability for a given domain by 
combining the SQLite cache with live network probing.

Role of Tower B in the system
Tower B complements Tower A by operating on domain-level infrastructure signals rather than URL lexical features. This 
means it can detect cases where a URL's text looks harmless but the underlying infrastructure is suspicious 
(e.g., a newly registered domain with a short-lived TLS certificate serving no HSTS header). Conversely, Tower B alone 
has limited recall - purely lexical phishing pages (legitimate-looking infrastructure, suspicious URL text) are 
Tower A's domain.

The two towers are fused in Prototype 3. In Prototype 2, Tower B runs standalone and its output is the sole basis for 
the decision.

Feature vector construction
The feature_spec_B.json file locks the 18 feature columns and their order. When constructing the input row for 
predict_proba(), each feature value is read from the metadata dictionary. If a value is missing (None - indicating a 
failed probe), it is replaced with the NaN.

The HGB model was trained with this same sentinel, so the model has learned what to do when a probe fails. This is 
preferable to imputing a mean or median because it preserves the information that the probe failed, which is itself a 
weak phishing signal (many phishing domains block TLS or HTTP probes after their campaign is detected).

Design note - why not impute missing values?
Using -9999 as a sentinel (rather than, say, column mean imputation) was validated during the Tower B data processing 
phase. The HGB model's internal missing-value handling effectively learns the best split direction for each feature 
when the sentinel is present. Imputing the mean would mask the information that a probe failed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import settings
from .cache_sqlite import SQLiteCache
from .metadata_extract import extract_domain_metadata


class TowerB:
    """
    Loads the trained Tower B model and scores a domain.

    Instantiated once at backend startup. The model and feature spec are loaded from disk at initialisation; the cache
    is opened (or created) at the same time.

    Parameters
    model_path : Path  Path to hgb.joblib.
    feature_spec_path : Path  Path to feature_spec_B.json.
    """

    def __init__(self, model_path: Path, feature_spec_path: Path):
        # Load the trained HGB model
        self.model = joblib.load(model_path)

        # Load the feature contract - 18 column names in their canonical order
        spec = json.loads(Path(feature_spec_path).read_text(encoding="utf-8"))
        self.features: list = spec["features"]

        # Load pre-computed permutation importances produced by 04_train_eval.py.
        # HGB does not expose feature_importances_ directly, so these are saved separately and loaded here for use by
        # the Prototype 3 XAI layer.
        importances_path = Path(feature_spec_path).parent / "feature_importances_B.json"
        if importances_path.exists():
            imp_data = json.loads(importances_path.read_text(encoding="utf-8"))
            imp_map = imp_data.get("importances", {})
            self.importances = [imp_map.get(f, 0.0) for f in self.features]
        else:
            self.importances = None

        # Initialise the domain metadata cache
        self.cache = SQLiteCache(settings.cache_sqlite_path)

    def score(self, domain: str) -> Tuple[Optional[float], Dict[str, Any]]:
        """
        Returns a p_b score for the given domain.

        The method first checks the SQLite cache. If a fresh result is available, the cached metadata is used directly.
        Otherwise, live network probing is performed and the result is stored in the cache.

        The debug dictionary returned alongside the score is included in the /predict response, making it easy to
        inspect cache hits and metadata values during testing.

        Parameters
        domain : str Hostname to score (no scheme, no path).

        Returns
        (float, dict)
            p_b in [0, 1] and a debug dictionary, or
            (None, debug_dict) if scoring failed.
        """
        debug: Dict[str, Any] = {
            "domain":      domain,
            "cache_hit":   False,
            "cache_age_s": None,
        }

        # Cache lookup
        cached = self.cache.get(domain, ttl_seconds=settings.cache_ttl_seconds)
        if cached:
            meta               = cached.value
            debug["cache_hit"]   = True
            debug["cache_age_s"] = cached.age_seconds
        else:
            # Live network probe
            meta = extract_domain_metadata(
                domain,
                tls_timeout_s  = settings.tls_timeout_s,
                http_timeout_s = settings.http_timeout_s,
            )
            self.cache.set(domain, meta)

        # Build feature row in the canonical column order
        row = [
            meta.get(f, None)  # None becomes NaN in the DataFrame - handled natively by HGB
            for f in self.features
        ]
        X = np.array([row], dtype=np.float64).reshape(1, -1)

        # Inference
        try:
            if hasattr(self.model, "predict_proba"):
                p_b = float(self.model.predict_proba(X)[0, 1])
            else:
                # Fallback for models that expose decision_function instead
                z = float(self.model.decision_function(X)[0])
                p_b = float(1.0 / (1.0 + np.exp(-z)))

            debug["ok"] = True
            debug["meta"] = meta
            return p_b, debug

        except Exception as exc:
            debug["ok"] = False
            debug["error"] = str(exc)
            debug["meta"] = meta
            return None, debug
