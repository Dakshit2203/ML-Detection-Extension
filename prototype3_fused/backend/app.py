"""
Prototype 3 - Fused System - backend/app.py

FastAPI application delivering the full two-tower fused /predict endpoint.

Both models are loaded once at startup and shared across all requests. Tower A and Tower B are submitted to a
ThreadPoolExecutor so their work runs concurrently - important because Tower B network probes can take 2–8 seconds
while Tower A completes in under 1 ms.

Run from prototype3_fused/:
    uvicorn backend.app:app --host 127.0.0.1 --port 8002
"""

from __future__ import annotations

import importlib.util
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from .config import (
    MODEL_A_PATH, FEATURE_SPEC_A, FEATURES_A_DIR,
    MODEL_B_PATH, FEATURE_SPEC_B, TOWER_B_BACKEND,
    ARTIFACTS_DIR, ADAPTIVE_STATE,
    WEIGHT_A, WEIGHT_B,
    TARGET_FPR_RED, TARGET_FPR_ORANGE,
    SCORE_BUFFER_MAXLEN, MIN_SCORES_TO_ADAPT, MIN_SCORES_TO_BLOCK,
    FALLBACK_RED, FALLBACK_ORANGE, LEARNING_WARN_AT,
    DNS_TIMEOUT_S, TLS_TIMEOUT_S, HTTP_TIMEOUT_S, CACHE_TTL_S,
)
from .url_normalise import is_scannable_url, normalise_url_for_inference, extract_hostname
from .fusion import fuse, apply_decision_policy
from .xai import explain_tower_a, explain_tower_b


def _load(module_name: str, file_path: Path):
    """
    Loads a Python module directly from its file path and registers it in
    sys.modules under module_name. This avoids the namespace collision that
    occurs when both prototype1_tower_a/ and prototype2_tower_b/ each contain
    a package named 'backend' — importing by path bypasses that ambiguity.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ── Tower A — Prototype 1 ─────────────────────────────────────────────────────

P1_BACKEND = Path(MODEL_A_PATH).parents[1]  # prototype1_tower_a/backend/

# The feature group modules (group_a_structure etc.) are imported by name
# inside TowerAInference.__init__, so the features/ directory must be on sys.path.
if str(FEATURES_A_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_A_DIR))

_p1_inf = _load("p1_inference", P1_BACKEND / "inference.py")
_p1_adp = _load("p1_adaptive", P1_BACKEND / "adaptive_threshold.py")

TowerAInference = _p1_inf.TowerAInference
AdaptiveThreshold = _p1_adp.AdaptiveThreshold

# ── Tower B — Prototype 2 ─────────────────────────────────────────────────────

P2_BACKEND = TOWER_B_BACKEND / "backend"

# towerB.py uses relative imports (from .config import settings, etc.).
# Registering a fake package called "p2_backend" means those relative imports
# resolve correctly: ".config" becomes "p2_backend.config" and so on.
_p2_pkg = types.ModuleType("p2_backend")
_p2_pkg.__path__ = [str(P2_BACKEND)]
_p2_pkg.__package__ = "p2_backend"
sys.modules["p2_backend"] = _p2_pkg

_load("p2_backend.config", P2_BACKEND / "config.py")
_load("p2_backend.cache_sqlite", P2_BACKEND / "cache_sqlite.py")
_load("p2_backend.metadata_extract", P2_BACKEND / "metadata_extract.py")
_load("p2_backend.towerB", P2_BACKEND / "towerB.py")

TowerB = sys.modules["p2_backend.towerB"].TowerB


# Singletons - created once at startup
app = FastAPI(
    title="Two-Tower Fused System - Prototype 3",
    version="1.0.0",
)

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

tower_a = TowerAInference(MODEL_A_PATH, FEATURE_SPEC_A, FEATURES_A_DIR)
tower_b = TowerB(MODEL_B_PATH, FEATURE_SPEC_B)
adaptive = AdaptiveThreshold(
    state_path=ADAPTIVE_STATE,
    maxlen=SCORE_BUFFER_MAXLEN,
    min_samples=MIN_SCORES_TO_ADAPT,
)

# Two workers - one per tower
executor = ThreadPoolExecutor(max_workers=2)


class PredictRequest(BaseModel):
    url: str
    mode: str = "auto"

# Endpoints
@app.get("/")
def root():
    return {"status": "ok", "message": "Prototype 3 fused backend. Use /health or /docs.", "port": 8002}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "tower_a_loaded": tower_a.model is not None,
        "tower_b_loaded": tower_b.model is not None,
        "fusion_weights": {"tower_a": WEIGHT_A, "tower_b": WEIGHT_B},
        "buffer_size": len(adaptive.state.scores),
    }


@app.get("/config")
def config():
    th = adaptive.get_thresholds(
        target_fpr_red=TARGET_FPR_RED,
        target_fpr_orange=TARGET_FPR_ORANGE,
        fallback_red=FALLBACK_RED,
        fallback_orange=FALLBACK_ORANGE,
    )
    return {
        "fusion": {"weight_a": WEIGHT_A, "weight_b": WEIGHT_B},
        "adaptive_threshold": th,
        "timeouts_s": {"dns": DNS_TIMEOUT_S, "tls": TLS_TIMEOUT_S, "http": HTTP_TIMEOUT_S},
    }

@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    """
    Full two-tower prediction pipeline:
      1. Reject unscannable URLs.
      2. Run Tower A and Tower B concurrently via ThreadPoolExecutor.
      3. Fuse: p_fused = 0.75*pA + 0.25*pB (falls back to pA if Tower B fails).
      4. Update the adaptive buffer and derive thresholds.
      5. Apply the three-phase decision policy.
      6. Compute XAI attributions for both towers.
      7. Return the full response.
    """
    if not is_scannable_url(req.url):
        return {"ok": True, "skipped": True, "reason": "unsupported_scheme", "url": req.url}

    url_norm = normalise_url_for_inference(req.url)
    hostname = extract_hostname(url_norm)

    if not hostname:
        return {"ok": True, "skipped": True, "reason": "no_hostname", "url": req.url, "url_normalized": url_norm}

    # Submit both towers concurrently
    def run_tower_a():
        features_df = tower_a.extract_features(url_norm)
        p_a = tower_a.predict_proba(url_norm)
        return p_a, features_df

    def run_tower_b():
        return tower_b.score(hostname)

    future_a = executor.submit(run_tower_a)
    future_b = executor.submit(run_tower_b)

    p_a, features_df = future_a.result()
    p_b, tower_b_debug = future_b.result()

    # Fuse scores
    p_fused, tower_b_available = fuse(p_a, p_b)

    # Update adaptive buffer and derive thresholds
    adaptive.update(p_fused)
    th = adaptive.get_thresholds(
        target_fpr_red=TARGET_FPR_RED,
        target_fpr_orange=TARGET_FPR_ORANGE,
        fallback_red=FALLBACK_RED,
        fallback_orange=FALLBACK_ORANGE,
    )

    outcome = apply_decision_policy(
        p_fused = p_fused,
        learning_mode = bool(th["learning_mode"]),
        buffer_size = int(th["buffer_size"]),
        t_red = float(th["threshold_red"]),
        t_orange = float(th["threshold_orange"]),
        min_to_block = MIN_SCORES_TO_BLOCK,
        learning_warn_at = LEARNING_WARN_AT,
    )
    th["phase"] = outcome["phase"]

    # XAI attributions
    xai_a = explain_tower_a(tower_a.model, features_df, tower_a.feature_cols, top_n=5)

    xai_b = []
    if tower_b_available and p_b is not None:
        meta = tower_b_debug.get("meta", {})
        xai_b = explain_tower_b(
            tower_b.model,
            tower_b.features,
            meta,
            importances=tower_b.importances,
            top_n=5,
        )

    return {
        "ok": True,
        "skipped": False,
        "reason": None,

        "url": req.url,
        "url_normalized": url_norm,
        "hostname": hostname,

        "p_a": round(p_a, 6),
        "p_b": round(p_b, 6) if p_b is not None else None,
        "tower_b_available": tower_b_available,
        "p_fused": round(p_fused, 6),

        "risk_level": outcome["risk_level"],
        "decision": outcome["decision"],

        "fusion": {
            "weight_a": WEIGHT_A,
            "weight_b": WEIGHT_B,
            "formula": f"p_fused = {WEIGHT_A}*p_A + {WEIGHT_B}*p_B",
        },

        "thresholds": th,
        "xai": {"tower_a": xai_a, "tower_b": xai_b},
        "debug": {"tower_b": tower_b_debug},
    }
