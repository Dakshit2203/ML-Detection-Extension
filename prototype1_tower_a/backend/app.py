"""
Tower A - Prototype 1 - backend/app.py

Purpose
FastAPI application providing the /predict endpoint consumed by the Chrome extension's service worker.

Endpoints
  GET /health     - Liveness check; returns {"status": "ok"}.
  POST /predict    - Score a URL with the Tower A LR model.
  POST /reset-adaptive  - Clear the adaptive threshold buffer (for testing).

Three-phase adaptive threshold
The core architectural novelty of Prototype 1. Standard training thresholds (best_f1=0.297, fpr_cap=0.146) produce
FPR=1.0 on externally distributed benign URLs (Phase 6 finding). The adaptive system resolves this by deriving
thresholds from a rolling buffer of observed real-browser scores.

Running the backend
-------------------
From the prototype1_tower_a/ directory:
    uvicorn backend.app:app --host 127.0.0.1 --port 8000

Or with auto-reload during development:
    uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import (
    MODEL_PATH,
    FEATURE_SPEC_PATH,
    FEATURES_DIR,
    ARTIFACTS_DIR,
    DEFAULT_TARGET_FPR_RED,
    DEFAULT_TARGET_FPR_ORANGE,
    SCORE_BUFFER_MAXLEN,
    MIN_SCORES_TO_ADAPT,
    MIN_SCORES_TO_BLOCK,
)
from .inference import TowerAInference
from .url_normalise import normalise_url_for_inference, is_scannable_url
from .adaptive_threshold import AdaptiveThreshold

# Application and singleton initialisation
app = FastAPI(
    title="Tower A - Prototype 1 Backend",
    version="1.0.0",
    description=(
        "Phishing URL scorer using LR on 35 handcrafted URL features "
        "with adaptive quantile-based thresholding. "
        "Part of the Two-Tower Hybrid ML Framework (COMP1682 FYP)."
    )
)

# TowerAInference is expensive to initialise (model deserialisation + module
# imports). It is instantiated once at startup and reused for every request.
predictor = TowerAInference(MODEL_PATH, FEATURE_SPEC_PATH, FEATURES_DIR)

# AdaptiveThreshold persists its score buffer to adaptive_state.json so it
# survives backend restarts. The buffer grows as real browsing sessions
# accumulate scores and calibrates the thresholds accordingly.
adaptive = AdaptiveThreshold(
    state_path=ARTIFACTS_DIR / "adaptive_state.json",
    maxlen=SCORE_BUFFER_MAXLEN,
    min_samples=MIN_SCORES_TO_ADAPT,
)

# Phase 1 fallback thresholds. These are deliberately conservative so that
# nothing is blocked during the learning phase.
# FALLBACK_RED (0.99): only clear-cut phishing URLs (p≥99%) trigger even
# a warn during Phase 1 (since LEARNING_MODE_WARN_AT = 0.9999 takes
# precedence, the effective threshold during Phase 1 is 0.9999).
# FALLBACK_ORANGE: not used during Phase 1 (overridden by phase logic).
FALLBACK_RED = 0.99
FALLBACK_ORANGE = 0.95

# Minimum p_phish that triggers a warning during Phase 1 (learning mode).
# Set very high to avoid annoying the user before the buffer is calibrated.
LEARNING_MODE_WARN_AT = 0.9999

# Request / response schemas
class PredictRequest(BaseModel):
    # Request body for POST /predict.
    url: str = Field(..., description="Raw URL to score.")
    mode: str | None = Field(default="auto", description="auto | manual")

class PredictResponse(BaseModel):
    #Response body for POST /predict.
    ok: bool
    skipped: bool = False
    reason: str | None = None
    url: str
    url_normalised: str | None = None
    p_phish: float | None = None
    risk_level: str | None = None
    decision: str | None = None
    thresholds: dict | None = None
    
# Endpoints
@app.get("/")
def root():
    """Root endpoint - confirms the backend is running."""
    return {"status": "ok", "message": "Tower A backend running. Use /health or /docs."}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "buffer_size": len(adaptive.state.scores),
    }

@app.post("/reset-adaptive")
def reset_adaptive():
    adaptive.reset()
    return {"ok": True, "buffer_size": len(adaptive.state.scores)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Step 1: Scanability check 
    scannable, reason = is_scannable_url(req.url)
    if not scannable:
        return PredictResponse(ok=True, skipped=True, reason=reason, url=req.url)

    # Step 2: Normalisation 
    url_norm = normalise_url_for_inference(req.url)

    # Steps 3–4: Feature extraction + LR inference 
    p = predictor.predict_proba(url_norm)

    # Step 5: Buffer update (p_phish score only - no URL stored)
    adaptive.update(p)

    # Step 6: Derive thresholds from current buffer state 
    th = adaptive.get_thresholds(
        target_fpr_red=DEFAULT_TARGET_FPR_RED,
        target_fpr_orange=DEFAULT_TARGET_FPR_ORANGE,
        fallback_red=FALLBACK_RED,
        fallback_orange=FALLBACK_ORANGE,
    )
    learning_mode = bool(th["learning_mode"])
    t_red = float(th["threshold_red"])
    t_orange = float(th["threshold_orange"])
    buffer_size = int(th["buffer_size"])

    # Step 7: Phase-aware decision policy 

    if learning_mode:
        # Phase 1: learning - conservative fallbacks, no blocking
        if p >= LEARNING_MODE_WARN_AT:
            risk, decision = "suspicious", "warn"
        elif p >= 0.50:
            risk, decision = "low", "allow"
        else:
            risk, decision = "safe", "allow"

    elif buffer_size < MIN_SCORES_TO_BLOCK:
        # Phase 2: adapting - thresholds active but block downgraded to warn
        if p >= t_red:
            risk, decision = "danger", "warn" # block -> warn (sample not stable)
        elif p >= t_orange:
            risk, decision = "suspicious", "warn"
        elif p >= 0.50:
            risk, decision = "low", "allow"
        else:
            risk, decision = "safe", "allow"

    else:
        # Phase 3: stable - full three-level enforcement
        if p >= t_red:
            risk, decision = "danger", "block"
        elif p >= t_orange:
            risk, decision = "suspicious", "warn"
        elif p >= 0.50:
            risk, decision = "low", "allow"
        else:
            risk, decision = "safe", "allow"

    # Add the phase label to the threshold dict for popup display
    th["phase"] = (
        "learning" if learning_mode else
        "adapting_no_block" if buffer_size < MIN_SCORES_TO_BLOCK else
        "stable_block_enabled"
    )

    # Step 8: Return response 
    return PredictResponse(
        ok=True,
        url=req.url,
        url_normalised=url_norm,
        p_phish=p,
        risk_level=risk,
        decision=decision,
        thresholds=th,
    )
