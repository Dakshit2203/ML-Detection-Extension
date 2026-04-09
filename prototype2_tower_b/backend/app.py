"""
Tower B - Prototype 2 - backend/app.py

FastAPI application providing the Tower B /predict endpoint.

Prototype 2 purpose
This backend demonstrates Tower B operating as a standalone detector using only domain-level infrastructure
metadata (DNS, TLS, HTTP). It does not use Tower A's lexical features. This standalone deployment is an intermediate
step before Prototype 3 fuses both towers.

Running the backend standalone is useful for two reasons:
  1. It proves that Tower B can independently classify domains based on infrastructure signals before combining it
  with Tower A.
  2. It provides a testable artifact that confirms the metadata extraction pipeline, caching, and model inference all
  work correctly in isolation.

Endpoints
  GET  /health   Liveness check. Confirms the model is loaded.
  GET  /config   Returns the active threshold and cache configuration.
  POST /predict  Scores a URL using Tower B metadata signals.

Data flow for POST /predict
  1. Reject unscannable URLs (chrome://, file://, etc.) silently.
  2. Extract the hostname from the URL (Tower B operates on domains only).
  3. Call TowerB.score(hostname):
       a. Check SQLite cache for a fresh result.
       b. If cache miss: probe DNS, TLS, HTTP (2–8 seconds total).
       c. Build the 18-feature vector and call HGB predict_proba().
  4. Apply the decision policy (fixed thresholds by default).
  5. Return p_b, risk_level, decision, threshold state, and a debug block.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .config import settings, TOWER_B_DIR
from .url_normalise import is_scannable_url, normalise_url_for_inference, extract_hostname
from .towerB import TowerB
from .decision_policy import make_policy

# Application and singleton initialisation
app = FastAPI(
    title="Tower B - Prototype 2 Backend",
    version="1.0.0",
    description=(
        "Domain metadata phishing scorer using HGB on 18 DNS/TLS/HTTP features. "
        "Standalone deployment - part of the Two-Tower Hybrid ML Framework."
    )
)

# Both singletons are created once at startup and reused for every request.
# TowerB loads the model and opens the SQLite cache connection.
# policy is either FixedThresholdPolicy or AdaptiveQuantilePolicy.
tower_b = TowerB(
    model_path = TOWER_B_DIR / "hgb.joblib",
    feature_spec_path = TOWER_B_DIR / "feature_spec_B.json",
)

policy = make_policy()

# Request schema
class PredictRequest(BaseModel):
    """ Request body for POST /predict. """
    url: str
    mode: str = "auto"

# Endpoints
@app.get("/")
def root():
    """Root endpoint - confirms the backend is running."""
    return {
        "status":  "ok",
        "message": "Tower B backend running. Use /health or /docs.",
        "port":    8001,
    }

@app.get("/health")
def health():
    """
    Liveness check. Returns the model load status and active decision mode.
    The extension or test scripts can poll this before making /predict calls.
    """
    return {
        "status":        "ok",
        "towerB_loaded": tower_b.model is not None,
        "decision_mode": settings.decision_mode,
        "cache_ttl_days": settings.cache_ttl_seconds // (24 * 3600),
    }

@app.get("/config")
def config():
    """
    Returns the full active configuration for inspection during testing.
    Useful for confirming that threshold values and timeouts are as expected without having to read config.py directly.
    """
    return {
        "decision_mode": settings.decision_mode,
        "thresholds": {
            "threshold_red":    settings.threshold_red,
            "threshold_orange": settings.threshold_orange,
        },
        "cache": {
            "sqlite_path": str(settings.cache_sqlite_path),
            "ttl_seconds": settings.cache_ttl_seconds,
        },
        "timeouts_s": {
            "dns":  settings.dns_timeout_s,
            "tls":  settings.tls_timeout_s,
            "http": settings.http_timeout_s,
        },
    }

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Scores a URL using Tower B's domain metadata pipeline.

    The response includes a debug block containing the raw metadata values and cache hit status. This is intentionally
    verbose for Prototype 2 to make the metadata signals observable during development and testing.

    The debug block will be removed or collapsed in the Prototype 3 extension popup, which only displays the final p_b score.
    """
    raw = req.url or ""

    # Scanability check
    if not is_scannable_url(raw):
        return {
            "ok":           True,
            "skipped":      True,
            "reason":       "unsupported_scheme",
            "url":          raw,
            "url_normalised": raw,
        }

    # Hostname extraction
    # Tower B operates on domain level only. If the URL has no parseable host, return a safe default (allow) rather
    # than raising an error.
    url_norm = normalise_url_for_inference(raw)
    hostname = extract_hostname(url_norm)
    if not hostname:
        return {
            "ok":           True,
            "skipped":      True,
            "reason":       "no_hostname",
            "url":          raw,
            "url_normalised": url_norm,
        }

    # Score via Tower B pipeline
    p_b, debug = tower_b.score(hostname)

    # If the model returned None (e.g., prediction error), return a safe default.
    if p_b is None:
        return {
            "ok":           True,
            "skipped":      False,
            "reason":       "scoring_failed",
            "url":          raw,
            "url_normalised": url_norm,
            "hostname":     hostname,
            "p_b":          None,
            "risk_level":   "unknown",
            "decision":     "allow",
            "thresholds":   {"mode": settings.decision_mode},
            "debug":        debug,
        }

    # Update adaptive buffer if that policy is active
    if hasattr(policy, "update"):
        policy.update(p_b)

    # Apply decision policy
    outcome = policy.decide(p_b)

    return {
        "ok":           True,
        "skipped":      False,
        "reason":       None,
        "url":          raw,
        "url_normalised": url_norm,
        "hostname":     hostname,
        "p_b":          p_b,
        "risk_level":   outcome["risk_level"],
        "decision":     outcome["decision"],
        "thresholds":   outcome["thresholds"],
        "debug":        debug,
    }
