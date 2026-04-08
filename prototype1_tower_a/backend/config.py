"""
Tower A - Prototype 1 - backend/config.py

Purpose
Centralised path and parameter configuration for the Tower A backend. All module-level constants used across the 
backend are defined here so that no other file contains hardcoded paths or magic numbers.
"""

from __future__ import annotations
from pathlib import Path

# Root paths
BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_lr.joblib"
FEATURE_SPEC_PATH = ARTIFACTS_DIR / "feature_spec.json"
FEATURES_DIR = BASE_DIR / "features"

# Adaptive threshold parameters
DEFAULT_TARGET_FPR_RED = 0.02 # Target FPR for the "danger" (block) threshold.
DEFAULT_TARGET_FPR_ORANGE = 0.05 # Target FPR for the "suspicious" (warn) threshold.
SCORE_BUFFER_MAXLEN = 500 # Maximum number of scores retained in the adaptive buffer (rolling window).
MIN_SCORES_TO_ADAPT = 50 # Minimum scores required before the quantile threshold is computed
MIN_SCORES_TO_BLOCK = 150 # Minimum scores required before blocking is enabled
 