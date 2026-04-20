"""
Prototype 3 - Fused System - backend/config.py

Centralised configuration for the Prototype 3 backend.

Prototype 3 integrates both towers into a single FastAPI application. Rather than running two separate backends 
(port 8000 and 8001 as in Prototypes 1 and 2), the fused system loads both models at startup and runs them in parallel 
via ThreadPoolExecutor on a single port (8002).

The fusion formula is:
p_fused = 0.75 * p_A + 0.25 * p_B

The 3:1 weighting reflects Tower A's empirically stronger performance (PR-AUC 0.9744 vs Tower B's 0.3340 on the
validation set). Tower B acts as a correction signal rather than a co-equal detector.

The same three-phase mechanism from Prototype 1 is applied to p_fused. Thresholds calibrate to real browsing data rather 
than training-derived fixed values, resolving the FPR=1.0 failure found in Prototype 1 Phase 6 external validation.
"""

from __future__ import annotations
from pathlib import Path

# Root paths

# Root of the prototype3_fused/ directory
BASE_DIR = Path(__file__).resolve().parents[1]

# Prototype 1 artifacts - Tower A model and feature spec
TOWER_A_ARTIFACTS = (
    BASE_DIR.parent
    / "prototype1_tower_a" / "backend" / "artifacts"
)
MODEL_A_PATH = TOWER_A_ARTIFACTS / "model_lr.joblib"
FEATURE_SPEC_A = TOWER_A_ARTIFACTS / "feature_spec.json"

# Prototype 1 features directory - Groups A–E extraction modules
FEATURES_A_DIR = (
    BASE_DIR.parent
    / "prototype1_tower_a" / "features"
)

# Prototype 2 artifacts - Tower B model and feature spec
TOWER_B_ARTIFACTS = (
    BASE_DIR.parent
    / "prototype2_tower_b" / "artifacts" / "towerB"
)
MODEL_B_PATH = TOWER_B_ARTIFACTS / "hgb.joblib"
FEATURE_SPEC_B = TOWER_B_ARTIFACTS / "feature_spec_B.json"

# Prototype 2 backend - Tower B inference and cache modules
TOWER_B_BACKEND = (
    BASE_DIR.parent
    / "prototype2_tower_b"
)

# Prototype 3 local artifacts - adaptive threshold state
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ADAPTIVE_STATE = ARTIFACTS_DIR / "adaptive_state.json"


# Fusion weights
WEIGHT_A = 0.75
WEIGHT_B = 0.25


# Adaptive threshold parameters (mirrors Prototype 1 config)
TARGET_FPR_RED = 0.02  # target FPR for danger/block threshold
TARGET_FPR_ORANGE = 0.05  # target FPR for suspicious/warn threshold
SCORE_BUFFER_MAXLEN = 500  # rolling window size
MIN_SCORES_TO_ADAPT = 50  # Phase 1 -> Phase 2 boundary
MIN_SCORES_TO_BLOCK = 150  # Phase 2 -> Phase 3 boundary

# Conservative fallback thresholds used during Phase 1 (learning mode).
# These are set high deliberately so that nothing is blocked before the adaptive buffer has enough data to calibrate a
# meaningful threshold.
FALLBACK_RED = 0.99
FALLBACK_ORANGE = 0.95

# Minimum fused score that triggers a warning during Phase 1.
# Kept very high (0.9999) to avoid false positives in the early scans.
LEARNING_WARN_AT = 0.9999

# Tower B metadata extraction timeouts
DNS_TIMEOUT_S = 2.0
TLS_TIMEOUT_S = 3.0
HTTP_TIMEOUT_S = 4.0
CACHE_TTL_S = 7 * 24 * 3600  # 7 days

# SQLite cache for Tower B domain metadata
CACHE_DB = (
    BASE_DIR.parent
    / "prototype2_tower_b" / "backend" / ".cache_domain_meta.sqlite"
)