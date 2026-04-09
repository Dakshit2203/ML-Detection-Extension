"""
Tower B - Prototype 2 - backend/config.py

All configuration for the Tower B backend is centralised here.
No other module contains hardcoded paths or magic numbers.

Prototype 2 context
Tower B scores URLs based solely on domain-level infrastructure metadata:
DNS resolution behaviour, TLS certificate characteristics, and HTTP response headers. It operates independently of
Tower A's lexical features and is intended to complement Tower A rather than replace it.

The model (hgb.joblib) was trained on 18 numeric features derived from live network probing of the 92,666 domains in
the shared dataset. See feature_spec_B.json for the locked feature list.

Artifact location
Artifacts are stored in artifacts/towerB/ (inside the prototype root), not in backend/artifacts/. This keeps the
trained model and feature spec alongside the prototype rather than buried inside the backend package.

Cache
Metadata probing is network-bound and slow (typically 2–8 seconds per domain across DNS, TLS, and HTTP calls). A SQLite
cache persists probing results for 7 days so that revisited domains are served instantly.
The cache file is written to backend/.cache_domain_meta.sqlite by default.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

# Root of the prototype2_tower_b/ directory (one level up from backend/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Artefacts directory - contains the towerB/ subdirectory with model + spec
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TOWER_B_DIR = ARTIFACTS_DIR / "towerB"

@dataclass(frozen=True)
class Settings:
    """
    Immutable settings for the Tower B backend.

    Frozen dataclass: any attempt to mutate a setting at runtime raises an exception, preventing accidental
    configuration drift across requests.
    """

    # Network probe timeouts
    # These control how long the metadata extractor waits for each signal.
    # Values are conservative: TLS handshakes on slow servers can take 2–3 s, and HTTP responses including redirects
    # can take 3–5 s.
    dns_timeout_s: float = 2.0   # getaddrinfo timeout
    tls_timeout_s: float = 3.0   # SSL handshake timeout
    http_timeout_s: float = 4.0   # Full HTTP round-trip timeout

    # SQLite domain metadata cache
    # Probing results are cached per eTLD+1 domain. The TTL of 7 days reflects the expectation that infrastructure
    # signals (DNS, TLS cert) change slowly for legitimate domains and change rapidly for short-lived phishing domains.
    cache_sqlite_path: Path = PROJECT_ROOT / "backend" / ".cache_domain_meta.sqlite"
    cache_ttl_seconds: int = 7 * 24 * 3600  # 7 days

    # Decision policy
    # "fixed" uses constant thresholds (safe for Prototype 2 standalone).
    # "adaptive_quantile" would require a live buffer as in Tower A; this is not needed here since Tower B feeds into
    # the Prototype 3 fusion layer.
    decision_mode: str = "fixed"

    # Fixed thresholds tuned against the Tower B validation set operating points.
    threshold_red:    float = 0.99  # score >= red  -> danger / block
    # score >= orange -> suspicious / warn
    threshold_orange: float = 0.50  # midpoint - Tower B used as signal not blocker

    # Adaptive mode parameters (unused in Prototype 2 standalone)
    # Retained so that the decision_policy module can be reused in Prototype 3 without modification if an adaptive
    # mode is enabled later.
    buffer_max:        int   = 500
    min_samples:       int   = 80
    target_fpr_red:    float = 0.02
    target_fpr_orange: float = 0.05
    min_scores_to_block: int = 80


# Module-level singleton - imported by all other backend modules.
settings = Settings()
