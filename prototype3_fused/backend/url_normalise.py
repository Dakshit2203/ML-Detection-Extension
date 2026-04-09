"""
Prototype 3 - Fused System - backend/url_normalize.py

URL normalisation and hostname extraction for the fused backend. Reproduced from Prototypes 1 and 2 so Prototype 3 is
self-contained at import time and does not depend on the internal structure of earlier prototypes.
"""

from __future__ import annotations
from urllib.parse import urlparse, urlunparse


def is_scannable_url(url: str) -> bool:
    """
    Returns True for http/https URLs that both towers can process.
    Browser-internal schemes have no URL structure for Tower A and no probing target for Tower B.
    """
    if not url:
        return False
    u = str(url).strip().lower()
    return not (
        u.startswith("chrome://") or
        u.startswith("chrome-extension://") or
        u.startswith("edge://") or
        u.startswith("about:") or
        u.startswith("file://") or
        u.startswith("view-source:")
    )


def normalise_url_for_inference(url: str) -> str:
    """
    Lowercases the scheme and host; preserves the path exactly.
    Minimal normalisation is intentional - the Tower A feature modules must receive the same URL form that was used
    during training.
    """
    u = (url or "").strip()
    if not u:
        return u
    if not u.lower().startswith(("http://", "https://")):
        u = "https://" + u
    parsed = urlparse(u)
    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))


def extract_hostname(url_norm: str) -> str | None:
    """
    Returns the lowercased hostname for Tower B domain-level probing and SQLite cache key lookup.
    """
    try:
        host = urlparse(url_norm).hostname
        return host.strip(".").lower() if host else None
    except Exception:
        return None
