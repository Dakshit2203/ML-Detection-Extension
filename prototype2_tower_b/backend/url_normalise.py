"""
Tower B - Prototype 2 - backend/url_normalise.py

URL utilities for the Tower B backend.

Tower B only needs the hostname extracted from a URL - it does not process the path, query string, or fragment. 
Despite this, normalisation and scanability filtering are still applied so the behaviour is consistent with Tower A and 
predictable across both prototypes.

Three functions are exported:

  is_scannable_url(url) -> bool
    Returns True for http/https URLs; False for browser-internal schemes.

  normalise_url_for_inference(url) -> str
    Strips whitespace and ensures a scheme is present.

  extract_hostname(url_norm) -> str | None
    Extracts just the host component (no port) for use as the cache key and as the target for DNS/TLS/HTTP probing.
"""

from __future__ import annotations

from urllib.parse import urlparse


def is_scannable_url(url: str) -> bool:
    """
    Returns True if the URL is a normal http/https page that Tower B can probe.

    Browser-internal schemes (chrome://, about:, file://) cannot be probed for DNS or TLS signals and must be silently
    skipped. This check mirrors the equivalent function in the Tower A backend so both towers agree on what is skipped.

    Parameters
    url : str
        Raw URL string from the extension navigation event.
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
    Applies minimal normalisation before hostname extraction.

    Tower B does not need full URL normalisation (path, query, fragment are irrelevant). Only ensure a scheme is
    present so that urlparse can identify the host component correctly.

    Parameters
    url : str
        Raw URL string.

    Returns
    str
        URL with a scheme guaranteed present.
    """
    u = (url or "").strip()
    if not u:
        return u
    if not u.lower().startswith(("http://", "https://")):
        u = "https://" + u
    return u


def extract_hostname(url_norm: str) -> str | None:
    """
    Extracts the hostname from a normalised URL string.
    The returned hostname is used as:
      - The SQLite cache key for metadata lookups.
      - The target domain for DNS, TLS, and HTTP probing.

    Leading/trailing dots are stripped (e.g. ".example.com." -> "example.com") and the result is lowercased for
    consistent cache key matching.

    Parameters
    url_norm : str
        URL with a scheme already present.

    Returns
    str or None
        Lowercased hostname, or None if extraction fails.
    """
    try:
        host = urlparse(url_norm).hostname
        if not host:
            return None
        return host.strip(".").lower()
    except Exception:
        return None
