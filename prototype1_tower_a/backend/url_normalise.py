"""
Tower A — Prototype 1 — backend/url_normalise.py

Purpose
URL normalisation and scanability filtering for the Tower A backend.

Two functions are exported:
  is_scannable_url(url) -> (bool, reason_str)
    Returns (True, "ok") if the URL is a normal http/https page that the feature extraction pipeline can process.
    Returns (False, reason) for browser-internal URLs that must be silently skipped.

  normalise_url_for_inference(url) -> str
    Applies minimal normalisation (strip, lowercase scheme and host) to produce the canonical URL form expected by
    the feature modules.

The isScannableUrl() function in service_worker.js performs the same scheme-prefix checks. Both implementations must
agree on what is skipped to avoid the extension sending URLs that the backend would reject.
"""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

# Determines whether a URL can be scored by the Tower A model.
def is_scannable_url(url: str) -> tuple[bool, str]:
    u = (url or "").strip().lower()

    if not u:
        return False, "empty_url"

    # Browser-internal schemes that cannot be processed by the feature pipeline.
    # This list mirrors the equivalent check in service_worker.js (isScannableUrl).
    blocked_prefixes = (
        "chrome://",
        "chrome-extension://",
        "edge://",
        "about:",
        "file://",
        "view-source:",
    )
    if u.startswith(blocked_prefixes):
        return False, "unsupported_scheme"

    return True, "ok"

# Applies minimal normalisation to a URL before feature extraction.
# Normalisation steps applied:
# 1. Strip leading/trailing whitespace.
# 2. Prepend "http://" if no scheme is present (bare domains).
# 3. Lowercase the scheme and host components only.
#
# The path, query string, and fragment are left unchanged to preserve the lexical signal content that the feature
# modules rely on.

def normalise_url_for_inference(url: str) -> str:
    u = (url or "").strip()
    if not u:
        return u

    # Add default scheme for bare domains (e.g., "example.com/path")
    if not u.lower().startswith(("http://", "https://")):
        u = "http://" + u

    # Lowercase scheme and host; preserve path, query, fragment as-is.
    parsed = urlparse(u)
    normalised = urlunparse((
        parsed.scheme.lower(),  # scheme: lowercase
        parsed.netloc.lower(),  # host+port: lowercase
        parsed.path,  # path: unchanged (case-sensitive in some servers)
        parsed.params,  # params: unchanged
        parsed.query,  # query string: unchanged
        parsed.fragment  # fragment: unchanged
    ))
    return normalised