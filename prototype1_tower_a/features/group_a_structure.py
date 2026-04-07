"""
Tower A Feature Extraction - Group A: Structural URL Features

Extracts 12 features from the structural components of a normalised URL. All features in this group operate on the
parsed URL parts (host, path, query, fragment) rather than the raw URL string, so they are independent of scheme presence.

Features (12):
    url_len         Total character count of the normalised URL
    host_len        Character count of the hostname
    path_len        Character count of the path component
    query_len       Character count of the query string
    num_dots        Number of dot separators in the hostname
    num_slashes     Total slash count in the full URL
    num_subdomains  Subdomain label count above the eTLD+1 (-1 if eTLD+1 missing)
    etld1_missing   Binary: PSL-based eTLD+1 could not be resolved
    has_query       Binary: query string is present
    has_fragment    Binary: fragment component is present
    has_port        Binary: an explicit port is present in the URL
    is_ip_host      Binary: the hostname is a raw IP address rather than a domain

Called by extract_features.py (pipeline) and by the Flask inference backend. The function signature and column names
are locked - changes here must be reflected in feature_spec.json and the extension backend.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

import pandas as pd


def _is_ip(host: str) -> int:
    """Return 1 if host is a valid IPv4 or IPv6 address, 0 otherwise."""
    try:
        ipaddress.ip_address(host)
        return 1
    except ValueError:
        return 0

def _subdomain_count(host: str, etld1: str, etld1_missing: bool) -> int:
    """
    Count subdomain labels present above the eTLD+1.

    Returns -1 when eTLD+1 is unresolvable so the model distinguishes a genuinely absent value from a domain with
    zero subdomains.
    """
    if etld1_missing:
        return -1
    host_parts  = host.split(".") if host else []
    etld1_parts = etld1.split(".") if etld1 else []
    return max(len(host_parts) - len(etld1_parts), 0)

def extract_group_a(
    df: pd.DataFrame,
    *,
    url_col: str = "url_norm",
    etld1_col: str = "etld1",
) -> pd.DataFrame:
    """
    Extract all 12 Group A structural features.

    Parameters
    df - DataFrame with at least url_col and etld1_col columns.
    url_col - Column containing the normalised URL string.
    etld1_col - Column containing the PSL-resolved eTLD+1 (may be NaN).

    Returns a DataFrame of 12 feature columns with the same index as df.
    """
    urls  = df[url_col].astype(str)
    etld1 = (
        df[etld1_col] if etld1_col in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )

    # A missing eTLD+1 occurs when tldextract cannot resolve a public suffix, typically for IP hosts, bare TLDs, or
    # unknown private suffixes.
    etld1_missing = etld1.isna() | (etld1.astype(str).str.strip() == "")

    parsed = urls.map(lambda u: urlparse(u))
    host = parsed.map(lambda p: (p.hostname or "").lower())
    path = parsed.map(lambda p: p.path or "")
    query = parsed.map(lambda p: p.query or "")
    fragment = parsed.map(lambda p: p.fragment or "")
    port = parsed.map(lambda p: p.port)

    feats = pd.DataFrame(index=df.index)

    feats["url_len"] = urls.str.len()
    feats["host_len"] = host.str.len()
    feats["path_len"] = path.str.len()
    feats["query_len"] = query.str.len()
    feats["num_dots"] = host.str.count(r"\.")
    feats["num_slashes"] = urls.str.count("/")

    feats["num_subdomains"] = [
        _subdomain_count(str(h), str(e), bool(m))
        for h, e, m in zip(host, etld1, etld1_missing)
    ]

    feats["etld1_missing"] = etld1_missing.astype(int)
    feats["has_query"] = (query.str.len() > 0).astype(int)
    feats["has_fragment"] = (fragment.str.len() > 0).astype(int)
    feats["has_port"] = port.notna().astype(int)
    feats["is_ip_host"] = host.map(_is_ip).astype(int)

    assert feats.shape[1] == 12, f"Expected 12 Group A features, got {feats.shape[1]}"
    return feats