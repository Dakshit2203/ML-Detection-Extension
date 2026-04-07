"""
Tower A Feature Extraction - Group E: Brand Impersonation Features

Extracts 4 features that detect structural brand impersonation in URLs. These identify a specific phishing technique: 
placing a known brand name somewhere in the URL (subdomain, path, query) while the actual registrant domain (eTLD+1) 
belongs to a different, typically malicious entity. The combination of brand presence and eTLD+1 mismatch is a stronger 
signal than brand presence alone, which Group D captures as brand_keyword_present.

Features (4):
  brand_in_subdomain: Binary: a brand keyword appears in a subdomain label but not in the eTLD+1 
                      (e.g., paypal.malicious-site.com)
  brand_in_path       Binary: a brand keyword appears in the path or query (e.g., malicious.com/paypal/verify)
  brand_mismatch      Binary: a brand keyword appears anywhere in the URL but is absent from the eTLD+1 token set
                      - the primary impersonation signal
  deep_subdomain      Binary: three or more subdomain labels above the eTLD+1, indicating structural nesting used to 
                      obscure the real registered domain (e.g., a.b.c.real-domain.com)

The BRAND_KEYWORDS set is locked and must match group_d_tokens.py exactly.

For brand_mismatch, when eTLD+1 is missing the feature defaults to 0 to avoid false positives caused by an inability to
resolve the suffix rather than genuine impersonation.

Shared between the training pipeline and the Flask inference backend.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import pandas as pd

# Must stay in sync with group_d_tokens.py and the extension backend.
BRAND_KEYWORDS = {
    "paypal", "google", "apple", "microsoft", "amazon", "facebook",
    "meta", "instagram", "netflix", "bank", "hsbc", "barclays",
    "lloyds", "natwest", "santander",
}

_DELIM_RE = re.compile(r"[.\-]+")
_PATH_DELIM_RE = re.compile(r"[./\-_?=&#:]+")


def _host_tokens(host: str) -> set[str]:
    """Split hostname on dots and hyphens; return non-empty lowercase tokens."""
    return {t for t in _DELIM_RE.split(host.lower()) if t} if host else set()

def _path_tokens(path: str) -> set[str]:
    """Split path on common URL delimiters; return non-empty lowercase tokens."""
    return {t for t in _PATH_DELIM_RE.split(path.lower()) if t} if path else set()

def _etld1_tokens(etld1: str) -> set[str]:
    """Tokenise the eTLD+1 string for brand mismatch comparison."""
    if not isinstance(etld1, str) or not etld1.strip():
        return set()
    return {t for t in _DELIM_RE.split(etld1.lower()) if t}

def extract_group_e(
    df: pd.DataFrame,
    *,
    url_col: str = "url_norm",
    etld1_col: str = "etld1",
) -> pd.DataFrame:
    """
    Extract all 4 Group E brand impersonation features.

    Parameters
    df - DataFrame with at least url_col and etld1_col.
    url_col - Column containing the normalised URL string.
    etld1_col - Column containing the PSL-resolved eTLD+1 (may be NaN).

    Returns a DataFrame of 4 feature columns with the same index as df.
    """
    urls  = df[url_col].astype(str)
    etld1 = (
        df[etld1_col] if etld1_col in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )

    etld1_missing = etld1.isna() | (etld1.astype(str).str.strip() == "")

    parsed = urls.map(lambda u: urlparse(u))
    host_str = parsed.map(lambda p: (p.hostname or "").lower())
    path_str = parsed.map(lambda p: p.path  or "")
    query_str = parsed.map(lambda p: p.query or "")

    host_tok_s = host_str.map(_host_tokens)
    path_tok_s = (path_str + "/" + query_str).map(_path_tokens)
    etld1_tok_s = etld1.astype(str).map(_etld1_tokens)

    # brand_in_subdomain: brand appears in the host label set but not in the eTLD+1 portion, meaning it sits in a 
    # subdomain position rather than the registrant domain.
    brand_in_subdomain = []
    for h_toks, e_toks, e_missing in zip(host_tok_s, etld1_tok_s, etld1_missing):
        host_brand = h_toks & BRAND_KEYWORDS
        if e_missing or not host_brand:
            brand_in_subdomain.append(0)
        else:
            brand_in_subdomain.append(int(bool(host_brand - (e_toks & BRAND_KEYWORDS))))

    # brand_in_path: brand appears in path or query tokens.
    brand_in_path = [int(bool(pt & BRAND_KEYWORDS)) for pt in path_tok_s]

    # brand_mismatch: brand appears anywhere in the URL but not in the eTLD+1.
    # Conservatively set to 0 when eTLD+1 is unresolvable.
    brand_mismatch = []
    for h_toks, pt_toks, e_toks, e_missing in zip(
        host_tok_s, path_tok_s, etld1_tok_s, etld1_missing
    ):
        all_url_brands = (h_toks | pt_toks) & BRAND_KEYWORDS
        if e_missing or not all_url_brands:
            brand_mismatch.append(0)
        else:
            brand_mismatch.append(int(not bool(e_toks & BRAND_KEYWORDS)))

    # deep_subdomain: three or more subdomain labels above the eTLD+1 indicates deliberate nesting to obscure the real 
    # registered domain.
    deep_subdomain = []
    for h, e, e_missing in zip(host_str, etld1.astype(str), etld1_missing):
        if e_missing:
            deep_subdomain.append(0)
        else:
            sub_count = max(len(h.split(".") if h else []) - len(e.split(".") if e else []), 0)
            deep_subdomain.append(int(sub_count >= 3))

    feats = pd.DataFrame(index=df.index)
    feats["brand_in_subdomain"] = brand_in_subdomain
    feats["brand_in_path"] = brand_in_path
    feats["brand_mismatch"] = brand_mismatch
    feats["deep_subdomain"] = deep_subdomain

    assert feats.shape[1] == 4, f"Expected 4 Group E features, got {feats.shape[1]}"
    return feats
