"""
Tower A Feature Extraction - Group C: Shannon Entropy Features

Extracts 3 Shannon entropy (base-2) features from the URL and its components.
Entropy measures the information density of a string's character distribution.

Phishing URLs frequently use algorithmically generated or obfuscated hostnames that exhibit higher entropy than
human-readable legitimate domain names. Long paths stuffed with random hex tokens or base64 payloads similarly inflate
path entropy relative to meaningful path strings like /about or /products.

Features (3):
  entropy_full  Shannon entropy of the entire normalised URL
  entropy_host  Shannon entropy of the hostname component
  entropy_path  Shannon entropy of the path component

An empty string returns 0.0, which is the correct mathematical value - a constant sequence has zero information content.

Shared between the training pipeline and the Flask inference backend.
"""

from __future__ import annotations

import math
from urllib.parse import urlparse

import pandas as pd


def _shannon_entropy(s: str) -> float:
    """
    Compute the Shannon entropy (base-2) of string s.

    H = -sum(p_i * log2(p_i)) where p_i is the relative frequency of each distinct character. Returns 0.0 for empty strings.
    """
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return float(-sum((c / n) * math.log2(c / n) for c in freq.values()))


def extract_group_c(
    df: pd.DataFrame,
    *,
    url_col: str = "url_norm",
) -> pd.DataFrame:
    """
    Extract all 3 Group C entropy features.

    Parameters
    df - DataFrame with at least url_col.
    url_col - Column containing the normalised URL string.

    Returns a DataFrame of 3 feature columns with the same index as df.
    """
    urls = df[url_col].astype(str)

    # Entropy is computed independently per component to capture randomness
    # at different structural levels of the URL.
    parsed = urls.map(lambda u: urlparse(u))
    host = parsed.map(lambda p: (p.hostname or "").lower())
    path = parsed.map(lambda p: p.path or "")

    feats = pd.DataFrame(index=df.index)

    feats["entropy_full"] = urls.map(_shannon_entropy)
    feats["entropy_host"] = host.map(_shannon_entropy)
    feats["entropy_path"] = path.map(_shannon_entropy)

    assert feats.shape[1] == 3, f"Expected 3 Group C features, got {feats.shape[1]}"
    return feats