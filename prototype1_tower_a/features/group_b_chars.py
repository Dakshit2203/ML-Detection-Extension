"""
Tower A Feature Extraction - Group B: Character Composition Features

Extracts 10 features from the character-level composition of a normalised URL. These capture common obfuscation
techniques used in phishing URLs: excessive digit use (IP address substitution), high symbol density (encoding tricks),
unusual @ counts (credential harvesting), percent signs (encoding markers), and uppercase characters in path/query
(preserved by the normalisation contract,which deliberately does not lowercase path or query components).

Features (10):
  digit_ratio       Ratio of digit characters to total URL length
  alpha_ratio       Ratio of alphabetic characters to total URL length
  symbol_ratio      Ratio of non-alphanumeric characters to total URL length
  upper_ratio_pq    Ratio of uppercase characters in path + query combined
  count_at          Count of @ symbols
  count_dash        Count of hyphens
  count_underscore  Count of underscores
  count_percent     Count of % characters (percent-encoding markers)
  count_equal       Count of = signs (query parameter separators)
  count_question    Count of ? characters

All ratios use total URL length as the denominator. A length of 0 is treated as 1 to avoid division by zero.
upper_ratio_pq is computed over path + query only, since scheme and host are already lowercased by the normalisation
contract.

Shared between the training pipeline and the Flask inference backend.
"""

from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd


def _upper_ratio(path: str, query: str) -> float:
    """
    Fraction of uppercase characters in the combined path and query string.

    Phishing URLs sometimes include uppercase letters in paths to evade case-sensitive blacklist matching. The
    normalisation contract preserves case here, so uppercase is a genuine content signal.
    """
    combined = path + query
    if not combined:
        return 0.0
    return sum(1 for c in combined if c.isupper()) / len(combined)


def extract_group_b(
    df: pd.DataFrame,
    *,
    url_col: str = "url_norm",
) -> pd.DataFrame:
    """
    Extract all 10 Group B character composition features.

    Parameters
    df - DataFrame with at least url_col.
    url_col - Column containing the normalised URL string.

    Returns a DataFrame of 10 feature columns with the same index as df.
    """
    urls = df[url_col].astype(str)

    parsed = urls.map(lambda u: urlparse(u))
    path = parsed.map(lambda p: p.path  or "")
    query = parsed.map(lambda p: p.query or "")

    url_len = urls.str.len().replace(0, 1)

    feats = pd.DataFrame(index=df.index)

    feats["digit_ratio"] = urls.str.count(r"\d") / url_len
    feats["alpha_ratio"] = urls.str.count(r"[A-Za-z]") / url_len
    feats["symbol_ratio"] = urls.map(
        lambda u: sum(1 for c in u if not c.isalnum())
    ) / url_len

    feats["upper_ratio_pq"] = [_upper_ratio(str(p), str(q)) for p, q in zip(path, query)]
    feats["count_at"] = urls.str.count(r"@")
    feats["count_dash"] = urls.str.count(r"-")
    feats["count_underscore"] = urls.str.count(r"_")
    feats["count_percent"] = urls.str.count(r"%")
    feats["count_equal"] = urls.str.count(r"=")
    feats["count_question"] = urls.str.count(r"\?")

    assert feats.shape[1] == 10, f"Expected 10 Group B features, got {feats.shape[1]}"
    return feats