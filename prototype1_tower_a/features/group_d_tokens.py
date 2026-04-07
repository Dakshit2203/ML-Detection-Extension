"""
Tower A Feature Extraction - Group D: Token Semantic Features

Extracts 6 features from the token-level lexical content of a normalised URL. Tokenisation splits the URL on common 
delimiter characters (., /, -, _, ?, =,&, #, :) and examines the resulting word segments. This surfaces the vocabulary
embedded in the URL structure - the words a phisher chose to make a counterfeit URL appear legitimate.

Features (6):
  num_tokens              Total token count after splitting on delimiters
  avg_token_len           Mean character length of tokens (0.0 for empty URLs)
  max_token_len           Maximum token length (0 for empty URLs)
  brand_keyword_present   Binary: any token matches a known brand name
  auth_token_present      Binary: any token matches an authentication-related word
  action_token_present    Binary: any token matches an action or urgency word

The three keyword sets are locked and must remain identical across this module, group_e_brand.py, 
scheme_robustness_check.py, and the extension backend.

brand_keyword_present is a weak standalone signal (legitimate brand sites also contain their own name). Its value comes 
from combination with Group E's brand_mismatch, which detects the case where a brand appears in the URL but not in the 
eTLD+1 - the structural signature of impersonation.

Shared between the training pipeline and the Flask inference backend.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# These sets must stay in sync with group_e_brand.py and the extension backend.
AUTH_TOKENS = {
    "login", "signin", "sign-in", "secure", "verify", "verification",
    "auth", "authenticate", "account", "password", "passwd", "session",
}

ACTION_TOKENS = {
    "update", "confirm", "reset", "validate", "unlock",
    "recover", "activate", "continue",
}

BRAND_KEYWORDS = {
    "paypal", "google", "apple", "microsoft", "amazon", "facebook",
    "meta", "instagram", "netflix", "bank", "hsbc", "barclays",
    "lloyds", "natwest", "santander",
}

_TOKEN_RE = re.compile(r"[./\-_?=&#:]+", flags=re.UNICODE)


def _tokenize(url: str) -> list[str]:
    """Split url on URL delimiter characters and return non-empty lowercase tokens."""
    return [t for t in _TOKEN_RE.split(url.lower()) if t]


def extract_group_d(
    df: pd.DataFrame,
    *,
    url_col: str = "url_norm",
) -> pd.DataFrame:
    """
    Extract all 6 Group D token semantic features.

    Parameters
    df - DataFrame with at least url_col.
    url_col - Column containing the normalised URL string.

    Returns a DataFrame of 6 feature columns with the same index as df.
    """
    urls = df[url_col].astype(str)
    tokens = urls.map(_tokenize)

    feats = pd.DataFrame(index=df.index)

    feats["num_tokens"] = tokens.map(len)

    feats["avg_token_len"] = tokens.map(
        lambda t: float(np.mean([len(x) for x in t])) if t else 0.0
    )

    feats["max_token_len"] = tokens.map(
        lambda t: int(max((len(x) for x in t), default=0))
    )

    feats["brand_keyword_present"] = tokens.map(lambda t: int(any(x in BRAND_KEYWORDS for x in t)))
    feats["auth_token_present"] = tokens.map(lambda t: int(any(x in AUTH_TOKENS    for x in t)))
    feats["action_token_present"] = tokens.map(lambda t: int(any(x in ACTION_TOKENS  for x in t)))

    assert feats.shape[1] == 6, f"Expected 6 Group D features, got {feats.shape[1]}"
    return feats