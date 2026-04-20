"""
Tower A - Phase 2: Scheme Influence Robustness Check

Verifies that feature separability in Groups B–E is not primarily driven by whether a URL had an explicit scheme
(http:// or https://) in the raw data. During preprocessing, scheme-less URLs receive a placeholder http:// prefix,
so had_scheme_raw and placeholder_added are strongly correlated with label. If Groups B–E were sensitive to scheme
presence, they would be indirectly leaking label information through that correlation rather than capturing genuine
obfuscation patterns.

Method:
  Groups A–E are computed twice. Once on url_norm as-is, and once on a scheme-stripped version of url_norm for the
  B–E string computations only. Group A is computed identically in both cases since it uses urlparse(), which requires
  a scheme for correct parsing.

  If AUC, KS, and Cohen's d remain stable across variants, scheme presence is not a meaningful driver of feature
  separability.

Inputs:  01_preprocessing/outputs/towerA_dataset.csv
Outputs: 02_feature_engineering/outputs/scheme_robustness/
  features_handcrafted_AE_scheme_neutral.csv
  descriptive_feature_metrics_scheme_neutral.csv
  group_separation_summary_scheme_neutral.csv
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PROC = BASE_DIR.parent
DATASET_PATH = DATA_PROC / "01_preprocessing" / "outputs" / "towerA_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "scheme_robustness"

# These keyword lists must match group_d_tokens.py and group_e_brand.py exactly. Any divergence would mean the
# robustness check is testing different features from those actually used in training.
AUTH_TOKENS = {
    "login","signin","sign-in","secure","verify","verification",
    "auth","authenticate","account","password","passwd","session"
}
ACTION_TOKENS = {
    "update","confirm","reset","validate","unlock","recover","activate","continue"
}
BRAND_KEYWORDS = {
    "paypal","google","apple","microsoft","amazon","facebook",
    "meta","instagram","netflix","bank","hsbc","barclays",
    "lloyds","natwest","santander"
}

TOKEN_SPLIT_RE  = re.compile(r"[./\-_?=&#:]+", flags=re.UNICODE)
SCHEME_STRIP_RE = re.compile(r"^https?://", flags=re.IGNORECASE)

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    return float(h)

def parse_url(url: str):
    p = urlparse(url)
    return p.hostname or "", p.path or "", p.query or "", p.fragment or "", p.port

def is_ip_host(host: str) -> int:
    import ipaddress
    try:
        ipaddress.ip_address(host)
        return 1
    except Exception:
        return 0

def etld1_tokens(etld1: str) -> set[str]:
    if not isinstance(etld1, str) or not etld1.strip():
        return set()
    return {p for p in re.split(r"[.\-]+", etld1.lower()) if p}

def tokenize(url_str: str) -> list[str]:
    return [t for t in TOKEN_SPLIT_RE.split(url_str.lower()) if t]

def cohens_d(x0: np.ndarray, x1: np.ndarray) -> float:
    n0, n1 = x0.size, x1.size
    if n0 < 2 or n1 < 2:
        return 0.0
    m0, m1 = float(x0.mean()), float(x1.mean())
    v0, v1 = float(x0.var(ddof=1)), float(x1.var(ddof=1))
    sp = math.sqrt(((n0-1)*v0 + (n1-1)*v1) / (n0+n1-2)) if (n0+n1-2) else 0.0
    return float((m1-m0)/sp) if sp else 0.0

def single_feature_auc(y: np.ndarray, s: np.ndarray) -> float:
    if np.all(s == s[0]):
        return 0.5
    auc = float(roc_auc_score(y, s))
    return max(auc, 1.0 - auc)

def iqr_val(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, 75) - np.percentile(x, 25))

def extract_features_ae(df: pd.DataFrame, scheme_neutral_be: bool) -> pd.DataFrame:
    """
    Extract Groups A–E.
    When scheme_neutral_be=True, Groups B–E use the scheme-stripped URL string while Group A still parses from the
    full url_norm (urlparse requires a scheme for hostname extraction to work correctly).
    """
    url_norm = df["url_norm"].astype(str)
    etld1 = df.get("etld1", pd.Series([None]*len(df), index=df.index))
    etld1_missing = etld1.isna() | (etld1.astype(str).str.strip() == "")

    parsed = url_norm.map(parse_url)
    parts = pd.DataFrame(parsed.tolist(), columns=["host","path","query","fragment","port"])
    host = parts["host"].astype(str)
    path = parts["path"].astype(str)
    query = parts["query"].astype(str)
    fragment = parts["fragment"].astype(str)

    url_for_be = (
        url_norm.str.replace(SCHEME_STRIP_RE, "", regex=True)
        if scheme_neutral_be else url_norm
    )

    feats = pd.DataFrame(index=df.index)

    feats["A_url_len"] = url_norm.str.len()
    feats["A_host_len"] = host.str.len()
    feats["A_path_len"] = path.str.len()
    feats["A_query_len"] = query.str.len()
    feats["A_num_dots"] = host.str.count(r"\.")
    feats["A_num_slashes"] = url_norm.str.count("/")
    feats["A_has_query"] = (query.str.len() > 0).astype(int)
    feats["A_has_fragment"] = (fragment.str.len() > 0).astype(int)
    feats["A_has_port"] = parts["port"].notna().astype(int)
    feats["A_is_ip_host"] = host.map(is_ip_host).astype(int)
    feats["A_etld1_missing"] = etld1_missing.astype(int)

    def subdomain_count(h: str, e: str, missing: bool) -> int:
        if missing:
            return -1
        hl = len(h.split(".")) if h else 0
        el = len(e.split(".")) if e else 0
        return max(hl - el, 0)

    feats["A_num_subdomains"] = [
        subdomain_count(h_, e_, m_)
        for h_, e_, m_ in zip(host.astype(str), etld1.astype(str), etld1_missing.tolist())
    ]

    s = url_for_be.astype(str)
    n = s.str.len().replace(0, 1)
    feats["B_digit_ratio"] = s.str.count(r"\d") / n
    feats["B_alpha_ratio"] = s.str.count(r"[A-Za-z]") / n
    feats["B_symbol_ratio"] = s.map(lambda x: sum(1 for ch in x if not ch.isalnum())) / n
    feats["B_count_pct"] = s.str.count("%")
    feats["B_count_at"] = s.str.count("@")
    feats["B_count_dash"] = s.str.count("-")
    feats["B_count_underscore"] = s.str.count("_")
    pq = (path + "?" + query).astype(str)
    feats["B_uppercase_in_pq"] = pq.map(lambda x: int(any(ch.isupper() for ch in x)))

    feats["C_entropy_url"] = s.map(shannon_entropy)
    feats["C_entropy_host"] = host.astype(str).map(shannon_entropy)
    feats["C_entropy_path"] = path.astype(str).map(shannon_entropy)

    tokens = s.map(tokenize)
    feats["D_num_tokens"] = tokens.map(len)
    feats["D_avg_token_len"] = tokens.map(
        lambda t: float(np.mean([len(x) for x in t])) if t else 0.0
    )
    feats["D_max_token_len"] = tokens.map(lambda t: int(max((len(x) for x in t), default=0)))
    feats["D_auth_token_present"] = tokens.map(lambda t: int(any(x in AUTH_TOKENS for x in t)))
    feats["D_action_token_present"] = tokens.map(lambda t: int(any(x in ACTION_TOKENS for x in t)))
    feats["D_brand_present"] = tokens.map(lambda t: int(any(x in BRAND_KEYWORDS for x in t)))

    host_toks = host.astype(str).str.lower().map(
        lambda h: set(re.split(r"[.\-]+", h)) if h else set()
    )
    path_toks = path.astype(str).str.lower().map(
        lambda p: set(re.split(r"[./\-_]+", p)) if p else set()
    )
    feats["E_brand_in_host"] = [int(len(ht & BRAND_KEYWORDS) > 0) for ht in host_toks]
    feats["E_brand_in_path"] = [int(len(pt & BRAND_KEYWORDS) > 0) for pt in path_toks]

    etld1_tok = etld1.astype(str).map(etld1_tokens)
    brand_any = feats["D_brand_present"].astype(int)
    mismatch = []
    for present, e_miss, e_tok in zip(brand_any.tolist(), etld1_missing.tolist(), etld1_tok.tolist()):
        if e_miss:
            mismatch.append(0)
        else:
            mismatch.append(int(present == 1 and not any(b in e_tok for b in BRAND_KEYWORDS)))
    feats["E_brand_mismatch"] = mismatch

    return feats

def evaluate_features(features: pd.DataFrame, labels: pd.Series,
                      group_prefix_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = labels.to_numpy().astype(int)
    out_rows = []
    for col in features.columns:
        x = features[col].to_numpy()
        x0 = x[y == 0]
        x1 = x[y == 1]
        try:
            ks = float(ks_2samp(x0, x1).statistic)
        except Exception:
            ks = 0.0
        try:
            auc = single_feature_auc(y, x.astype(float))
        except Exception:
            auc = 0.5
        out_rows.append({
            "feature": col,
            "mean_benign": float(np.mean(x0)),
            "median_benign": float(np.median(x0)),
            "std_benign": float(np.std(x0, ddof=1)) if x0.size > 1 else 0.0,
            "iqr_benign": iqr_val(x0),
            "mean_phish": float(np.mean(x1)),
            "median_phish": float(np.median(x1)),
            "std_phish": float(np.std(x1, ddof=1)) if x1.size > 1 else 0.0,
            "iqr_phish": iqr_val(x1),
            "cohens_d": cohens_d(x0.astype(float), x1.astype(float)),
            "ks_stat": ks,
            "single_feature_auc": auc,
        })
    metrics = pd.DataFrame(out_rows)

    def group_of(feat: str) -> str:
        for prefix, gname in group_prefix_map.items():
            if feat.startswith(prefix):
                return gname
        return "Other"

    metrics["group"] = metrics["feature"].map(group_of)

    g = metrics.groupby("group", dropna=False)
    group_summary = g.agg(
        n_features = ("feature", "count"),
        auc_mean = ("single_feature_auc", "mean"),
        auc_median = ("single_feature_auc", "median"),
        auc_max = ("single_feature_auc", "max"),
        ks_mean = ("ks_stat", "mean"),
        d_mean = ("cohens_d", "mean"),
    ).reset_index()
    pct_above = g.apply(
        lambda x: float(np.mean(x["single_feature_auc"] >= 0.60))
    ).reset_index(name="pct_auc_ge_0.60")
    group_summary = group_summary.merge(pct_above, on="group", how="left")
    return metrics, group_summary

def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            f"Run 01_preprocessing/preprocess.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f" Rows: {len(df):,}")

    feats_neutral = extract_features_ae(df, scheme_neutral_be=True)

    feat_path = OUTPUT_DIR / "features_handcrafted_AE_scheme_neutral.csv"
    pd.concat([df[["url_norm","label"]], feats_neutral], axis=1).to_csv(feat_path, index=False)
    print(f" Written: {feat_path}")

    group_map = {
        "A_": "Group A", "B_": "Group B", "C_": "Group C",
        "D_": "Group D", "E_": "Group E",
    }
    metrics_df, group_df = evaluate_features(feats_neutral, df["label"], group_map)

    metrics_path = OUTPUT_DIR / "descriptive_feature_metrics_scheme_neutral.csv"
    group_path = OUTPUT_DIR / "group_separation_summary_scheme_neutral.csv"
    metrics_df.to_csv(metrics_path, index=False)
    group_df.to_csv(group_path, index=False)
    print(f" Written: {metrics_path}")
    print(f" Written: {group_path}")

    print("\nGroup-level AUC (scheme-neutral):")
    for _, row in group_df.iterrows():
        print(f" Group {row['group']}: mean_auc={row['auc_mean']:.4f}")
    print("\nCompare these values against descriptive_feature_metrics.csv.")


if __name__ == "__main__":
    main()