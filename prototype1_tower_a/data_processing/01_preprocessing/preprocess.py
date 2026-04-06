"""
Tower A - Phase 1: URL Preprocessing abd Dataset Construction

Merges three raw data sources into single balanced dataset of 92,666 URLs.
Raw CSVs go in 01_preprocessing/raw/

Sources:
    Phishtank.csv           phishing URLS (filter: varified=yes, online=yes)
    Tranco.csv              top-1M benign domains
    malicious_phish.csv     Kaggle benign URL collection (filter: type=benign)

Outputs written to 01_preprocessing/outputs/:
    towerA_dataset.csv          final merged dataset
    preprocessing_summary.txt   per-source statistic report
    dataset_clean/              individually cleaned source files

Normalisation contract:
    - Strip leading/trailing whitespace and wrapper quotes
    - Reject URLs containing internal whitespace
    - Add http:// prefix when no scheme is present
    - Lowercase scheme and host only
    - Preserve path, quary, and fragment verbatim
"""

import csv
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import tldextract

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
OUTPUTS_DIR = BASE_DIR / "outputs"
CLEAN_DIR = OUTPUTS_DIR / "dataset_clean"
FINAL_OUTPUT = OUTPUTS_DIR / "towerA_dataset.csv"
SUMMARY_TXT = OUTPUTS_DIR / "preprocessing_summary.txt"

PHISHTANK_PATH = RAW_DIR / "Phishtank.csv"
TRANCO_PATH = RAW_DIR / "Tranco.csv"
KAGGLE_PATH = RAW_DIR / "malicious_phish.csv"

SEED = 42
SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

def pct(n, d: int) -> str:
    return "0.00%" if d == 0 else f"{100.0 * n / d:.2f}%"

def normalise_url(raw) -> tuple:
    """
    Apply the URL normalisation contract and return (url_norm, meta, error_reason).

    Return error_reason=None on success, or a descriptive string on failure.
    path, query, and fragment are preserved verbatim
    """
    if not isinstance(raw, str):
        return None, None, "non_string"

    s = raw.strip()

    if (s.startswith('"') and s.endswith('"') or s.startswith("'") and s.endswith("'")):
        s=s[1:-1]

    if re.search(r"\s", s):
        return None, None, "internal_whitespace"

    had_scheme = bool(SCHEME_RE.match(s))
    placeholder_added = False
    if not had_scheme:
        s = "http://" + s
        placeholder_added = True

    try:
        p = urlparse(s)
    except Exception:
        return None, None, "parse_error"

    if not p.hostname:
        return None, None, "no_host"

    try:
        port = p.port
    except ValueError:
        return None, None, "bad_port"

    scheme = (p.scheme or "").lower()
    host = p.hostname.lower()
    netloc = host + (f":{port}" if port is not None else "")
    path = p.path or ""
    query = p.query or ""
    frag = p.fragment or ""

    url_norm = (
        f"{scheme}://{netloc}{path}"
        + (f"?{query}" if query else "")
        + (f"#{frag}" if frag else "")
    )

    meta = {
        "had_scheme_raw": had_scheme,
        "placeholder_added": placeholder_added,
        "scheme_norm": scheme,
        "host_norm": host,
        "port": port,
        "has_path": bool(path and path != "/"),
        "has_query": bool(query),
        "has_fragment": bool(frag),
        "url_len": len(url_norm),
    }
    return url_norm, meta, None

def clean(df_in: pd.DataFrame, label: int, source: str) -> tuple:
    """ Normalise all URLs in df_in and return the cleaned DataFrame, rejection counts,
    and pre-dedup size"""
    reject = Counter()
    rows = []

    for raw in df_in["url_raw"]:
        url_norm, meta, reason = normalise_url(raw)
        if reason:
            reject[reason] += 1
            continue

        ext = tldextract.extract(url_norm)
        etld1 = f"{ext.domain}.{ext.suffix}" if ext.suffix else None

        rows.append({
            "url_raw": raw,
            "url_norm": url_norm,
            "label": label,
            "source": source,
            "scheme_norm": meta["scheme_norm"],
            "host_norm": meta["host_norm"],
            "port": meta["port"],
            "etld1": etld1,
            "suffix": ext.suffix if ext.suffix else None,
            "has_path": meta["has_path"],
            "has_query": meta["has_query"],
            "has_fragment": meta["has_fragment"],
            "url_len": meta["url_len"],
            "had_scheme_raw": meta["had_scheme_raw"],
            "placeholder_added": meta["placeholder_added"],
        })

    df = pd.DataFrame(rows)
    pre_dedup = len(df)
    df = df.drop_duplicates(subset=["url_norm"]).reset_index(drop=True)
    return df, reject, pre_dedup

def build_summary_block(name: str, df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                        reject: Counter, pre_dedup: int) -> str:
    """ Format a pre-source statistics block for summary."""
    n_raw = len(df_raw)
    n_clean = len(df_clean)

    had_scheme = int(df_clean["had_scheme_raw"].sum()) if n_clean else 0
    placeholder = int(df_clean["placeholder_added"].sum()) if n_clean else 0

    ports = df_clean["port"].dropna().astype(int) if n_clean else pd.Series({}, dtype=int)
    port_counts = Counter(ports.tolist())
    non_default = {p: c for p, c in port_counts.items() if p not in (80,443)}
    non_default_sorted = sorted(non_default.items(), key=lambda kv: (-kv[1], kv[0]))

    has_path = int(df_clean["has_path"].sum()) if n_clean else 0
    has_query = int(df_clean["has_query"].sum()) if n_clean else 0
    has_frag = int(df_clean["has_fragment"].sum()) if n_clean else 0

    lens = df_clean["url_len"] if n_clean else pd.Series([], dtype=int)
    uniq_etld1 = int(df_clean["etld1"].nunique(dropna=True)) if n_clean else 0
    etld1_counts = df_clean["etld1"].value_counts() if n_clean else pd.Series([], dtype=int)
    max_per_dom = int(etld1_counts.max()) if not etld1_counts.empty else 0
    avg_per_dom = (n_clean / uniq_etld1) if uniq_etld1 else None
    suffix_top = df_clean["suffix"].value_counts().head(10) if n_clean else pd.Series([], dtype=int)

    out = [f"[{name}]",
           f"Raw rows: {n_raw}",
           f"Clean rows (pre-dedup): {pre_dedup}",
           f"Dedup removed: {pre_dedup - n_clean}",
           f"Clean unique URLs: {n_clean}", "",
           "Rejected (by reason):",
           f"  Total rejected: {sum(reject.values())}"]
    for r, c in sorted(reject.items(), key=lambda kv: (-kv[1], kv[0])):
        out.append(f" - {r}: {c}")
    out += ["", "Scheme:",
            f"  Raw has scheme: {had_scheme} / {n_clean} ({pct(had_scheme, n_clean)})",
            f"  Placeholder added: {placeholder} / {n_clean} ({pct(placeholder, n_clean)})",
            "", "Ports:",
            f"  URLs with explicit port: {len(ports)} / {n_clean} ({pct(len(ports), n_clean)})",
            f"  Default: 80: {port_counts.get(80, 0)}",
            f"  Default: 443: {port_counts.get(443, 0)}",
            "   Non-default ports:"]
    if non_default_sorted:
        for p, c in non_default_sorted:
            out.append(f" - {p}: {c} ({pct(c, n_clean)})")
    else:
        out.append(f" (none)")
    out += ["", "Structure:",
            f"  Has path: {has_path} / ({n_clean} {pct(has_path, n_clean)})",
            f"  Has query: {has_query} / ({n_clean} {pct(has_query, n_clean)})",
            f"  Has fragments: {has_frag} / ({n_clean} {pct(has_frag, n_clean)})"]
    if n_clean:
        out += ["", "URL length (url_norm):",
                f"  min={int(lens.min())} median={float(lens.median()):.1f} "
                f"mean={float(lens.mean()):.1f} max={int(lens.max())}",
                f"  >500 chars: {int((lens > 500).sum())} / {n_clean}",
                f"  >1000 chars: {int((lens > 1000).sum())} / {n_clean}"]
    out += ["", "Domains:",
        f"  Unique eTLD+1: {uniq_etld1}",
        f"  Avg URLs per eTLD+1: {avg_per_dom}",
        f"  Max URLs for a single eTLD+1: {max_per_dom}",
        "  Top suffixes:"]
    if not suffix_top.empty:
        total = n_clean
        for suf, c in suffix_top.items():
            out.append(f" - {suf}: {int(c)} ({pct(int(c), total)})")
        other = total - int(suffix_top.sum())
        if other > 0:
            out.append(f" - {other}: {other} ({pct(other, n_clean)})")
    else:
        out.append("    (none)")
    out.append("\n" + "-" * 50 + "\n")
    return "\n".join(out)

def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    for path in (PHISHTANK_PATH, TRANCO_PATH, KAGGLE_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw data file: {path}\n"
                f"Place source CSVs in: {RAW_DIR}"
            )

    pt = pd.read_csv(PHISHTANK_PATH)
    pt = pt[(pt["verified"] == "yes") & (pt["online"] == "yes")]
    pt_df = pt[["url"]].rename(columns={"url": "url_raw"})

    tr = pd.read_csv(TRANCO_PATH, header=None)
    domains = (tr.iloc[:, 0] if tr.shape[1] == 1 else tr.iloc[:, 1]).dropna()
    tr_df = pd.DataFrame({"url_raw": domains.astype(str)})

    kg = pd.read_csv(KAGGLE_PATH)
    kg = kg[kg["type"] == "benign"]
    kg_df = kg[["url"]].rename(columns={"url": "url_raw"})

    pt_clean, pt_rej, pt_pre = clean(pt_df, label=1, source="phishtank")
    tr_clean, tr_rej, tr_pre = clean(tr_df, label=0, source="tranco")
    kg_clean, kg_rej, kg_pre = clean(kg_df, label=0, source="kaggle")

    pt_clean.to_csv(CLEAN_DIR / "phishtank_clean.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    tr_clean.to_csv(CLEAN_DIR / "tranco_clean.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    kg_clean.to_csv(CLEAN_DIR / "kaggle_benign_clean.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    # Any benign domain that also appears in the phishing set is removed to prevent the model learning to distinguish sources rather than URL characteristics.
    phishing_domains = set(pt_clean["etld1"].dropna())
    kg_no = kg_clean[~kg_clean["etld1"].isin(phishing_domains)].copy()
    tr_no = tr_clean[~tr_clean["etld1"].isin(phishing_domains)].copy()

    # Tranco domains are structural outliers (domain-only, no path) relative to kaggle URLs (which have full paths).
    # Capping Tranco at 25@ of benign prevents the benign set efrom being structurally dominated by a single source pattern.
    n_phish = len(pt_clean)
    n_target_benign = n_phish
    n_tranco_cap = int(0.25 * n_target_benign)
    n_tranco = min(n_tranco_cap, len(tr_no))
    n_kaggle = n_target_benign - n_tranco

    if n_kaggle > len(kg_no):
        shortfall = n_kaggle - len(kg_no)
        n_kaggle = len(kg_no)
        n_tranco = min(len(tr_no), n_tranco + shortfall)
        if (n_kaggle + n_tranco) < n_target_benign:
            raise RuntimeError("Insufficient benign samples after cleaning/overlap removal.")

    kg_sample = kg_no.sample(n=n_kaggle, random_state=SEED) if n_kaggle else kg_no.iloc[:0]
    tr_sample = tr_no.sample(n=n_tranco, random_state=SEED) if n_tranco else tr_no.iloc[:0]

    benign_sample = pd.concat(
        [df for df in (kg_sample, tr_sample) if not df.empty],
        ignore_index=True
    )

    final_df = (
        pd.concat([pt_clean, benign_sample], ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    final_df.to_csv(FINAL_OUTPUT, index=False, quoting=csv.QUOTE_MINIMAL)

    benign_pool = pd.concat(
        [df for df in (kg_no, tr_no) if not df.empty],
        ignore_index=True
    )

    lines = [
        "=== Tower A Preprocessing Summary ===",
        f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Seed: {SEED}", "",
        build_summary_block("phishtank", pt_df, pt_clean, pt_rej, pt_pre),
        build_summary_block("kaggle_benign", kg_df, kg_clean, kg_rej, kg_pre),
        build_summary_block("tranco", tr_df, tr_clean, tr_rej, tr_pre),
        "[leakage_mitigation]",
        f"Phishing unique eTLD+1: {len(phishing_domains)}",
        f"Kaggle benign removed (overlap): {len(kg_clean) - len(kg_no)}",
        f"Tranco removed (overlap): {len(tr_clean) - len(tr_no)}", "",
        "[benign_pool_pre_balance]",
        f"Total benign pool: {len(benign_pool)}",
        f"Kaggle benign: {len(kg_no)} ({pct(len(kg_no), len(benign_pool))})",
        f"Tranco benign: {len(tr_no)} ({pct(len(tr_no), len(benign_pool))})",
        "\n" + "-" * 50 + "\n",
        "[final_dataset]",
        f"Total rows: {len(final_df)}",
        f"Phishing: {int((final_df.label == 1).sum())}",
        f"Benign:   {int((final_df.label == 0).sum())}",
        "Benign source mix:",
    ]
    be = int((final_df.label == 0).sum())
    for src, c in final_df[final_df.label == 0]["source"].value_counts().items():
        lines.append(f"  - {src}: {int(c)} ({pct(int(c), be)})")

    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("=== Preprocessing complete ===")
    print(f"  Final dataset:  {FINAL_OUTPUT}  ({len(final_df):,} rows)")
    print(f"  Cleaned files:  {CLEAN_DIR}")
    print(f"  Summary:        {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
