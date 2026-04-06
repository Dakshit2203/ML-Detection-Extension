"""
Tower A - Phase 1: Dataset Audit

Validate towerA_dataset.csv after preprocessing. Checks schema, label integrity. duplicate URLs,
conflicting eTLD+1 labels, domain concentration, and structural URL statistics. The shortcut-risk heuristics flag
features that may differ so strongly between classes that they could false performance - for example, a large gap in
quary presence rates could mean the model is partially learning source structure rather than phishing indicators.

Writes: 01_preprocessing/outputs/audit_report.txt
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "outputs" / "towerA_dataset.csv"
OUTPUT_REPORT = BASE_DIR / "outputs" / "audit_report.txt"

IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
B64_RE = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
HEX_IN_URL_RE = re.compile(r"%[0-9A-Fa-f]{2}")

def pct(n: int, d: int) -> str:
    return "0.00%" if d == 0 else f"{100.0 * n / d:.2f}%"

def parse_parts(url: str):
    try:
        p = urlparse(url)
        return p.hostname or "", (p.path or ""), (p.query or ""), (p.fragment or ""), p.port
    except Exception:
        return None

def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}\n"
            f"Run preprocess.py first."
        )

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S UTC")
    lines = [
        "=== Tower A Final Dataset Audit (read-only) ===",
        f"Timestamp: {ts}",
        f"Input: {INPUT_PATH}",
        f"Output: {OUTPUT_REPORT}",
        "",
    ]

    df = pd.read_csv(INPUT_PATH)
    n = len(df)
    expected = ["url_norm", "label", "source", "etld1"]

    lines += ["[schema]", f"Columns present: {len(df.columns)}"]
    missing_cols = [c for c in expected if c not in df.columns]
    lines.append(
        f"WARNING: Missing expected columns: {missing_cols}" if missing_cols
        else "OK: All expected core columns present."
    )
    lines.append("")

    lines += ["[size]", f"Total rows: {n}", ""]

    lines.append("[missing_values]")
    for c in expected:
        if c in df.columns:
            m = int(df[c].isna().sum())
            lines.append(f"  {c}: {m} missing ({pct(m, n)})")
    lines.append("")

    lines.append("[labels]")
    if "label" not in df.columns:
        lines += [" ERROR: No label column.", ""]
    else:
        for k, v in df["label"].value_counts(dropna=False).items():
            lines.append(f"  label={k}: {int(v)} ({pct(int(v), n)})")
        bad = sorted(set(df["label"].dropna().unique()) - {0, 1})
        lines.append(
            f" WARNING: Non-binary labels found: {bad}" if bad
            else "  OK: Labels are binary (0/1)."
        )
        lines.append("")

    lines.append("[source_integrity]")
    if {"source", "label"}.issubset(df.columns):
        cross = pd.crosstab(df["source"], df["label"])
        lines += [" Source x label counts:", cross.to_string(), ""]
        if "phishtank" in cross.index and 0 in cross.columns and int(cross.loc["phishtank", 0]) > 0:
            lines.append(" WARNING: Found benign-labelled rows from phishtank.")
        if "kaggle" in cross.index and 1 in cross.columns and int(cross.loc["kaggle", 1]) > 0:
            lines.append(" WARNING: Found phishing-labelled rows from kaggle.")
        if "tranco" in cross.index and 1 in cross.columns and int(cross.loc["tranco", 1]) > 0:
            lines.append(" WARNING: Found phishing-labelled rows from tranco.")
        lines.append(" OK: Source/label mapping consistent if no warnings above.")
    else:
        lines.append(" WARNING: Cannot check - missing source or label column.")
    lines.append("")

    lines.append("[duplicates]")
    if "url_norm" in df.columns:
        dup = int(df.duplicated(subset=["url_norm"]).sum())
        lines.append(f" Duplicate url_norm rows: {dup} ({pct(dup, n)})")
    else:
        lines.append(" WARNING: No url_norm column for duplicate check.")

    if {"etld1", "label"}.issubset(df.columns):
        nunq = df.dropna(subset=["etld1"]).groupby("etld1")["label"].nunique()
        conflicts = int((nunq > 1).sum())
        lines.append(f" eTLD+1 with conflicting labels: {conflicts}")
        if conflicts:
            top_conf = df[df["etld1"].isin(nunq[nunq > 1].index)]["etld1"].value_counts().head(10)
            lines += [" Top conflicting eTLD+1 (by URL count):", top_conf.to_string()]
    else:
        lines.append(" WARNING: Cannot check eTLD+1 conflicts - missing etld1/label.")
    lines.append("")

    lines.append("[domain_concentration]")
    if {"etld1", "label"}.issubset(df.columns):
        for lab in (0, 1):
            sub = df[df["label"] == lab].dropna(subset=["etld1"])
            total = len(sub)
            uniq = int(sub["etld1"].nunique())
            avg = (total / uniq) if uniq else None
            lines.append(f"  label={lab}: urls={total}, unique_etld1={uniq}, avg_per_etld1={avg}")
            if total:
                vc = sub["etld1"].value_counts()
                top10 = vc.head(10)
                lines.append(f"  label={lab}: max_urls_single_etld1={int(vc.max())}")
                lines.append(f"    top10_total={int(top10.sum())} ({pct(int(top10.sum()), total)})")
    else:
        lines.append("  WARNING: Cannot compute - missing etld1/label.")
    lines.append("")

    lines.append("[structural_distributions]")
    if {"url_norm", "label"}.issubset(df.columns):
        parsed = df["url_norm"].astype(str).map(parse_parts)
        bad_parse = int(parsed.isna().sum())
        lines.append(f"  Parse failures (urlparse): {bad_parse} ({pct(bad_parse, n)})")

        parts = pd.DataFrame(parsed.tolist(), columns=["_host", "_path", "_query", "_frag", "_port"])
        df2 = pd.concat([df, parts], axis=1)
        s_url = df2["url_norm"].astype(str)
        s_host = df2["_host"].astype(str)
        s_path = df2["_path"].astype(str)
        s_query = df2["_query"].astype(str)

        df2["_url_len"] = s_url.str.len()
        df2["_has_query"] = s_query.str.len().gt(0)
        df2["_has_frag"] = df2["_frag"].astype(str).str.len().gt(0)
        df2["_num_slashes"] = s_url.str.count("/")
        df2["_has_ip_host"] = s_host.map(lambda h: bool(IP_RE.match(h)))
        df2["_has_punycode"] = s_host.str.contains("xn--", na=False)
        df2["_pct_enc"] = s_url.map(lambda x: len(HEX_IN_URL_RE.findall(x)))
        df2["_b64_like"] = s_url.map(lambda x: bool(B64_RE.search(x)))
        df2["_upper_in_pq"] = (s_path + "?" + s_query).map(
            lambda x: any(ch.isupper() for ch in x)
        )

        def stats(col: str):
            out = []
            for lab in (0, 1):
                sub = df2[df2["label"] == lab]
                if sub.empty:
                    out.append((lab, "no data"))
                elif sub[col].dtype == bool or sub[col].dtype == object:
                    c = int(sub[col].sum())
                    out.append((lab, f"{c} / {len(sub)} ({pct(c, len(sub))})"))
                else:
                    s = sub[col]
                    out.append((lab, f"min={s.min()} median={s.median():.1f} "
                                     f"mean={s.mean():.1f} max={s.max()}"))
            return out

        for label_name, col in [
            ("URL length", "_url_len"),
            ("Has query", "_has_query"),
            ("Has fragment", "_has_frag"),
            ("Slash count", "_num_slashes"),
            ("IP host", "_has_ip_host"),
            ("Punycode host", "_has_punycode"),
            ("Percent-encoding", "_pct_enc"),
            ("Base64-like", "_b64_like"),
            ("Uppercase path/query", "_upper_in_pq"),
        ]:
            lines.append(f"  {label_name}:")
            for lab, txt in stats(col):
                lines.append(f"    label={lab}: {txt}")

        # Flag structural differences that could indicate the model is partly learning source-specific patterns
        # rather than genuine phishing indicators
        lines += ["", "  Shortcut-risk flags:"]
        q0 = df2[df2["label"] == 0]["_has_query"].mean()
        q1 = df2[df2["label"] == 1]["_has_query"].mean()
        lines.append(
            f" WARNING: Query presence differs >70pp (benign={q0:.2f}, phish={q1:.2f})"
            if abs(q1 - q0) > 0.70
            else f" OK: Query presence - benign={q0:.2f}, phish={q1:.2f}"
        )
        m0 = df2[df2["label"] == 0]["_num_slashes"].median()
        m1 = df2[df2["label"] == 1]["_num_slashes"].median()
        lines.append(
            f" WARNING: Median slash count large gap (benign={m0}, phish={m1})"
            if abs(m1 - m0) > 8
            else f" OK: Median slash count - benign={m0}, phish={m1}"
        )
    else:
        lines.append(" WARNING: Cannot compute - missing url_norm/label.")

    lines += ["", "=== Audit complete ==="]

    OUTPUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Audit complete. Report written to: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()