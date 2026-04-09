"""
Tower B - Data Processing - 01_audit.py

Audits the towerB_metadata_master.csv dataset.

Purpose
This script reads the metadata CSV that was produced during the Tower B metadata collection phase and produces a
concise audit report confirming:
  - Row count and label balance
  - Per-feature missingness rates
  - Probe success rates (dns_ok, tls_ok, http_ok) by label
  - The two preprocessing leakage artefacts (had_scheme_raw, placeholder_added) are still present and still
  show near-perfect label correlation

The audit report (outputs/01_audit_report.txt) serves as the documented justification for the feature exclusions
applied in 02_feature_spec.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
# The metadata CSV was produced during the Tower B metadata collection phase.
DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_CSV = DATA_DIR / "towerB_metadata_master.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_TXT = OUTPUT_DIR / "01_audit_report.txt"

# Features known from the earlier diagnostic to be problematic.
# These are documented here for traceability - the decision to exclude them
# is justified by the correlation values printed in the audit report.
KNOWN_LEAKAGE_FEATURES = ["had_scheme_raw", "placeholder_added"]
KNOWN_MISSING_FEATURES = ["port"] # 99.97% missing in the dataset

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        print("Expected location: Tower B - Data Processing/data/processed/towerB_metadata_master.csv")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)

    lines = []
    lines.append("=" * 70)
    lines.append("Tower B - Data Processing - Audit Report")
    lines.append(f"Input: {INPUT_CSV}")
    lines.append("=" * 70)

    # Dataset shape
    lines.append(f"\nRows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns)}")

    if "label" not in df.columns:
        print("ERROR: 'label' column not found in dataset.")
        sys.exit(1)

    # Label balance
    label_counts = df["label"].value_counts().sort_index()
    lines.append("\nLabel distribution:")
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        lines.append(f" label={label}: {count:,} ({pct:.1f}%)")

    # Probe success rates
    # The three binary success flags indicate how many URLs were successfully probed. Failures are expected - many
    # phishing domains are short-lived and become unreachable before probing completes.
    lines.append("\nProbe success rates (overall):")
    for flag in ["dns_ok", "tls_ok", "http_ok"]:
        if flag in df.columns:
            rate = df[flag].mean()
            lines.append(f" {flag}: {rate:.4f} ({rate*100:.1f}%)")

    lines.append("\nProbe success rates by label:")
    for flag in ["dns_ok", "tls_ok", "http_ok"]:
        if flag in df.columns:
            by_label = df.groupby("label")[flag].mean()
            lines.append(f" {flag}:")
            for lbl, val in by_label.items():
                lines.append(f" label={lbl}: {val:.4f}")

    # Missingness
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "label" in numeric_cols:
        numeric_cols.remove("label")

    lines.append(f"\nNumeric feature columns: {len(numeric_cols)}")
    lines.append("\nMissingness by feature (fraction missing):")
    miss = df[numeric_cols].isnull().mean().sort_values(ascending=False)
    for col, frac in miss.items():
        flag = " <- 100% missing (EXCLUDE)" if frac == 1.0 else (
               " <- partial missing" if frac > 0 else ""
        )
        lines.append(f" {col}: {frac:.4f}{flag}")
    
    # ── Leakage check ──────────────────────────────────────────────────────────
    # had_scheme_raw and placeholder_added are preprocessing artefacts that correlate perfectly with the label because
    # they reflect the data source (PhishTank URLs have schemes; Tranco domains do not). Including either feature would
    # let the model learn the data source rather than the phishing signal. Both must be excluded from the feature spec.
    lines.append("\nLeakage check (Pearson r with label):")
    for col in KNOWN_LEAKAGE_FEATURES:
        if col in df.columns:
            r = df[col].corr(df["label"])
            flag = " <- LEAKAGE ARTEFACT - EXCLUDE" if abs(r) > 0.5 else ""
            lines.append(f" {col}: r={r:.4f}{flag}")
        else:
            lines.append(f" {col}: NOT FOUND in dataset")

    # ── Correlation table for all numeric features ─────────────────────────────
    lines.append("\nAll numeric features - |Pearson r| with label (sorted):")
    corrs = df[numeric_cols].corrwith(df["label"]).abs().sort_values(ascending=False)
    for col, r_abs in corrs.items():
        if df[col].nunique() <= 1:
            note = " <- EXCLUDE (zero variance — constant column)"
        elif col in KNOWN_LEAKAGE_FEATURES:
            note = " <- EXCLUDE (leakage artefact)"
        elif col in KNOWN_MISSING_FEATURES:
            note = " <- EXCLUDE (99.97% missing)"
        else:
            note = ""
        lines.append(f"  {col}: |r|={r_abs:.4f}{note}")

    report = "\n".join(lines)
    print(report)
    OUTPUT_TXT.write_text(report, encoding="utf-8")
    print(f"\nReport written: {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
