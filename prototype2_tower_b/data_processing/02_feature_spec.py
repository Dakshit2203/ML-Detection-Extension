"""
Tower B - Data Processing - 02_feature_spec.py

Produces feature_spec_B.json: the locked feature contract for Tower B.

Purpose
This script reads the metadata CSV, applies the exclusion decisions documented in the 01_audit.py report, and writes a
JSON file listing the 18 numeric features that will be used for training and inference.

The feature spec is the single point of truth that connects three parts of the system:
  - 03_split.py uses it to validate that the split CSVs contain the right columns.
  - 04_train_eval.py uses it to build the feature matrix for training.
  - The Prototype 2 backend (backend/towerB.py) uses it to build the inference feature vector in the same column order.

Any mismatch between the columns present at training time and at inference time would cause silent score corruption.
The feature spec prevents this by acting as a shared contract that all components read from the same file.

Exclusion criteria
Features are excluded if they meet any of the following conditions:

  1. Preprocessing leakage artefact: Pearson |r| with label > 0.50.
     Identified: had_scheme_raw (r=0.936), placeholder_added (r=-0.936).
     These correlate with the data source (PhishTank vs Tranco) rather than the phishing signal itself.

  2. 99.97% missing: port.
     A feature that is majority missing carries no information and would force the model to learn a meaningless split.

  3. Non-numeric dtype: string identifiers and free-text fields
     (tls_issuer_cn, http_final_url, http_final_host, http_final_host, http_scheme_used, collected_at, etc.). These
     cannot be passed directly to a gradient boosting model without text vectorisation, which is out of scope for Tower B.

  4. Target and identifier columns: label, url_raw, url_norm, host_norm, scheme_norm, etld1, suffix, source.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
DATA_DIR = Path(__file__).resolve().parent / "data"
INPUT_CSV = DATA_DIR / "towerB_metadata_master.csv"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_SPEC_JSON = OUTPUT_DIR / "feature_spec_B.json"
OUTPUT_SUMMARY = OUTPUT_DIR / "02_feature_spec_summary.txt"

# The feature spec is also copied to the artifacts/towerB/ directory so the backend can find it without needing to
# reference the data_processing path.
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "towerB"
ARTIFACT_SPEC = ARTIFACTS_DIR / "feature_spec_B.json"

# Exclusion sets
# Leakage artifacts identified in 01_audit.py. Both correlate with the data source label (PhishTank = scheme present;
# Tranco = no scheme), not with the phishing signal. Including either would allow the model to trivially learn the
# source instead of the security signal.
LEAKAGE_EXCLUDE = {"had_scheme_raw", "placeholder_added"}

# 99.97% missing - carries no information.
MISSING_EXCLUDE = {"port"}

ZERO_VARIANCE_EXCLUDE = {"dns_has_ipv6", "tls_issuer_is_letsencrypt"}

# String/identifier columns that cannot be used as numeric features.
# Includes the target variable, URL identifiers, and raw string metadata.
STRING_EXCLUDE = {
    "label",
    "url_raw", "url_norm", "host_norm", "scheme_norm",
    "etld1", "suffix", "source",
    "tls_issuer_cn", "tls_not_before", "tls_not_after",
    "http_final_url", "http_final_host", "http_final_host",
    "http_scheme_used", "collected_at",
}

ALL_EXCLUDE = LEAKAGE_EXCLUDE | MISSING_EXCLUDE | ZERO_VARIANCE_EXCLUDE | STRING_EXCLUDE

# Only features from these signal groups are included. This keeps the feature set tightly scoped to the three probing
# signals Tower B is designed to use.
ALLOWED_PREFIXES = ("dns_", "tls_", "http_")

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {INPUT_CSV}")
    if not INPUT_CSV.exists():
        print(f"ERROR: Input file not found: {INPUT_CSV}")
        sys.exit(1)

    # Read a small sample to determine dtypes - do not need all 92,666 rows to identify column types. The full dataset
    # is used for missingness checks.
    df_sample = pd.read_csv(INPUT_CSV, nrows=5000)
    df_full = pd.read_csv(INPUT_CSV, usecols=lambda c: c not in STRING_EXCLUDE)

    # ── Candidate selection ────────────────────────────────────────────────────
    # Start from all columns in the dataset, then apply exclusions.
    included = []
    excluded = []

    for col in df_sample.columns:
        reason = None

        if col in LEAKAGE_EXCLUDE:
            reason = "preprocessing leakage artefact (|r| > 0.50 with label)"
        elif col in MISSING_EXCLUDE:
            reason = "99.97% missing across all rows"
        elif col in ZERO_VARIANCE_EXCLUDE:
            reason = "zero variance (constant column - confirmed by audit: |r|=nan)"
        elif col in STRING_EXCLUDE:
            reason = "non-numeric / identifier column"
        elif not pd.api.types.is_numeric_dtype(df_sample[col]):
            reason = "non-numeric dtype"
        elif not col.startswith(ALLOWED_PREFIXES):
            reason = "not in allowed prefix group (dns_*, tls_*, http_*)"
        else:
            # Check missingness - exclude if more than 50% missing in full dataset
            if col in df_full.columns:
                miss_rate = df_full[col].isnull().mean()
                if miss_rate > 0.50:
                    reason = f"excessive missingness ({miss_rate:.1%})"

        if reason:
            excluded.append((col, reason))
        else:
            included.append(col)

    # Sort for deterministic ordering - critical for reproducibility.
    # The model will be trained with features in this order; the backend must use the same order at inference time.
    included = sorted(set(included))

    # Write feature spec
    spec = {
        "name": "TowerB_metadata_v1",
        "version": "1.0",
        "target": "label",
        "n_features": len(included),
        "features": included,
        "exclusions": {
            "leakage": sorted(LEAKAGE_EXCLUDE),
            "missing": sorted(MISSING_EXCLUDE),
            "non_numeric": sorted(STRING_EXCLUDE),
        },
    }

    spec_json = json.dumps(spec, indent=2)
    OUTPUT_SPEC_JSON.write_text(spec_json, encoding="utf-8")
    ARTIFACT_SPEC.write_text(spec_json, encoding="utf-8")

    # Write summary
    lines = []
    lines.append("=" * 70)
    lines.append("Tower B - Feature Specification Summary")
    lines.append("=" * 70)
    lines.append(f"\nIncluded features: {len(included)}")
    for f in included:
        lines.append(f" {f}")

    lines.append(f"\nExcluded features: {len(excluded)}")
    for col, reason in sorted(excluded):
        lines.append(f" {col}: {reason}")

    summary = "\n".join(lines)
    print(summary)
    OUTPUT_SUMMARY.write_text(summary, encoding="utf-8")

    print(f"\nFeature spec written:")
    print(f" {OUTPUT_SPEC_JSON}")
    print(f" {ARTIFACT_SPEC} (backend copy)")
    print(f"\nFeatures included: {len(included)}")
    print(f"Features excluded: {len(excluded)}")

if __name__ == "__main__":
    main()
