"""
Tower A - Phase 6: Extract Features from External Datasets

Applies the same 35-feature extraction pipeline to the external validation datasets built in step 01. Using the shared
feature modules from prototype1_tower_a/features/ guarantees that extraction is byte-for-byte identical to what was
used during training, so any score differences are due to URL content rather than implementation drift.

Inputs:  06_external_validation/outputs/<latest_run>/datasets/
Outputs: 06_external_validation/outputs/<latest_run>/features/
  features_aggressive.csv
  features_neutral.csv
  feature_extraction_manifest.json

Run AFTER 01_build_dataset.py.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract

BASE = Path(__file__).resolve().parent
OUT_ROOT = BASE / "outputs"
P1_ROOT = BASE.parents[1]
FEATURES_DIR = P1_ROOT / "features"

if str(FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(FEATURES_DIR))

from group_a_structure import extract_group_a
from group_b_chars import extract_group_b
from group_c_entropy import extract_group_c
from group_d_tokens import extract_group_d
from group_e_brand import extract_group_e

SCENARIOS = ["aggressive", "neutral"]

def latest_run_dir() -> Path:
    runs = sorted(OUT_ROOT.glob("external_validation__*"), key=lambda p: p.name)
    if not runs:
        raise FileNotFoundError(
            f"No external_validation__* directories found under {OUT_ROOT}.\n"
            f"Run 01_build_dataset.py first."
        )
    return runs[-1]

def ensure_etld1(df: pd.DataFrame) -> pd.DataFrame:
    """Add etld1 column if absent, or normalise NaN representations if present."""
    if "etld1" in df.columns:
        df["etld1"] = df["etld1"].astype(str).replace({"nan": np.nan, "": np.nan})
        return df

    def _etld1(u: str):
        ex = tldextract.extract(u)
        if ex.domain and ex.suffix:
            return f"{ex.domain}.{ex.suffix}".lower()
        return np.nan

    df["etld1"] = df["url_norm"].astype(str).map(_etld1)
    return df

def load_dataset(run_dir: Path, scenario: str) -> pd.DataFrame:
    p = run_dir / "datasets" / f"external_dataset_{scenario}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing dataset: {p}\nRun 01_build_dataset.py first.")
    df = pd.read_csv(p)
    for col in ("url_norm", "label"):
        if col not in df.columns:
            raise RuntimeError(f"{p} missing required column: {col}")
    df["url_norm"] = df["url_norm"].astype(str)
    df["label"] = df["label"].astype(int)
    return ensure_etld1(df)

def main() -> None:
    run_dir = latest_run_dir()
    feat_dir = run_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run dir: {run_dir}")
    print(f"Feature modules: {FEATURES_DIR}")

    manifest = {
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "features_dir": str(FEATURES_DIR),
        "outputs": {},
    }

    for scenario in SCENARIOS:
        print(f"\nExtracting features: {scenario}")
        df = load_dataset(run_dir, scenario)
        print(f" Rows: {len(df):,}")

        A = extract_group_a(df, url_col="url_norm", etld1_col="etld1")
        B = extract_group_b(df, url_col="url_norm")
        C = extract_group_c(df, url_col="url_norm")
        D = extract_group_d(df, url_col="url_norm")
        E = extract_group_e(df, url_col="url_norm", etld1_col="etld1")

        feats = pd.concat([A, B, C, D, E], axis=1)
        out_path = feat_dir / f"features_{scenario}.csv"
        feats.to_csv(out_path, index=False)
        print(f" Features: {feats.shape[1]} columns -> {out_path}")

        manifest["outputs"][scenario] = {
            "rows": int(len(df)),
            "feature_cols": int(feats.shape[1]),
            "features_csv": str(out_path),
        }

    manifest_path = run_dir / "feature_extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
