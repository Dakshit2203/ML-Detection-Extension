"""
Tower A - Phase 6: Build External Validation Datasets

Constructs two phishing/benign datasets from data sources not seen during training to test whether the model
generalises beyond its training distribution.

The motivation is that Phase 5 test metrics are measured on data from the same distribution as training. Kaggle benign
URLs have full URL paths, while real-world benign traffic to top domains is often just the bare domain or a simple path.
Evaluating the model on Tranco top-domain URLs with synthetic paths tests whether stored thresholds transfer to this
different structure - and they do not (FPR=1.0), which motivated adaptive thresholding.

Two scenarios vary the structural similarity of benign URLs to phishing:
  aggressive  benign URLs with login/signin/account paths (high false-positive risk)
  neutral     benign URLs with generic about/contact/news paths

Overlap with the training dataset is removed at both URL and domain level to ensure this is a genuine held-out evaluation.

Raw files go in 06_external_validation/raw/ (gitignored):
  phishtank.csv   fresh phishing URLs
  tranco.csv      Tranco top-N domain list for benign construction

Outputs: 06_external_validation/outputs/<timestamp>/datasets/
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tldextract

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
OUT_ROOT = BASE / "outputs"

PHISHTANK_FILE = RAW / "phishtank.csv"
TRANCO_FILE = RAW / "tranco.csv"
ORIGINAL_DATASET = BASE.parent / "01_preprocessing" / "outputs" / "towerA_dataset.csv"

TOP_N_DOMAINS = 10_000
AGGRESSIVE_PATHS = ["/", "/login", "/signin", "/account", "/verify", "/auth"]
NEUTRAL_PATHS = ["/", "/about", "/contact", "/products", "/news"]

def run_id(prefix: str = "external_validation") -> str:
    return f"{prefix}__{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}"

def get_etld1(url: str):
    ex = tldextract.extract(str(url))
    if ex.domain and ex.suffix:
        return f"{ex.domain}.{ex.suffix}".lower()
    return None

def detect_url_column(df: pd.DataFrame) -> str:
    for c in ("url_norm", "url", "URL", "Url", "raw_url"):
        if c in df.columns:
            return c
    raise RuntimeError(f"Cannot detect URL column. Columns found: {list(df.columns)}")

def load_original_domains() -> tuple[set[str], set[str]]:
    """Load training URLs and eTLD+1 domains used for overlap removal."""
    if not ORIGINAL_DATASET.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {ORIGINAL_DATASET}\n"
            f"Run 01_preprocessing/preprocess.py first."
        )
    df = pd.read_csv(ORIGINAL_DATASET)
    url_col = detect_url_column(df)
    urls = df[url_col].astype(str).tolist()
    domains = {d for d in (get_etld1(u) for u in urls) if d is not None}
    return set(urls), domains

def load_tranco_domains(path: Path, top_n: int) -> pd.Series:
    tr = pd.read_csv(path, header=None)
    if tr.shape[1] < 1:
        raise RuntimeError(f"Unexpected Tranco format: {path}")
    return tr.iloc[:, -1].astype(str).head(top_n)

def build_scenario(paths: list[str], label_name: str) -> pd.DataFrame:
    """Combine fresh PhishTank URLs with Tranco-derived benign URLs for one scenario."""
    phish = pd.read_csv(PHISHTANK_FILE)
    if "url" not in phish.columns:
        raise RuntimeError(f"{PHISHTANK_FILE} missing 'url' column")
    phish_df = pd.DataFrame({
        "url_norm": phish["url"].astype(str),
        "label": 1,
        "source": "phish_external",
    })
    phish_df["etld1"] = phish_df["url_norm"].map(get_etld1)

    # Construct benign URLs by appending scenario-specific paths to each Tranco domain.
    # This deliberately varies URL structure to test threshold sensitivity to path presence.
    domains = load_tranco_domains(TRANCO_FILE, TOP_N_DOMAINS)
    benign_df = pd.DataFrame({
        "url_norm": [f"https://{d}{p}" for d in domains for p in paths],
        "label": 0,
        "source": f"benign_{label_name}",
    })
    benign_df["etld1"] = benign_df["url_norm"].map(get_etld1)

    df = pd.concat(
        [phish_df[["url_norm","label","etld1","source"]], benign_df],
        ignore_index=True
    )
    df.drop_duplicates(subset=["url_norm"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def remove_overlap(df: pd.DataFrame,
                   existing_urls: set[str],
                   existing_domains: set[str]) -> pd.DataFrame:
    """Remove any URL or domain that appeared in the training dataset."""
    before = len(df)
    df = df[~df["url_norm"].isin(existing_urls)]
    df = df[~df["etld1"].isin(existing_domains)]
    df = df.reset_index(drop=True)
    print(f" Overlap removal: {before:,} -> {len(df):,} ({before - len(df)} removed)")
    return df

@dataclass
class Manifest:
    created_at: str; run_dir: str; original_dataset: str
    phishtank_file: str; tranco_file: str; top_n_domains: int; scenarios: dict


def main() -> None:
    for path in (PHISHTANK_FILE, TRANCO_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw file: {path}\n"
                f"Place external validation CSVs in: {RAW}"
            )

    rid = run_id()
    out = OUT_ROOT / rid
    (out / "datasets").mkdir(parents=True, exist_ok=True)

    print(f"Run ID: {rid}")
    print("Loading training domains for overlap removal...")
    existing_urls, existing_domains = load_original_domains()
    print(f" Training URLs: {len(existing_urls):,} | Domains: {len(existing_domains):,}")

    scenario_meta = {}
    for name, paths in {"aggressive": AGGRESSIVE_PATHS, "neutral": NEUTRAL_PATHS}.items():
        print(f"\nBuilding scenario: {name}")
        df = build_scenario(paths, name)
        before = len(df)
        df = remove_overlap(df, existing_urls, existing_domains)
        counts = df["label"].value_counts().to_dict()
        scenario_meta[name] = {
            "paths": paths,
            "rows_before_overlap_removal": int(before),
            "rows_after_overlap_removal": int(len(df)),
            "label_counts": {str(k): int(v) for k, v in counts.items()},
        }
        out_path = out / "datasets" / f"external_dataset_{name}.csv"
        df.to_csv(out_path, index=False)
        print(f" Labels: {counts} -> {out_path}")

    manifest = Manifest(
        created_at=datetime.now().isoformat(), run_dir=str(out),
        original_dataset=str(ORIGINAL_DATASET), phishtank_file=str(PHISHTANK_FILE),
        tranco_file=str(TRANCO_FILE), top_n_domains=int(TOP_N_DOMAINS),
        scenarios=scenario_meta,
    )
    (out / "00_manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2), encoding="utf-8"
    )
    print(f"\nDatasets written to: {out}")


if __name__ == "__main__":
    main()
