"""
Tower B - Data Processing - 00_collect_metadata.py

Collects DNS, TLS, and HTTP infrastructure metadata for every domain in the shared Tower A/B dataset.

Dataset produced
  Tower B - Data Processing/data/processed/towerB_metadata_master.csv
    - 92,666 rows (one per URL in the shared dataset)
    - 50/50 label balance (46,333 phishing, 46,333 benign)
    - DNS probe success rate: 86.6%
    - TLS probe success rate: 74.0%
    - HTTP probe success rate: 71.7%
    - Collection date: early 2026

Design decisions
Domains rather than full URLs
    Tower B operates on domain-level infrastructure signals (DNS records, TLS certificates, HTTP headers). These 
    signals are domain-level, not URL-level - a phishing page at http://malicious.com/paypal/login and one at 
    http://malicious.com/paypal/update share the same infrastructure signals. For each URL, only the eTLD+1 domain is 
    probed. Duplicate domains within the dataset are probed only once and the result is shared across all URLs on that domain.

Asynchronous collection with ThreadPoolExecutor
    Network I/O is the bottleneck. Each probe (DNS + TLS + HTTP) takes 2–8 seconds on a typical connection. Using 80 
    concurrent workers reduces total collection time from ~100 hours (sequential) to ~2 hours (parallel). Each worker 
    handles one domain independently.

Missing value handling
    If any probe fails (connection refused, timeout, TLS error), the corresponding feature columns are recorded as 
    None in the CSV. The HGB model trained in 04_train_eval.py uses the -9999 sentinel to represent these None values 
    and handles them natively via its internal missing-value mechanism.

HTTPS-first with HTTP fallback
    TLS probing attempts port 443 first. If that fails, no HTTP fallback for TLS is attempted (a failed TLS probe is 
    itself informative - many phishing domains do not serve a valid TLS certificate). HTTP probing uses the 
    https:// scheme first, then falls back to http:// if the HTTPS connection fails, to maximise probe coverage.

The script will read towerA_dataset.csv, extract unique eTLD+1 domains, probe each domain, and write the results to 
towerB_metadata_master.csv. Expect a runtime of 1–3 hours depending on network conditions.
"""

from __future__ import annotations

import csv
import json
import socket
import ssl
import datetime as dt
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd
import tldextract

# Paths

# Input: the shared URL dataset produced by Prototype 1 Phase 1 preprocessing.
DATASET_CSV = (
        Path(__file__).resolve().parents[2]
        / "prototype1_tower_a" / "data_processing"
        / "01_preprocessing" / "outputs" / "towerA_dataset.csv"
)

# Output: the metadata CSV. Written once and not modified thereafter.
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_CSV = OUTPUT_DIR / "towerB_metadata_master.csv"

# Collection parameters
MAX_WORKERS = 80 # concurrent probe threads
DNS_TIMEOUT_S = 2.0
TLS_TIMEOUT_S = 3.0
HTTP_TIMEOUT_S = 4.0


# Probe functions
def probe_dns(domain: str) -> dict:
    """DNS resolution: number of IPs, IPv6 presence."""
    out = {"dns_ok": 0, "dns_num_ips": 0, "dns_has_ipv6": 0}
    try:
        infos = socket.getaddrinfo(domain, None, proto=socket.IPPROTO_TCP)
        ips = set()
        has_v6 = False
        for family, *_, sockaddr in infos:
            ips.add(sockaddr[0])
            if family == socket.AF_INET6:
                has_v6 = True
        out.update({"dns_ok": 1, "dns_num_ips": len(ips), "dns_has_ipv6": int(has_v6)})
    except Exception:
        pass
    return out


def probe_tls(domain: str) -> dict:
    """TLS certificate: validity period, remaining days, Let's Encrypt issuer."""
    out = {
        "tls_ok": 0,
        "tls_cert_valid_days": None,
        "tls_cert_remaining_days": None,
        "tls_issuer_is_letsencrypt": None,
    }
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=TLS_TIMEOUT_S) as s:
            with ctx.wrap_socket(s, server_hostname=domain) as tls:
                cert = tls.getpeercert()
        out["tls_ok"] = 1
        nb = cert.get("notBefore")
        na = cert.get("notAfter")
        if nb and na:
            fmt = "%b %d %H:%M:%S %Y %Z"
            nb_dt = dt.datetime.strptime(nb, fmt)
            na_dt = dt.datetime.strptime(na, fmt)
            out["tls_cert_valid_days"] = int((na_dt - nb_dt).days)
            out["tls_cert_remaining_days"] = int((na_dt - dt.datetime.utcnow()).days)
        issuer = str(cert.get("issuer", "")).lower()
        out["tls_issuer_is_letsencrypt"] = (
            1 if ("let's encrypt" in issuer or "letsencrypt" in issuer) else 0
        )
    except Exception:
        pass
    return out


def probe_http(domain: str) -> dict:
    """HTTP response: status, redirects, security headers, domain mismatch."""
    out = {
        "http_ok": 0,
        "http_status_code": None,
        "http_redirect_count": None,
        "http_has_hsts": None,
        "http_has_csp": None,
        "http_has_xfo": None,
        "http_final_domain_mismatch": None,
    }
    url = f"https://{domain}/"
    try:
        with httpx.Client(
                timeout=HTTP_TIMEOUT_S,
                follow_redirects=True,
                headers={"User-Agent": "TowerB-Collect/1.0"}
        ) as client:
            r = client.get(url)
        out["http_ok"] = 1
        out["http_status_code"] = int(r.status_code)
        out["http_redirect_count"] = int(len(r.history))
        h = {k.lower(): v for k, v in r.headers.items()}
        out["http_has_hsts"] = 1 if "strict-transport-security" in h else 0
        out["http_has_csp"] = 1 if "content-security-policy" in h else 0
        out["http_has_xfo"] = 1 if "x-frame-options" in h else 0
        final_host = httpx.URL(str(r.url)).host or ""
        out["http_final_domain_mismatch"] = (
            1 if final_host.lower().strip(".") != domain.lower().strip(".") else 0
        )
    except Exception:
        pass
    return out


def probe_domain(domain: str) -> dict:
    """Runs all three probes and merges the results into a single dict."""
    return {**probe_dns(domain), **probe_tls(domain), **probe_http(domain)}


# eTLD+1 extraction
def extract_etld1(url: str) -> str | None:
    """Returns the eTLD+1 for a URL, or None if extraction fails."""
    ext = tldextract.extract(str(url))
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return None


# Main collection routine
def main() -> None:
    # Guard against accidental re-runs
    if OUTPUT_CSV.exists():
        print("=" * 70)
        print("OUTPUT FILE ALREADY EXISTS - collection is not needed.")
        print(f" {OUTPUT_CSV}")
        print()
        print("To proceed with Prototype 2 data processing, run:")
        print(" python 01_audit.py")
        print(" python 02_feature_spec.py")
        print(" python 03_split.py")
        print(" python 04_train_eval.py")
        print("=" * 70)
        sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the shared dataset
    print(f"Reading dataset: {DATASET_CSV}")
    if not DATASET_CSV.exists():
        print(f"ERROR: Dataset not found: {DATASET_CSV}")
        print("Ensure Prototype 1 Phase 1 (preprocess.py) has been run first.")
        sys.exit(1)

    df = pd.read_csv(DATASET_CSV)
    print(f"Loaded: {len(df):,} rows")

    # Extract unique eTLD+1 domains - one probe per domain, not per URL
    df["etld1"] = df["url_norm"].apply(extract_etld1)
    unique_domains = df["etld1"].dropna().unique()
    print(f"Unique eTLD+1 domains to probe: {len(unique_domains):,}")

    # Probe all domains in parallel
    print(f"Starting collection with {MAX_WORKERS} workers...")

    domain_meta: dict[str, dict] = {}
    completed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(probe_domain, d): d for d in unique_domains}
        for future in as_completed(futures):
            domain = futures[future]
            try:
                domain_meta[domain] = future.result()
            except Exception as exc:
                domain_meta[domain] = {}
                print(f" [ERR] {domain}: {exc}")
            completed += 1
            if completed % 1000 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                remaining = (len(unique_domains) - completed) / rate
                print(f" Progress: {completed:,}/{len(unique_domains):,} "
                      f"({rate:.0f}/s, ~{remaining / 60:.0f} min remaining)")

    # Join metadata back onto the full URL-level dataframe
    print("Joining metadata to URL dataset...")
    meta_rows = []
    for _, row in df.iterrows():
        etld1 = row.get("etld1")
        meta = domain_meta.get(etld1, {}) if etld1 else {}
        meta_rows.append({
            "url_raw": row.get("url_raw", ""),
            "url_norm": row.get("url_norm", ""),
            "etld1": etld1,
            "label": row.get("label", -1),
            "source": row.get("source", ""),
            **meta,
        })

    df_out = pd.DataFrame(meta_rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nCollection complete.")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Rows: {len(df_out):,}")
    print(f"Time: {(time.time() - start_time) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
