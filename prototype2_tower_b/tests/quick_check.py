"""
Tower B - Prototype 2 - tests/quick_check.py

Smoke test for the Tower B backend.

This script verifies three things:
  1. The backend is reachable and the model is loaded (/health).
  2. The configuration endpoint returns expected keys (/config).
  3. A small set of URLs can be scored without errors (/predict).

It is not a parity check (Tower B has no offline equivalent to compare against - metadata probing is stateful and 
time-dependent). Instead, it confirms that the full pipeline from URL to p_b score works end-to-end.

Start the backend first:
    uvicorn backend.app:app --host 127.0.0.1 --port 8001

Then run from the prototype2_tower_b/ directory:
    python tests/quick_check.py

Expected output: all URLs produce a p_b score between 0 and 1 with no errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from tests/ or from prototype2_tower_b/ 
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import requests

API_URL = "http://127.0.0.1:8001"

# A small representative set: benign, borderline, and likely phishing.
TEST_URLS = [
    "https://www.google.com/",
    "https://www.bbc.co.uk/news",
    "https://paypal-update.secure-verify.com/login",
    "https://accounts.google.com-secure-login.xyz/signin",
    "https://reddit.com/r/learnpython/comments/abc",
    "http://secure-paypal.com/webscr?cmd=_login-run",
]

def check_health() -> bool:
    """Returns True if the backend is reachable and the model is loaded."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        data = r.json()
        print(f" status: {data.get('status')}")
        print(f" model_loaded: {data.get('towerB_loaded')}")
        print(f" decision_mode: {data.get('decision_mode')}")
        return data.get("status") == "ok" and data.get("towerB_loaded") is True
    except requests.exceptions.ConnectionError:
        return False


def check_config() -> None:
    """Prints the active configuration returned by /config."""
    r = requests.get(f"{API_URL}/config", timeout=5)
    cfg = r.json()
    print(f"thresholds: red={cfg['thresholds']['threshold_red']}"
          f" orange={cfg['thresholds']['threshold_orange']}")
    print(f"cache TTL: {cfg['cache']['ttl_seconds'] // 3600}h")
    print(f"timeouts: dns={cfg['timeouts_s']['dns']}s"
          f"tls={cfg['timeouts_s']['tls']}s"
          f"http={cfg['timeouts_s']['http']}s")


def check_predictions() -> int:
    """
    Sends each test URL to /predict and prints the result.
    Returns the number of URLs that failed (scored without a p_b value).
    """
    failures = 0
    for url in TEST_URLS:
        try:
            r = requests.post(f"{API_URL}/predict", json={"url": url, "mode": "test"}, timeout=15)
            data = r.json()

            if data.get("skipped"):
                print(f" [SKIP] {url}")
                continue

            p_b = data.get("p_b")
            risk = data.get("risk_level", "?")
            decision = data.get("decision", "?")
            cached = data.get("debug", {}).get("cache_hit", False)
            tag = "CACHE" if cached else "PROBE"

            if p_b is not None:
                print(f" [PASS][{tag}] p_b={p_b:.4f} {risk}/{decision} {url}")
            else:
                reason = data.get("reason", "unknown")
                print(f" [FAIL] reason={reason} {url}")
                failures += 1

        except Exception as exc:
            print(f" [ERR ] {exc} {url}")
            failures += 1

    return failures

def main() -> None:
    print("=" * 70)
    print("Tower B - Prototype 2 - Quick Check")
    print(f"Backend: {API_URL}")
    print("=" * 70)

    print("\n[1] Health check")
    if not check_health():
        print(f"\nERROR: Cannot reach backend at {API_URL}")
        print("Start the backend first:")
        print(" uvicorn backend.app:app --host 127.0.0.1 --port 8001")
        sys.exit(1)

    print("\n[2] Configuration")
    check_config()

    print(f"\n[3] Predictions ({len(TEST_URLS)} URLs)")
    print(" Note: first-time probes may take 5-10 seconds per URL.")
    failures = check_predictions()

    print()
    print("=" * 70)
    if failures == 0:
        print("RESULT: PASS - all URLs scored successfully.")
    else:
        print(f"RESULT: {failures} URL(s) failed to score. Check the backend logs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
