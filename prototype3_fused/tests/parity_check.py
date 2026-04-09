"""
Prototype 3 - Fused System - tests/parity_check.py

Validates that the fused backend's reported p_fused is consistent with the formula 0.75*p_A + 0.25*p_B, using the
p_a and p_b values returned in the same response.

This confirms that the fusion formula in app.py and fusion.py are correct and that no rounding error has been
introduced by JSON serialisation. A max |diff| < 1e-5 constitutes a PASS.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import requests

API_URL  = "http://127.0.0.1:8002"
WEIGHT_A = 0.75
WEIGHT_B = 0.25

TEST_URLS = [
    "https://www.google.com/",
    "https://www.bbc.co.uk/news",
    "https://github.com/Dakshit2203/ML-Detection-Extension",
    "https://www.amazon.co.uk/",
    "https://stackoverflow.com/questions/tagged/python",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://www.gov.uk/government/publications",
    "https://paypal-update.secure-verify.com/login",
    "https://accounts.google.com-secure-login.xyz/signin",
    "https://example.com/verify-your-account?token=abc123",
    "https://reddit.com/r/learnpython/comments/abc",
    "https://news.ycombinator.com/",
    "http://secure-paypal.com/webscr?cmd=_login-run",
    "http://update-your-bank-details.info/NatWest/login",
    "https://apple-id-verify-now.com/appleid/auth",
]

def check_health() -> bool:
    try:
        data = requests.get(f"{API_URL}/health", timeout=5).json()
        print(f" status: {data.get('status')}")
        print(f" tower_a_loaded: {data.get('tower_a_loaded')}")
        print(f" tower_b_loaded: {data.get('tower_b_loaded')}")
        print(f" fusion_weights: {data.get('fusion_weights')}")
        print(f" buffer_size: {data.get('buffer_size')}")
        return data.get("status") == "ok" and data.get("tower_a_loaded") and data.get("tower_b_loaded")
    except requests.exceptions.ConnectionError:
        return False

def main() -> None:
    print("=" * 70)
    print("Prototype 3 - Parity Check")
    print(f"Backend: {API_URL} | Formula: {WEIGHT_A}*pA + {WEIGHT_B}*pB")
    print(f"Test URLs: {len(TEST_URLS)}")
    print("=" * 70)

    print("\n[1] Health check")
    if not check_health():
        print(f"\nERROR: Cannot reach {API_URL}")
        print("Start the backend: uvicorn backend.app:app --host 127.0.0.1 --port 8002")
        sys.exit(1)

    print(f"\n[2] Parity check ({len(TEST_URLS)} URLs)")
    diffs = []
    pass_count = 0
    fail_count = 0

    for url in TEST_URLS:
        try:
            data = requests.post(
                f"{API_URL}/predict",
                json={"url": url, "mode": "parity_check"},
                timeout=20,
            ).json()

            if data.get("skipped"):
                print(f" [SKIP] {url}")
                continue

            p_a = data.get("p_a")
            p_b = data.get("p_b")
            p_fused = data.get("p_fused")
            b_ok = data.get("tower_b_available", False)

            if p_a is None or p_fused is None:
                print(f" [ERR ] Missing fields - {url}")
                fail_count += 1
                continue

            expected = WEIGHT_A * p_a + WEIGHT_B * p_b if (b_ok and p_b is not None) else p_a
            diff = abs(p_fused - expected)
            diffs.append(diff)

            status = "PASS" if diff < 1e-5 else "FAIL"
            pass_count += 1 if diff < 1e-5 else 0
            fail_count += 0 if diff < 1e-5 else 1

            b_tag = f"pB={p_b:.4f}" if b_ok and p_b is not None else "B=n/a"
            print(f" [{status}] diff={diff:.2e} pA={p_a:.4f} {b_tag} fused={p_fused:.4f} expected={expected:.4f}")
            print(f" {url[:72]}")

        except Exception as exc:
            fail_count += 1
            print(f" [ERR ] {exc} - {url[:72]}")

    print()
    print("=" * 70)
    if diffs:
        print(f"Max |diff|: {max(diffs):.2e}")
        print(f"Mean |diff|: {float(np.mean(diffs)):.2e}")
        print(f"Passed: {pass_count} / {pass_count + fail_count}")
        print()
        if max(diffs) < 1e-5 and fail_count == 0:
            print("RESULT: PASS - fusion formula and serialisation are correct.")
        else:
            print("RESULT: FAIL - check WEIGHT_A/WEIGHT_B in config.py and fusion.py.")
    else:
        print("RESULT: No successful comparisons - check backend connectivity.")
    print("=" * 70)


if __name__ == "__main__":
    main()
