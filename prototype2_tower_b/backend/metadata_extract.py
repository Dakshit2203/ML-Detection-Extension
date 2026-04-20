"""
Tower B - Prototype 2 - backend/metadata_extract.py

Live network probing functions for DNS, TLS, and HTTP infrastructure signals.

DNS, TLS, and HTTP signals are complementary to Tower A's URL lexical features. A phishing page can have a URL that
looks completely legitimate while still exhibiting infrastructure signals that betray its nature:
  DNS: Phishing domains often resolve to a larger number of IPs (fast-flux networks used to evade blacklisting) or
  fail to resolve altogether when the campaign has been taken down.

  TLS: Short-lived phishing domains frequently use Let's Encrypt certificates (free, automated, no identity
  verification). Certificate validity periods shorter than 90 days are a known phishing indicator.

  HTTP: Legitimate sites typically implement security headers (HSTS, CSP, X-Frame-Options). Their absence is weakly
  indicative of phishing. A high redirect count can indicate URL cloaking.

Missing value handling
If any probe fails (network timeout, connection refused, TLS error), the corresponding features are recorded as None.
The HGB model handles None values natively via its internal missing-value mechanism - it was trained with the same
Nan that towerB.py uses to fill None values before calling predict_proba(). This means a failed TLS probe
does not prevent scoring; it simply means the model uses its learned fallback for TLS features.

Timeout values
Timeouts are passed in from the caller (towerB.py reads them from config.py).
This keeps metadata_extract.py stateless and testable in isolation.
"""

from __future__ import annotations

import datetime as dt
import socket
import ssl
from typing import Any, Dict

import httpx


def _dns_probe(domain: str) -> Dict[str, Any]:
    """
    Resolves the domain and extracts basic DNS features.

    Features produced
    dns_ok: 1 if resolution succeeded, 0 otherwise.
    dns_num_ips: number of distinct IP addresses returned (IPv4 + IPv6).
    dns_has_ipv6: 1 if at least one IPv6 address is present, 0 otherwise.

    A high dns_num_ips value can indicate a fast-flux network, which is a known tactic used by phishing campaigns to
    keep domains reachable while rotating IPs faster than blacklists can track them.
    """
    result = {"dns_ok": 0, "dns_num_ips": 0, "dns_has_ipv6": 0}
    try:
        infos = socket.getaddrinfo(domain, None, proto=socket.IPPROTO_TCP)
        ips = set()
        has_v6 = False
        for family, *_, sockaddr in infos:
            ips.add(sockaddr[0])
            if family == socket.AF_INET6:
                has_v6 = True
        result["dns_ok"] = 1
        result["dns_num_ips"] = len(ips)
        result["dns_has_ipv6"] = 1 if has_v6 else 0
    except Exception:
        pass
    return result


def _tls_probe(domain: str, timeout_s: float) -> Dict[str, Any]:
    """
    Connects to port 443 and extracts TLS certificate features.

    Features produced
    tls_ok: 1 if TLS handshake succeeded, 0 otherwise.
    tls_cert_valid_days: total certificate validity period in days.
    tls_cert_remaining_days: days until certificate expiry (can be negative if the certificate has already expired).
    tls_issuer_is_letsencrypt: 1 if the issuer is Let's Encrypt, 0 otherwise.

    Let's Encrypt certificates are free and require no identity verification, making them the default choice for
    low-effort phishing infrastructure. While many legitimate sites also use Let's Encrypt, the combination of a short
    validity period and a Let's Encrypt issuer is a known risk signal.
    """
    result = {
        "tls_ok": 0,
        "tls_cert_valid_days": None,
        "tls_cert_remaining_days": None,
        "tls_issuer_is_letsencrypt": None,
    }
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=timeout_s) as raw_sock:
            with ctx.wrap_socket(raw_sock, server_hostname=domain) as tls_sock:
                cert = tls_sock.getpeercert()

        result["tls_ok"] = 1

        not_before = cert.get("notBefore")
        not_after = cert.get("notAfter")
        if not_before and not_after:
            fmt = "%b %d %H:%M:%S %Y %Z"
            nb_dt = dt.datetime.strptime(not_before, fmt)
            na_dt = dt.datetime.strptime(not_after, fmt)
            now_dt = dt.datetime.utcnow()
            result["tls_cert_valid_days"] = int((na_dt - nb_dt).days)
            result["tls_cert_remaining_days"] = int((na_dt - now_dt).days)

        issuer_str = str(cert.get("issuer", "")).lower()
        result["tls_issuer_is_letsencrypt"] = (
            1 if ("let's encrypt" in issuer_str or "letsencrypt" in issuer_str)
            else 0
        )
    except Exception:
        pass
    return result


def _http_probe(domain: str, timeout_s: float) -> Dict[str, Any]:
    """
    Makes an HTTPS GET request and extracts HTTP response features.

    Features produced
    http_ok: 1 if the request succeeded (any 2xx/3xx), 0 if connection failed or timed out.
    http_status_code: final HTTP status code after following redirects.
    http_redirect_count: number of redirects followed before the final response. High counts can indicate cloaking.
    http_has_hsts: 1 if Strict-Transport-Security header present.
    http_has_csp: 1 if Content-Security-Policy header present.
    http_has_xfo: 1 if X-Frame-Options header present.
    http_final_domain_mismatch: 1 if the final URL host differs from the originally requested domain (i.e., the request
                                was redirected to a completely different domain).

    httpx is used instead of the standard library's urllib because it handles redirects cleanly, supports connection
    timeouts per request, and provides a consistent API across HTTP/1.1 and HTTP/2.
    """
    result = {
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
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": "TowerB-Probe/1.0"}
        ) as client:
            response = client.get(url)

        result["http_ok"] = 1
        result["http_status_code"] = int(response.status_code)
        result["http_redirect_count"] = int(len(response.history))

        # Check for common security headers (case-insensitive lookup)
        headers = {k.lower(): v for k, v in response.headers.items()}
        result["http_has_hsts"] = 1 if "strict-transport-security" in headers else 0
        result["http_has_csp"] = 1 if "content-security-policy" in headers else 0
        result["http_has_xfo"] = 1 if "x-frame-options" in headers else 0

        # Detect cross-domain redirects: the final host after following all redirects is compared to the originally requested domain.
        final_host = httpx.URL(str(response.url)).host or ""
        result["http_final_domain_mismatch"] = (
            1 if final_host.lower().strip(".") != domain.lower().strip(".") else 0
        )
    except Exception:
        pass
    return result


def extract_domain_metadata(
    domain: str,
    tls_timeout_s: float = 3.0,
    http_timeout_s: float = 4.0,
) -> Dict[str, Any]:
    """
    Runs all three probes (DNS, TLS, HTTP) for a domain and returns a merged metadata dictionary.

    This is the single entry point called by towerB.py. All three probes are run sequentially. If any probe raises an
    exception, its features are left at their default (failure) values and the remaining probes continue. This ensures
    a partial result is always returned rather than an error.

    Parameters
    domain: str The hostname to probe (no scheme, no path).
    tls_timeout_s: float Timeout for the TLS handshake, in seconds.
    http_timeout_s: float Timeout for the full HTTP round-trip, in seconds.

    Returns
    Dict[str, Any]
        Flat dictionary of all features. None values indicate a failed probe for that signal. The caller (towerB.py)
        converts None to -9999 before passing the feature vector to the model.
    """
    dns = _dns_probe(domain)
    tls = _tls_probe(domain, tls_timeout_s)
    http = _http_probe(domain, http_timeout_s)

    return {**dns, **tls, **http}
