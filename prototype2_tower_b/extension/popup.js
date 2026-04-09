/**
 * Tower B - Prototype 2 - popup.js
 *
 * Purpose
 * Controls the Prototype 2 popup. Reads the last Tower B scan result from the service worker and renders it, including
 * the infrastructure metadata signals (dns_ok, tls_ok, http_ok) that form the basis of the score.
 *
 * What is shown and why
 * Unlike Prototype 1, which showed p_phish from lexical features, this popup shows:
 *
 *   - hostname: Tower B scores domains, not full URL paths. Showing the hostname makes clear what was actually assessed.
 *
 *   - p_b: the infrastructure-based phishing probability produced by the HGB model on 12 DNS/TLS/HTTP features.
 *
 *   - Metadata panel: shows which of the three probe types succeeded. This is intentionally verbose in Prototype 2 to
 *   make the signal basis of the score visible during development and testing. In Prototype 3 this detail is moved to
 *   an expandable section.
 *
 *   - Cache indicator: first visits trigger live probes (2–8 seconds); repeat visits return cached results instantly.
 *   Showing this status makes the latency difference understandable.
 *
 * Communication with the service worker
 * Uses the same two-message protocol as Prototype 1:
 *   { type: "getResult", tabId }  -> retrieve stored result on popup open
 *   { type: "scanNow", tabId, url } -> trigger immediate manual scan
 */

// DOM references
const hostnameDisplay= document.getElementById("hostnameDisplay");
const scoreCard = document.getElementById("scoreCard");
const riskPill= document.getElementById("riskPill");
const pScore= document.getElementById("pScore");
const decisionBadge= document.getElementById("decisionBadge");
const metaPanel= document.getElementById("metaPanel");
const metaDns= document.getElementById("metaDns");
const metaTls= document.getElementById("metaTls");
const metaHttp= document.getElementById("metaHttp");
const cacheIndicator= document.getElementById("cacheIndicator");
const scanBtn= document.getElementById("scanBtn");
const statusText= document.getElementById("statusText");

// Render helpers
/**
 * Renders a probe success indicator into a span element.
 * 1 = probe succeeded (green OK), 0 = probe failed (red FAIL), null/undefined = probe result not available (grey -).
 *
 * @param {HTMLElement} el    - The span to update.
 * @param {number|null} value - 1, 0, or null.
 */
function renderProbeStatus(el, value) {
    if (value === 1) {
        el.textContent = "OK";
        el.className = "meta-ok";
    } else if (value === 0) {
        el.textContent = "FAIL";
        el.className = "meta-fail";
    } else {
        el.textContent = "-";
        el.className = "meta-na";
    }
}

/**
 * Renders the full /predict response into the popup DOM.
 *
 * Handles four states:
 *   - No result yet: prompt user to scan.
 *   - Backend error: show error message.
 *   - Skipped URL: not scannable (browser-internal page).
 *   - Normal result: show score, risk, decision, and metadata panel.
 *
 * @param {Object|null} result  - Parsed /predict response from service worker.
 * @param {string|null} url     - The URL associated with the result.
 */
function renderResult(result, url) {
    // Hostname display
    // Extract hostname from the URL for display. If the result includes a hostname field (set by the backend), prefer
    // that as it reflects what was actually probed.
    let displayHost = "-";
    if (result?.hostname) {
        displayHost = result.hostname;
    } else if (url) {
        try { displayHost = new URL(url).hostname; } catch (_) { displayHost = url; }
    }
    hostnameDisplay.textContent = displayHost;
    hostnameDisplay.title = url || "";

    // Error state
    if (!result) {
        setStatus("Not scanned yet. Click 'Scan Current Domain' to score this domain.");
        return;
    }

    if (result.error) {
        setStatus(`Backend error: ${result.error}`, true);
        return;
    }

    // Skipped
    if (result.skipped) {
        setStatus("This URL cannot be scanned (browser-internal page or no hostname).");
        return;
    }

    // Normal result
    const risk = result.risk_level || "unknown";
    const dec = result.decision || "allow";
    const pb = typeof result.p_b === "number" ? result.p_b : null;
    const debug = result.debug || {};
    const meta = debug.meta || {};
    const cacheHit = debug.cache_hit === true;

    // Risk pill
    riskPill.textContent = risk.toUpperCase();
    riskPill.className = `risk-pill ${risk}`;

    // p_b as percentage
    pScore.textContent = pb !== null ? `${(pb * 100).toFixed(1)}%` : "-";

    // Decision badge
    decisionBadge.textContent = dec.toUpperCase();
    decisionBadge.className = `decision-badge ${dec}`;

    // Infrastructure signal indicators
    renderProbeStatus(metaDns, meta.dns_ok);
    renderProbeStatus(metaTls, meta.tls_ok);
    renderProbeStatus(metaHttp, meta.http_ok);

    // Cache indicator - explains latency to the user
    if (cacheHit) {
        const ageSecs = debug.cache_age_s;
        const ageStr = typeof ageSecs === "number"
            ? ` (${Math.round(ageSecs / 3600)}h old)`
            : "";
        cacheIndicator.textContent = `Cached result${ageStr}`;
        cacheIndicator.className = "cache-hit";
    } else {
        cacheIndicator.textContent = "Live probe";
        cacheIndicator.className = "cache-probe";
    }

    // Show hidden panels
    scoreCard.classList.remove("hidden");
    metaPanel.classList.remove("hidden");
    clearStatus();
}

function setStatus(message, isError = false) {
    statusText.textContent = message
    statusText.className = isError ? "error" : "";
}

function clearStatus() {
    statusText.textContent = "";
    statusText.className = "";
}


// Initialisation

async function init() {
    try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab) {
            setStatus("Cannot determine active tab.", true);
            return;
        }

    // Request the stored result from the service worker

        chrome.runtime.sendMessage(
            { type: "getResult", tabId: tab.id },
            (response) => {
                if (chrome.runtime.lastError) {
                    setStatus("Service worker not responding.", true);
                    return;
                }
                renderResult(response?.result || null, response?.url || tab.url);
            }
            );

    // Store tab info for the Scan Now button

        scanBtn.dataset.tabId = tab.id;
        scanBtn.dataset.url = tab.url;

    } catch (err) {
        setStatus(`Popup error: ${err.message}`, true);
    }
}


// Scan button

scanBtn.addEventListener("click", async () => {
    const tabId = parseInt(scanBtn.dataset.tabId, 10);
    const url = scanBtn.dataset.url || "";

    if (!url) {
        setStatus("No URL to scan.", true);
        return;
    }

    scanBtn.disabled = true;
    scanBtn.textContent = "Scanning…";
    setStatus("Probing domain infrastructure - this may take a few seconds on first visit.");

    chrome.runtime.sendMessage(
        { type: "scanNow", tabId, url },
        (response) => {
            scanBtn.disabled = false;
            scanBtn.textContent = "Scan Current Domain";

            if (chrome.runtime.lastError) {
                setStatus("Service worker error: " + chrome.runtime.lastError.message, true);
                return;
            }

            renderResult(response?.result || null, url);
        }
        );
});


// Entry point
init();