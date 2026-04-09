/**
 * Prototype 3 - Fused System - popup.js
 *
 * Renders the fused scan result: score card, individual tower scores, XAI attribution bars for both towers, and the
 * adaptive phase indicator.
 *
 * XAI bar rendering
 * Bars are normalised to the maximum absolute value in the top-5 list.
 * Tower A: positive contribution (blue) = pushes towards phishing; negative contribution (green) = pushes towards benign.
 * Tower B: unsigned global importance, always shown in blue.
 */

//  DOM references 

const urlDisplay = document.getElementById("urlDisplay");
const scoreCard = document.getElementById("scoreCard");
const riskPill = document.getElementById("riskPill");
const pFused = document.getElementById("pFused");
const decisionBadge = document.getElementById("decisionBadge");
const towerRow = document.getElementById("towerRow");
const scoreA = document.getElementById("scoreA");
const scoreB = document.getElementById("scoreB");
const reducedConf = document.getElementById("reducedConf");
const xaiPanel = document.getElementById("xaiPanel");
const xaiA = document.getElementById("xaiA");
const xaiB = document.getElementById("xaiB");
const thresholdRow = document.getElementById("thresholdRow");
const phaseLabel = document.getElementById("phaseLabel");
const bufferSize = document.getElementById("bufferSize");
const scanBtn = document.getElementById("scanBtn");
const statusText = document.getElementById("statusText");


//  XAI bar chart 

/**
 * Renders a list of feature attributions as a proportional bar chart.
 *
 * @param {HTMLElement} container    Target div.
 * @param {Array}       attributions [{feature, contribution|importance, direction}]
 * @param {string}      valueKey     "contribution" (Tower A) or "importance" (Tower B)
 */
function renderXaiBars(container, attributions, valueKey) {
    container.innerHTML = "";

    if (!attributions || attributions.length === 0) {
        container.textContent = "No attribution data available.";
        return;
    }

    const maxAbs = Math.max(...attributions.map(a => Math.abs(a[valueKey] || 0)));

    for (const attr of attributions) {
        const val = attr[valueKey] || 0;
        const pct = maxAbs > 0 ? (Math.abs(val) / maxAbs) * 100 : 0;
        const barClass = val < 0 ? "xai-bar negative" : "xai-bar";
        const name = attr.feature.replace(/_/g, " ");

        const row = document.createElement("div");
        row.className = "xai-feature";
        row.innerHTML = `
            <span class="xai-name" title="${attr.feature}">${name}</span>
            <div class="xai-bar-wrap">
            <div class="${barClass}" style="width:${pct.toFixed(1)}%"></div>
        </div>
        <span class="xai-val">${val >= 0 ? "+" : ""}${val.toFixed(4)}</span>
    `;
        container.appendChild(row);
    }
}


//  Result renderer 

function renderResult(result, url) {
  // URL display
    urlDisplay.textContent = url
        ? (url.length > 60 ? url.slice(0, 57) + "…" : url)
        : "No URL available.";
    if (url) urlDisplay.title = url;

    if (!result) {
        setStatus("Not scanned yet. Click 'Scan Current Page'.");
        return;
    }
    if (result.error) { setStatus(`Backend error: ${result.error}`, true); return; }
    if (result.skipped) { setStatus("This URL cannot be scanned."); return; }

    // Fused score card
    const risk = result.risk_level || "unknown";
    const dec = result.decision || "allow";
    const pf = typeof result.p_fused === "number" ? result.p_fused : null;
    const pa = typeof result.p_a === "number" ? result.p_a : null;
    const pb = typeof result.p_b === "number" ? result.p_b : null;

    riskPill.textContent = risk.toUpperCase();
    riskPill.className = `risk-pill ${risk}`;
    pFused.textContent = pf !== null ? `${(pf * 100).toFixed(1)}%` : "-";
    decisionBadge.textContent = dec.toUpperCase();
    decisionBadge.className = `decision-badge ${dec}`;

    // Tower scores
    scoreA.textContent = pa !== null ? `${(pa * 100).toFixed(1)}%` : "-";
    scoreB.textContent = pb !== null ? `${(pb * 100).toFixed(1)}%` : "n/a";

    if (!result.tower_b_available) reducedConf.classList.remove("hidden");

    // XAI bars
    const xaiData = result.xai || {};
    if (xaiData.tower_a?.length > 0) {
        renderXaiBars(xaiA, xaiData.tower_a, "contribution");
    } else {
        xaiA.textContent = "No Tower A attributions available.";
    }
    if (xaiData.tower_b?.length > 0) {
        renderXaiBars(xaiB, xaiData.tower_b, "importance");
    } else {
        xaiB.textContent = "Tower B attributions unavailable.";
    }

    // Adaptive phase
    const phaseMap = {
        "learning": "Learning (no blocking)",
        "adapting_no_block": "Adapting (warn only)",
        "stable_block_enabled": "Stable (full enforcement)",
    };
    phaseLabel.textContent = phaseMap[result.thresholds?.phase] || result.thresholds?.phase || "-";
    bufferSize.textContent = result.thresholds?.buffer_size ?? "-";

    scoreCard.classList.remove("hidden");
    towerRow.classList.remove("hidden");
    xaiPanel.classList.remove("hidden");
    thresholdRow.classList.remove("hidden");
    clearStatus();
}

function setStatus(msg, isError = false) {
    statusText.textContent = msg;
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
        if (!tab) { setStatus("Cannot determine active tab.", true); return; }

        chrome.runtime.sendMessage({ type: "getResult", tabId: tab.id }, (response) => {
            if (chrome.runtime.lastError) { setStatus("Service worker not responding.", true); return; }
            renderResult(response?.result || null, response?.url || tab.url);
        });

        scanBtn.dataset.tabId = tab.id;
        scanBtn.dataset.url = tab.url;

    } catch (err) {
        setStatus(`Popup error: ${err.message}`, true);
    }
}

scanBtn.addEventListener("click", () => {
    const tabId = parseInt(scanBtn.dataset.tabId, 10);
    const url = scanBtn.dataset.url || "";
    if (!url) { setStatus("No URL to scan.", true); return; }

    scanBtn.disabled = true;
    scanBtn.textContent = "Scanning…";
    setStatus("Running both towers - first-visit domains may take a few seconds.");

    chrome.runtime.sendMessage({ type: "scanNow", tabId, url }, (response) => {
        scanBtn.disabled = false;
        scanBtn.textContent = "Scan Current Page";
        if (chrome.runtime.lastError) {
            setStatus("Service worker error: " + chrome.runtime.lastError.message, true);
            return;
        }
        renderResult(response?.result || null, url);
    });
});

init();
