/**
 * Tower A - Prototype 1 - popup.js
 * Purpose
 * Controls the extension action popup. Reads the last scan result stored by the service worker for the currently 
 * active tab and renders it in the UI. Also handles the "Scan Current Page" button which triggers an immediate manual 
 * scan via a message to the service worker.
 * 
 * Communication with the service worker
 * All scan logic lives in service_worker.js. popup.js communicates via chrome.runtime.sendMessage with two message types:
 *   { type: "getResult", tabId: <number> }
 *     -> Service worker responds with { result, url } from its tabState Map.
 *
 *   { type: "scanNow", tabId: <number>, url: <string> }
 *     -> Service worker performs an immediate scan and responds with { result }.
 *
 * Rendering
 * renderResult() maps the /predict JSON response to the DOM elements defined in popup.html. It handles three states:
 *   - No result yet (show placeholder, prompt user to click Scan Now)
 *   - Skipped URL (internal browser page, cannot be scored)
 *   - Normal result (risk pill, p_phish percentage, decision badge, phase)
 */

// DOM references
const urlDisplay= document.getElementById("urlDisplay");
const scoreCard= document.getElementById("scoreCard");
const riskPill= document.getElementById("riskPill");
const pScore= document.getElementById("pScore");
const decisionBadge= document.getElementById("decisionBadge");
const thresholdRow= document.getElementById("thresholdRow");
const phaseLabel= document.getElementById("phaseLabel");
const bufferSize= document.getElementById("bufferSize");
const scanBtn= document.getElementById("scanBtn");
const statusText= document.getElementById("statusText");

// Render helpers
function renderResult(result, url) {
  // URL display
  if (url) {
    urlDisplay.textContent = url.length > 70 ? url.slice(0, 67) + "…" : url;
    urlDisplay.title = url;
  } else {
    urlDisplay.textContent = "No URL available.";
  }
  
  // No result yet
  if (!result) {
    setStatus("Not scanned yet. Click 'Scan Current Page' to score this URL.");
    return;
  }
  
  // Backend error
  if (result.error) {
    setStatus(`Backend error: ${result.error}`, true);
    return;
  }

  // Skipped URL (internal browser page)
  if (result.skipped) {
    setStatus("This URL cannot be scanned (browser-internal page).");
    return;
  }

  // Normal result
  const risk = result.risk_level || "unknown";
  const dec = result.decision || "allow";
  const p = typeof result.p_phish === "number" ? result.p_phish : null;
  const phase = result.thresholds?.phase || "unknown";
  const buf = result.thresholds?.buffer_size ?? "-";

  // Risk pill
  riskPill.textContent = risk.toUpperCase();
  riskPill.className = `risk-pill ${risk}`;

  // p_phish as percentage
  pScore.textContent = p !== null ? `${(p * 100).toFixed(1)}%` : "-";

  // Decision badge
  decisionBadge.textContent = dec.toUpperCase();
  decisionBadge.className = `decision-badge ${dec}`;

  // Adaptive phase
  const phaseDisplay = {
    "learning":           "Learning (no blocking)",
    "adapting_no_block":  "Adapting (warn only)",
    "stable_block_enabled": "Stable (full enforcement)"
  };
  phaseLabel.textContent = phaseDisplay[phase] || phase;
  bufferSize.textContent = buf;

  // Show hidden elements
  scoreCard.classList.remove("hidden");
  thresholdRow.classList.remove("hidden");
  clearStatus();
}

/** Shows a status message below the scan button. */
function setStatus(message, isError = false) {
  statusText.textContent = message;
  statusText.className = isError ? "error" : "";
}

/** Clears the status message area. */
function clearStatus() {
  statusText.textContent = "";
  statusText.className = "";
}

// Initialisation: load stored result on popup open
/**
 * On popup open, ask the service worker for the last stored result for
 * the active tab. Renders it immediately if available.
 */
async function init() {
  try {
    // Get the active tab in the current window
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
    
    // Store the tab ID and URL for the Scan Now button
    scanBtn.dataset.tabId = tab.id;
    scanBtn.dataset.url = tab.url;

  } catch (err) {
    setStatus(`Popup error: ${err.message}`, true);
  }
}

// Scan Now button
/**
 * Triggers an immediate manual scan of the active tab URL.
 * Disables the button during the request to prevent double-clicks.
 */
scanBtn.addEventListener("click", async () => {
  const tabId = parseInt(scanBtn.dataset.tabId, 10);
  const url = scanBtn.dataset.url || "";

  if (!url) {
    setStatus("No URL to scan.", true);
    return;
  }

  // Disable button and show loading state
  scanBtn.disabled = true;
  scanBtn.textContent = "Scanning…";
  setStatus("Calling Tower A backend…");

  chrome.runtime.sendMessage(
    { type: "scanNow", tabId, url },
    (response) => {
      // Re-enable button
      scanBtn.disabled = false;
      scanBtn.textContent = "Scan Current Page";

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
