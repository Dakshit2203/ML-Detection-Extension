/**
 * Tower A - Prototype 1 - service_worker.js
 *
 * Purpose
 * This MV3 service worker is the core of the Prototype 1 minimal extension. It intercepts all browser navigation 
 * events via chrome.webNavigation, debounces rapid navigations (to avoid flooding the backend during page load), calls 
 * the Tower A FastAPI /predict endpoint, and updates the extension badge with the resulting risk level.
 *
 * Design decisions
- Service worker (not background page): required by Chrome MV3.
- Stateless per-URL scoring: the backend holds all adaptive threshold state; the extension only stores the last result 
 per tab in an in-memory Map.
- Privacy: only the raw URL string is sent to localhost. No URL is persisted by the extension. The backend stores 
 scores only (not URLs), consistent with the GDPR privacy-by-design principle described in the project report.
- Debounce: a 800 ms debounce prevents multiple rapid-fire scans when a page issues several navigations during 
 load (e.g., redirects).
- Badge: colour-coded risk indicator visible at all times without opening the popup. Badge text: OK / LOW / WARN / DANG / ERR.
 
 * Data flow
 *   chrome.webNavigation.onCommitted
 *     -> debounce (800 ms)
 *     -> isScannableUrl()
 *     -> POST /predict to Tower A backend
 *     -> setBadge()  +  store result in tabState
 *     -> (if warn/block) showNotification()
 *
 * Usage
 *   Load the extension from chrome://extensions/ in Developer mode.
 *   Start the Tower A backend:
 *     cd prototype1_tower_a/backend
 *     uvicorn app:app --host 127.0.0.1 --port 8000
 */

// Constants and defaults
/**
 * Default settings. These are merged with any values the user has saved via chrome.storage.sync (populated by the 
 * options page in the full prototype).
 * For Prototype 1 minimal, only apiUrl and debounceMs are relevant.
 */
const DEFAULTS = {
  apiUrl:    "http://127.0.0.1:8000", // Tower A FastAPI backend
  debounceMs: 800, // ms to wait after navigation before scanning
  autoScan:  true, // scan on every navigation event
  notifyOn:  ["warn", "block"] // which decisions trigger a desktop notification
};

/**
 * A special tab-ID string used when a URL is scanned via the popup's "Scan Now" button rather than through an 
 * automatic navigation event. Badge updates are guarded by tabId >= 0; this key bypasses that check.
 */
const MANUAL_KEY = "manual";

/**
 * In-memory state per tab. Stores the last URL scanned, the last API result, and the debounce timer ID. Cleared when 
 * a tab is closed.
 * Shape: Map<tabId|"manual", { lastUrl, lastResult, timerId }>
 */
const tabState = new Map();

// Settings helpers
/**
 * Reads extension settings from chrome.storage.sync, falling back to DEFAULTS for any missing keys. Returns a Promise 
 * resolving to the merged settings.
 * @returns {Promise<Object>} Merged settings object.
 */
function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(Object.keys(DEFAULTS), (cfg) => {
      resolve({ ...DEFAULTS, ...cfg });
    });
  });
}

// URL utility
/**
 * Returns true if the URL is a normal http/https page that the backend can score. Returns false for browser-internal, 
 * file, and view-source URLs which the backend cannot handle and which should never be sent to localhost.
 *
 * This mirrors the server-side is_scannable_url() in backend/url_normalize.py so that the extension and backend agree 
 * on what is scanned.
 *
 * @param {string} url - Raw URL string from the navigation event.
 * @returns {boolean}
 */
function isScannableUrl(url) {
  if (!url) return false;
  const u = String(url).trim().toLowerCase();
  return !(
    u.startsWith("chrome://") ||
    u.startsWith("chrome-extension://") ||
    u.startsWith("edge://") ||
    u.startsWith("about:") ||
    u.startsWith("file://") ||
    u.startsWith("view-source:")
  );
}

// Backend call
/**
 * Sends a POST /predict request to the Tower A backend.
 *
 * The backend performs:
 *   1. URL normalisation
 *   2. 35-feature extraction (Groups A–E)
 *   3. LR inference (model_lr.joblib)
 *   4. Adaptive quantile threshold evaluation (three-phase enforcement)
 *   5. JSON response with p_phish, risk_level, decision, and threshold state
 */
async function callPredict(apiUrl, url, mode = "auto") {
  const endpoint = `${apiUrl.replace(/\/$/, "")}/predict`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body:JSON.stringify({ url, mode })
  });
  if (!response.ok) {
    throw new Error(`Backend returned HTTP ${response.status}`);
  }
  return await response.json();
}

// Badge helpers
/**
 * Risk level -> badge text mapping.
 * Short labels (≤4 chars) are required by Chrome's badge rendering.
 */
const BADGE_TEXT = {
  safe: "OK",
  low: "LOW",
  suspicious: "WARN",
  danger: "DANG"
};

/**
 * Risk level -> badge background colour.
 * Green = safe, amber = low, orange = suspicious, red = danger.
 */
const BADGE_COLOUR = {
  safe:       "#2e7d32",
  low:        "#f9a825",
  suspicious: "#ef6c00",
  danger:     "#c62828"
};

/**
 * Updates the extension badge for a given tab to reflect the scan result.
 * Guards against invalid tab IDs (< 0) which occur during manual scans.
 */
async function setBadge(tabId, riskLevel, decision, pPhish) {
  if (typeof tabId !== "number" || tabId < 0) return;

  const text = BADGE_TEXT[riskLevel]   || "";
  const colour = BADGE_COLOUR[riskLevel] || "#616161";
  const score = typeof pPhish === "number" ? (pPhish * 100).toFixed(1) + "%" : "-";
  const title = `Tower A: ${riskLevel} (${decision}) - p=${score}`;

  await chrome.action.setBadgeText({ tabId, text });
  await chrome.action.setBadgeBackgroundColor({ tabId, color: colour });
  await chrome.action.setTitle({ tabId, title });
}

/**
 * Clears the badge for a tab (no active scan result).
 */
async function clearBadge(tabId) {
  if (typeof tabId !== "number" || tabId < 0) return;
  await chrome.action.setBadgeText({ tabId, text: "" });
  await chrome.action.setTitle({ tabId, title: "Tower A: not scanned" });
}

/**
 * Sets an error badge when the backend is unreachable.
 */
async function setBadgeError(tabId) {
  if (typeof tabId !== "number" || tabId < 0) return;
  await chrome.action.setBadgeText({ tabId, text: "ERR" });
  await chrome.action.setBadgeBackgroundColor({ tabId, color: "#616161" });
  await chrome.action.setTitle({ tabId, title: "Tower A: backend unreachable" });
}

// Notification helper
/**
 * Shows a Chrome desktop notification when the risk level is warn or block.
 * Notifications are only shown if the decision is in the user's notifyOn list (default: ["warn", "block"]).
 */
async function maybeNotify(notifyOn, url, result) {
  const decision = result?.decision || "allow";
  if (!notifyOn.includes(decision)) return;

  const risk = result?.risk_level || "unknown";
  const score = typeof result?.p_phish === "number"
    ? ` (p=${(result.p_phish * 100).toFixed(1)}%)`
    : "";

  await chrome.notifications.create({
    type:    "basic",
    iconUrl: "icons/icon48.png",
    title:   `Phishing ${risk.toUpperCase()} - ${decision.toUpperCase()}`,
    message: `${url.slice(0, 80)}${score}`
  });
}

// Core scan function
/**
 * Performs the full scan pipeline for one URL in one tab:
 *   1. Skip non-scannable URLs silently.
 *   2. Call the Tower A backend.
 *   3. Update the badge.
 *   4. Store the result in tabState for the popup to read.
 *   5. Show a notification if the decision warrants one.
 *
 * All errors are caught and converted to an ERR badge - the browser experience is never interrupted by a backend failure.
 */
async function scanUrl(tabId, url, mode = "auto") {
  // Skip internal browser pages silently - no badge, no error.
  if (!isScannableUrl(url)) {
    const key = tabId >= 0 ? tabId : MANUAL_KEY;
    tabState.set(key, { lastUrl: url, lastResult: { skipped: true }, timerId: null });
    if (tabId >= 0) await clearBadge(tabId);
    return;
  }

  const stateKey = tabId >= 0 ? tabId : MANUAL_KEY;

  try {
    const { apiUrl, notifyOn } = await getSettings();
    const result = await callPredict(apiUrl, url, mode);

    // Store for popup.js to read via chrome.runtime.sendMessage
    tabState.set(stateKey, { lastUrl: url, lastResult: result, timerId: null });

    // Update badge (not applicable to manual key)
    if (tabId >= 0) {
      await setBadge(tabId, result.risk_level, result.decision, result.p_phish);
    }

    // Notify on high-risk decisions
    await maybeNotify(notifyOn, url, result);

  } catch (err) {
    // Backend unreachable or returned an error
    console.warn("[Tower A] Scan failed:", err.message);
    tabState.set(stateKey, {
      lastUrl: url,
      lastResult: { error: err.message },
      timerId: null
    });
    if (tabId >= 0) await setBadgeError(tabId);
  }
}

// Debounced navigation handler
/**
 * Schedules a scan for the given tab and URL after a debounce delay. If another navigation fires within the debounce
 * window (e.g., a redirect), the previous timer is cancelled and a new one is started.
 *
 * This prevents the backend being called multiple times for a single user-initiated navigation that results in one or
 * more intermediate redirects.
 */
function scheduleDebounced(tabId, url, debounceMs) {
  // Cancel any in-flight debounce timer for this tab
  const existing = tabState.get(tabId);
  if (existing?.timerId) {
    clearTimeout(existing.timerId);
  }

  // Schedule a new scan
  const timerId = setTimeout(() => {
    scanUrl(tabId, url, "auto");
  }, debounceMs);

  tabState.set(tabId, { lastUrl: url, lastResult: null, timerId });
}

// Event listeners
/**
 * webNavigation.onCommitted fires when the browser has committed to loading a new URL - i.e., the navigation will
 * definitely happen. This is preferred over onBeforeNavigate because the final URL is known at this point.
 *
 * Only scan main-frame navigations (frameId === 0) to avoid scoring iframes, CSS, JavaScript, and other sub-resources.
 */
chrome.webNavigation.onCommitted.addListener(async (details) => {
  // Only scan the top-level page (frameId 0 = main frame)
  if (details.frameId !== 0) return;

  const { debounceMs, autoScan } = await getSettings();
  if (!autoScan) return;

  scheduleDebounced(details.tabId, details.url, debounceMs);
});

/**
 * Listen for messages from popup.js.
 * Two message types are handled:
 *   - "getResult": popup is opening and wants the stored scan result for this tab.
 *   - "scanNow": user clicked "Scan Now" in the popup; trigger an immediate scan.
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

  if (message.type === "getResult") {
    // Popup is requesting the last stored result for its tab ID
    const entry = tabState.get(message.tabId);
    sendResponse({ result: entry?.lastResult || null, url: entry?.lastUrl || null });
    return true;  // keep channel open for async sendResponse
  }

  if (message.type === "scanNow") {
    // Immediate scan requested from the popup (no debounce)
    scanUrl(message.tabId, message.url, "manual").then(() => {
      const entry = tabState.get(message.tabId >= 0 ? message.tabId : MANUAL_KEY);
      sendResponse({ result: entry?.lastResult || null });
    });
    return true;  // async sendResponse
  }
});

/**
 * Clean up tabState when a tab is closed to prevent memory accumulation over long browser sessions.
 */
chrome.tabs.onRemoved.addListener((tabId) => {
  const entry = tabState.get(tabId);
  if (entry?.timerId) clearTimeout(entry.timerId);
  tabState.delete(tabId);
});
