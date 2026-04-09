/**
 * Tower B - Prototype 2 - service_worker.js
 *
 * Purpose
 * The MV3 service worker for the Prototype 2 standalone Tower B extension. It intercepts browser navigation events, 
 * sends the URL to the Tower B FastAPI backend (port 8001), and updates the extension badge with the resulting risk level.
 *
 * Differences from Prototype 1 (Tower A)
 * The service worker logic is structurally identical to Prototype 1 but targets a different backend and response format:
 *   - Port 8001 (Tower B) instead of 8000 (Tower A).
 *   - The response field is p_b (not p_phish), reflecting that Tower B scores domains rather than full URLs.
 *   - The badge title shows the hostname that was scored, not the full URL, because Tower B operates at domain level only.
 *   - The debug block in the response (dns_ok, tls_ok, http_ok) is passed to the popup so the infrastructure signals 
 *   are visible to the user.
 *
 * Why Tower B operates on hostnames
 * Tower B's features (DNS resolution, TLS certificate validity, HTTP headers) are domain-level signals. Two different 
 * paths on the same domain share identical infrastructure and would receive the same score. The backend extracts the 
 * hostname internally, but the full URL is still sent so that the backend can log it and the popup can display it.
 *
 * Caching behaviour
 * The Tower B backend caches domain metadata in SQLite for 7 days. This means that a second navigation to the same 
 * domain returns a cached result almost instantly, whereas the first visit triggers live DNS, TLS, and HTTP probes 
 * that may take 2–8 seconds. The cache_hit field in the debug block is surfaced in the popup so this latency difference 
 * is understandable.
 *
 * Data flow
 *   chrome.webNavigation.onCommitted
 *     -> debounce (800 ms)
 *     -> isScannableUrl()
 *     -> POST /predict to Tower B backend (port 8001)
 *     -> setBadge()
 *     -> store result in tabState for popup
 *     -> (if warn/block) showNotification()
 */

// Defaults

const DEFAULTS = {
    apiUrl: "http://127.0.0.1:8001",  // Tower B backend - port 8001
    debounceMs: 800,
    autoScan: true,
    notifyOn: ["warn", "block"]
};

// In-memory state per tab, keyed by tab ID.
// Shape: Map<tabId, { lastUrl, lastResult, timerId }>
const tabState = new Map();

let activeTabId = null;


// Settings

function getSettings() {
    return new Promise((resolve) => {
        chrome.storage.sync.get(Object.keys(DEFAULTS), (stored) => {
            resolve({ ...DEFAULTS, ...stored });
        });
    });
}

// URL utility
/**
 * Returns true for http/https URLs that Tower B can probe.
 * Browser-internal schemes cannot be sent to the backend - they have no meaningful domain infrastructure to inspect.
 *
 * @param {string} url
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
 * Sends a POST /predict request to the Tower B backend.
 *
 * The backend extracts the hostname from the URL, checks the SQLite cache, and either returns a cached result or
 * performs live DNS/TLS/HTTP probing before scoring with the HGB model.
 *
 * @param {string} apiUrl  - Base URL of the Tower B backend.
 * @param {string} url     - Full URL from the navigation event.
 * @param {string} mode    - "auto" or "manual".
 * @returns {Promise<Object>} Parsed /predict JSON response.
 */
async function callPredict(apiUrl, url, mode = "auto") {
    const endpoint = `${apiUrl.replace(/\/$/, "")}/predict`;
    const response = await fetch(endpoint, {
        method: "POST",
        headers: {"Content-Type": "application/json" },
        body:JSON.stringify({ url, mode })
    });
    if (!response.ok) {
        throw new Error(`Backend returned HTTP ${response.status}`);
    }
    return await response.json();
}

// Badge helpers
const BADGE_TEXT = {
    danger: "DANG",
    suspicious: "WARN",
    low: "LOW",
    unknown: "?"
};

const BADGE_COLOUR = {
    danger: "#c62828",
    suspicious: "#ef6c00",
    low: "#f9a825",
    unknown: "#616161"
};

/**
 * Updates the badge for a tab with the Tower B risk level.
 * The badge title shows the hostname rather than the full URL because Tower B scores domains, not individual pages.
 *
 * @param {number} tabId
 * @param {string} riskLevel - danger / suspicious / low / unknown
 * @param {string} decision  - block / warn / allow
 * @param {number} pB        - p_b score in [0, 1]
 * @param {string} hostname  - The domain that was scored
 */
async function setBadge(tabId, riskLevel, decision, pB, hostname) {
    if (typeof tabId !== "number" || tabId < 0) return;

    const text = BADGE_TEXT[riskLevel] || "";
    const colour = BADGE_COLOUR[riskLevel] || "#616161";
    const score = typeof pB === "number" ? (pB * 100).toFixed(1) + "%" : "-";
    const title = `Tower B: ${riskLevel} (${decision}) - p=${score} - ${hostname || ""}`;

    await chrome.action.setBadgeText({ tabId, text });
    await chrome.action.setBadgeBackgroundColor({ tabId, color: colour });
    await chrome.action.setTitle({ tabId, title });
}

async function clearBadge(tabId) {
    if (typeof tabId !== "number" || tabId < 0) return;
    await chrome.action.setBadgeText({ tabId, text: "" });
    await chrome.action.setTitle({ tabId, title: "Tower B: not scanned" });
}

async function setBadgeError(tabId) {
    if (typeof tabId !== "number" || tabId < 0) return;
    await chrome.action.setBadgeText({ tabId, text: "ERR" });
    await chrome.action.setBadgeBackgroundColor({ tabId, color: "#616161" });
    await chrome.action.setTitle({ tabId, title: "Tower B: backend unreachable" });
}

// Notification helper
/**
 * Shows a desktop notification when the decision warrants one.
 * The notification body names the hostname rather than the full URL to reflect that the risk assessment is domain-level.
 *
 * @param {string[]} notifyOn - Decisions that trigger a notification.
 * @param {string} url     - Full URL (for display only).
 * @param {Object} result  - Full /predict response.
 */
async function maybeNotify(notifyOn, url, result) {
    const decision = result?.decision || "allow";
    if (!notifyOn.includes(decision)) return;

    const risk = result?.risk_level || "unknown";
    const hostname = result?.hostname || url;
    const score = typeof result?.p_b === "number"
        ? ` (p=${(result.p_b * 100).toFixed(1)}%)`
        : "";

    await chrome.notifications.create({
        type: "basic",
        iconUrl: "icons/icon48.png",
        title: `Tower B: ${risk.toUpperCase()} - ${decision.toUpperCase()}`,
        message: `${hostname}${score}`
    });
}

// Core scan function
/**
 * Performs the full Tower B scan pipeline for a URL in one tab.
 *
 * Note on latency: if this is the first visit to a domain, the backend must perform live DNS, TLS, and HTTP probes
 * before responding. This can take 2–8 seconds. Subsequent visits to the same domain return a cached result almost
 * immediately. The cache_hit field in the stored result is displayed in the popup to make this behaviour transparent.
 *
 * @param {number} tabId - Chrome tab ID.
 * @param {string} url   - URL to score.
 * @param {string} mode  - "auto" | "manual".
 */
async function scanUrl(tabId, url, mode = "auto") {
    if (!isScannableUrl(url)) {
        if (tabId >= 0) await clearBadge(tabId);
        tabState.set(tabId, { lastUrl: url, lastResult: { skipped: true }, timerId: null });
        return;
    }

    try {
        const { apiUrl, notifyOn } = await getSettings();
        const result = await callPredict(apiUrl, url, mode);

        tabState.set(tabId, { lastUrl: url, lastResult: result, timerId: null });

        if (tabId >= 0 && result?.ok && !result?.skipped) {
            await setBadge(tabId, result.risk_level, result.decision, result.p_b, result.hostname);
        } else if (tabId >= 0) {
            await clearBadge(tabId);
        }

        await maybeNotify(notifyOn, url, result);

    } catch (err) {
        console.warn("[Tower B] Scan failed:", err.message);
        tabState.set(tabId, { lastUrl: url, lastResult: { error: err.message }, timerId: null });
        if (tabId >= 0) await setBadgeError(tabId);
    }
}

// Debounced navigation handler
/**
 * Schedules a scan after a debounce delay. Rapid navigations (e.g., redirects during page load) cancel and restart
 * the timer so only one scan fires per user-initiated navigation.
 *
 * @param {number} tabId
 * @param {string} url
 * @param {number} debounceMs
 */
function scheduleDebounced(tabId, url, debounceMs) {
    const existing = tabState.get(tabId);
    if (existing?.timerId) clearTimeout(existing.timerId);

    const timerId = setTimeout(() => {
        scanUrl(tabId, url, "auto");
        }, debounceMs);

    tabState.set(tabId, { lastUrl: url, lastResult: null, timerId });
}

// Event listeners

// Scan when the browser commits to a new top-level navigation.
chrome.webNavigation.onCommitted.addListener(async (details) => {
    if (details.frameId !== 0) return; // main frame only

    activeTabId = details.tabId;
    const { debounceMs, autoScan } = await getSettings();
    if (!autoScan) return;

    scheduleDebounced(details.tabId, details.url, debounceMs);
});

// Track the active tab when the user switches tabs.
chrome.tabs.onActivated.addListener(({ tabId }) => {
    activeTabId = tabId;
});

// Message handler for popup.js communication.
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "getResult") {
        const entry = tabState.get(message.tabId);
        sendResponse({ result: entry?.lastResult || null, url: entry?.lastUrl || null });
        return true;
    }

    if (message.type === "scanNow") {
        scanUrl(message.tabId, message.url, "manual").then(() => {
            const key = message.tabId >= 0 ? message.tabId : "manual";
            const entry = tabState.get(key);
            sendResponse({ result: entry?.lastResult || null });
        });
        return true;
    }
});

// Clean up state when a tab is closed.
chrome.tabs.onRemoved.addListener((tabId) => {
    const entry = tabState.get(tabId);
    if (entry?.timerId) clearTimeout(entry.timerId);
    tabState.delete(tabId);
});
