/**
 * Prototype 3 - Fused System - service_worker.js
 *
 * MV3 service worker for the full two-tower extension.
 * Intercepts navigation events, calls the fused backend (port 8002), and updates the badge with the fused risk level.
 *
 * The badge title shows p_fused, p_A, and p_B so the individual tower contributions are visible without opening the popup.
 *
 * Latency: on first-visit domains the Tower B probe takes 2–8 seconds.
 * The badge is not updated until the full fused response is received - showing an intermediate Tower-A-only badge
 * before Tower B resolves would misrepresent the final risk level.
 */

const DEFAULTS = {
    apiUrl: "http://127.0.0.1:8002",
    debounceMs: 800,
    autoScan: true,
    notifyOn: ["warn", "block"]
};

const tabState = new Map();
let activeTabId = null;


function getSettings() {
    return new Promise((resolve) => {
        chrome.storage.sync.get(Object.keys(DEFAULTS), (stored) => {
            resolve({ ...DEFAULTS, ...stored });
        });
    });
}


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

async function callPredict(apiUrl, url, mode = "auto") {
    const response = await fetch(`${apiUrl.replace(/\/$/, "")}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, mode })
    });
    if (!response.ok) throw new Error(`Backend returned HTTP ${response.status}`);
    return await response.json();
}


// Badge
const BADGE_TEXT = {
    danger: "DANG", suspicious: "WARN", low: "LOW", safe: "OK", unknown: "?"
};

const BADGE_COLOUR = {
    danger: "#c62828", suspicious: "#ef6c00", low: "#f9a825",
    safe: "#2e7d32", unknown: "#616161"
};

async function setBadge(tabId, result) {
    if (typeof tabId !== "number" || tabId < 0) return;

    const risk = result?.risk_level || "unknown";
    const dec = result?.decision || "allow";
    const pF = typeof result?.p_fused === "number" ? (result.p_fused * 100).toFixed(1) + "%" : "-";
    const pA = typeof result?.p_a === "number" ? (result.p_a * 100).toFixed(1) + "%" : "-";
    const pB = typeof result?.p_b === "number" ? (result.p_b * 100).toFixed(1) + "%" : "n/a";

    await chrome.action.setBadgeText({ tabId, text: BADGE_TEXT[risk] || "" });
    await chrome.action.setBadgeBackgroundColor({ tabId, color: BADGE_COLOUR[risk] || "#616161" });
    await chrome.action.setTitle({ tabId, title: `Two-Tower: ${risk} (${dec}) | fused=${pF} A=${pA} B=${pB}` });
}

async function clearBadge(tabId) {
    if (typeof tabId !== "number" || tabId < 0) return;
    await chrome.action.setBadgeText({ tabId, text: "" });
    await chrome.action.setTitle({ tabId, title: "Two-Tower: not scanned" });
}

async function setBadgeError(tabId) {
    if (typeof tabId !== "number" || tabId < 0) return;
    await chrome.action.setBadgeText({ tabId, text: "ERR" });
    await chrome.action.setBadgeBackgroundColor({ tabId, color: "#616161" });
    await chrome.action.setTitle({ tabId, title: "Two-Tower: backend unreachable" });
}


// Notification
async function maybeNotify(notifyOn, url, result) {
    if (!notifyOn.includes(result?.decision)) return;
    const risk = result?.risk_level || "unknown";
    const score = typeof result?.p_fused === "number"
        ? ` (fused=${(result.p_fused * 100).toFixed(1)}%)` : "";
    let host = url;
    try { host = new URL(url).hostname; } catch (_) {}
    await chrome.notifications.create({
        type: "basic", iconUrl: "icons/icon48.png",
        title: `Two-Tower: ${risk.toUpperCase()} - ${(result?.decision || "").toUpperCase()}`,
        message: `${host}${score}`
    });
}


// Core scan
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
            await setBadge(tabId, result);
        } else if (tabId >= 0) {
            await clearBadge(tabId);
        }

        await maybeNotify(notifyOn, url, result);

    } catch (err) {
        console.warn("[Two-Tower] Scan failed:", err.message);
        tabState.set(tabId, { lastUrl: url, lastResult: { error: err.message }, timerId: null });
        if (tabId >= 0) await setBadgeError(tabId);
    }
}


// Debounce
function scheduleDebounced(tabId, url, debounceMs) {
    const existing = tabState.get(tabId);
    if (existing?.timerId) clearTimeout(existing.timerId);
    const timerId = setTimeout(() => scanUrl(tabId, url, "auto"), debounceMs);
    tabState.set(tabId, { lastUrl: url, lastResult: null, timerId });
}


// Event listeners
chrome.webNavigation.onCommitted.addListener(async (details) => {
    if (details.frameId !== 0) return;
    activeTabId = details.tabId;
    const { debounceMs, autoScan } = await getSettings();
    if (autoScan) scheduleDebounced(details.tabId, details.url, debounceMs);
});

chrome.tabs.onActivated.addListener(({ tabId }) => { activeTabId = tabId; });

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "getResult") {
        const entry = tabState.get(message.tabId);
        sendResponse({ result: entry?.lastResult || null, url: entry?.lastUrl || null });
        return true;
    }
    if (message.type === "scanNow") {
        scanUrl(message.tabId, message.url, "manual").then(() => {
            const entry = tabState.get(message.tabId);
            sendResponse({ result: entry?.lastResult || null });
        });
        return true;
    }
});

chrome.tabs.onRemoved.addListener((tabId) => {
    const entry = tabState.get(tabId);
    if (entry?.timerId) clearTimeout(entry.timerId);
    tabState.delete(tabId);
});