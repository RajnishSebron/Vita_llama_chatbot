// ================================================================
// VitaLlama 3.1 — Frontend Application Logic (Vanilla JS)
// ================================================================

// ── State ─────────────────────────────────────────────────────────
let API_BASE       = '';
let isConnected    = false;
let sessionId      = crypto.randomUUID();
let isStreaming    = false;
let abortCtrl      = null;

// DOM handles
const $apiUrl        = document.getElementById('apiUrl');
const $statusDot     = document.getElementById('statusDot');
const $btnConnect    = document.getElementById('btnConnect');
const $messagesArea  = document.getElementById('messagesArea');
const $welcomeView   = document.getElementById('welcomeView');
const $typingInd     = document.getElementById('typingIndicator');
const $userInput     = document.getElementById('userInput');
const $btnSend       = document.getElementById('btnSend');
const $sidebar       = document.getElementById('sidebar');

// Sidebar fields
const $sStatus      = document.getElementById('sidebarStatus');
const $sSessionId   = document.getElementById('sidebarSessionId');
const $sBadge       = document.getElementById('sidebarBadge');
const $sLatency     = document.getElementById('sidebarLatency');
const $sWindow      = document.getElementById('sidebarWindow');
const $sSummary     = document.getElementById('sidebarSummary');
const $sCache       = document.getElementById('sidebarCache');

// ── Connection Management ─────────────────────────────────────────
async function toggleConnection() {
  if (isConnected) {
    disconnect();
    return;
  }
  const raw = $apiUrl.value.trim().replace(/\/+$/, '');
  if (!raw) { $apiUrl.focus(); return; }
  API_BASE = raw;
  $btnConnect.textContent = 'Checking…';
  $btnConnect.disabled = true;

  try {
    const r = await fetch(`${API_BASE}/health`, {
      headers: { 'ngrok-skip-browser-warning': 'true' },
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    markConnected(data);
  } catch (err) {
    alert(`Connection failed:\n${err.message}\n\nMake sure the Colab notebook is running and the URL is correct.`);
    disconnect();
  }
}

function markConnected(data) {
  isConnected = true;
  $statusDot.classList.replace('offline', 'online');
  $btnConnect.textContent = 'Disconnect';
  $btnConnect.classList.replace('disconnected', 'connected');
  $btnConnect.disabled = false;
  $userInput.disabled = false;
  $btnSend.disabled   = false;
  $apiUrl.disabled    = true;
  $userInput.focus();

  $sStatus.innerHTML = '<span style="color:var(--accent-green)">● Connected</span>';
  $sSessionId.textContent = sessionId;
  if (data?.cache_stats) {
    updateCacheDisplay(data.cache_stats);
  }
}

function disconnect() {
  isConnected = false;
  API_BASE = '';
  $statusDot.classList.replace('online', 'offline');
  $btnConnect.textContent = 'Connect';
  $btnConnect.classList.replace('connected', 'disconnected');
  $btnConnect.disabled = false;
  $userInput.disabled = true;
  $btnSend.disabled   = true;
  $apiUrl.disabled    = false;

  $sStatus.innerHTML = '<span style="color:var(--accent-red)">● Disconnected</span>';
}

// ── Send Message & SSE Streaming ──────────────────────────────────
async function sendMessage() {
  if (isStreaming || !isConnected) return;
  const text = $userInput.value.trim();
  if (!text) return;

  // Hide welcome
  if ($welcomeView) $welcomeView.style.display = 'none';

  // Render user bubble
  appendMessage('user', text);
  $userInput.value = '';
  autoResize($userInput);

  isStreaming = true;
  $btnSend.disabled = true;
  $typingInd.classList.add('active');
  scrollToBottom();

  // Create bot bubble placeholder
  const { bubble, badgeEl } = appendMessage('bot', '', true);

  abortCtrl = new AbortController();
  let fullText  = '';
  let source    = 'gpu';
  let latencyMs = null;

  try {
    const resp = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true',
      },
      body: JSON.stringify({
        message:    text,
        session_id: sessionId,
        temperature: 0.3,
        max_tokens:  512,
      }),
      signal: abortCtrl.signal,
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    $typingInd.classList.remove('active');

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();    // keep incomplete line

      let eventType = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          const raw = line.slice(6);
          try {
            const payload = JSON.parse(raw);
            handleSSE(eventType, payload, bubble, badgeEl);
            if (eventType === 'token' && payload.token) {
              fullText += payload.token;
            }
            if (eventType === 'metadata') {
              source = payload.source || 'gpu';
              if (payload.clinical_summary) $sSummary.textContent = payload.clinical_summary;
              if (payload.window_turns != null) $sWindow.textContent = `${payload.window_turns} / 6 turns active`;
              if (payload.latency_ms != null) latencyMs = payload.latency_ms;
            }
            if (eventType === 'done') {
              if (payload.duration_ms)       latencyMs = payload.duration_ms;
              if (payload.latency_ms)        latencyMs = payload.latency_ms;
              if (payload.clinical_summary)  $sSummary.textContent = payload.clinical_summary;
              source = payload.source || source;
            }
          } catch (_) { /* ignore parse errors */ }
        }
      }
    }

    // Final UI updates
    updateBadge(source, latencyMs);
    if (badgeEl) {
      badgeEl.className = `source-badge ${source}`;
      badgeEl.textContent = source === 'cache' ? '⚡ Instant Cache' : '🧠 GPU Inference';
    }
    if (latencyMs != null) {
      $sLatency.textContent = `${latencyMs} ms`;
    }

    // Refresh cache stats
    refreshCacheStats();

  } catch (err) {
    if (err.name !== 'AbortError') {
      bubble.textContent = `⚠️ Error: ${err.message}`;
      $typingInd.classList.remove('active');
    }
  } finally {
    isStreaming = false;
    $btnSend.disabled = false;
    $userInput.focus();
    scrollToBottom();
  }
}

function handleSSE(event, payload, bubble, badgeEl) {
  switch (event) {
    case 'token':
      bubble.textContent += payload.token;
      scrollToBottom();
      break;
    case 'ttft':
      // time-to-first-token diagnostic
      break;
    case 'metadata':
    case 'done':
      break;
  }
}

// ── UI Helpers ────────────────────────────────────────────────────
function appendMessage(role, text, returnEls = false) {
  const msg = document.createElement('div');
  msg.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '👤' : '🌱';

  const bubbleWrap = document.createElement('div');
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  bubbleWrap.appendChild(bubble);

  let badgeEl = null;
  if (role === 'bot') {
    badgeEl = document.createElement('div');
    badgeEl.className = 'source-badge';
    badgeEl.style.display = text === '' ? 'none' : 'inline-block';
    bubbleWrap.appendChild(badgeEl);
  }

  msg.appendChild(avatar);
  msg.appendChild(bubbleWrap);
  $messagesArea.appendChild(msg);
  scrollToBottom();

  if (returnEls) {
    // When streaming finishes, show badge
    const observer = new MutationObserver(() => {
      if (bubble.textContent.length > 0 && badgeEl) badgeEl.style.display = 'inline-block';
    });
    observer.observe(bubble, { childList: true, characterData: true, subtree: true });
    return { bubble, badgeEl };
  }
}

function updateBadge(source, latency) {
  if (source === 'cache') {
    $sBadge.innerHTML = '<span class="inference-badge cache">⚡ Instant Cache Hit</span>';
  } else {
    $sBadge.innerHTML = '<span class="inference-badge gpu">🧠 GPU Inference</span>';
  }
}

function updateCacheDisplay(stats) {
  const ratio = stats.total > 0 ? Math.round(stats.hits / stats.total * 100) : 0;
  $sCache.textContent = `Queries: ${stats.total}  |  Hits: ${stats.hits}  |  Ratio: ${ratio}%`;
}

async function refreshCacheStats() {
  if (!isConnected) return;
  try {
    const r = await fetch(`${API_BASE}/cache/stats`, {
      headers: { 'ngrok-skip-browser-warning': 'true' },
    });
    if (r.ok) {
      const d = await r.json();
      const ratio = d.total > 0 ? Math.round(d.hits / d.total * 100) : 0;
      $sCache.textContent = `Queries: ${d.total}  |  Hits: ${d.hits}  |  Ratio: ${ratio}%`;
    }
  } catch (_) { /* silent */ }
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    $messagesArea.scrollTop = $messagesArea.scrollHeight;
  });
}

function fillExample(btn) {
  $userInput.value = btn.textContent;
  autoResize($userInput);
  $userInput.focus();
}

// ── Session Reset ─────────────────────────────────────────────────
async function resetSession() {
  if (!isConnected) return;
  try {
    await fetch(`${API_BASE}/session/${sessionId}/reset`, {
      method: 'POST',
      headers: { 'ngrok-skip-browser-warning': 'true' },
    });
  } catch (_) { /* ignore */ }

  // Generate fresh session
  sessionId = crypto.randomUUID();
  $sSessionId.textContent = sessionId;
  $sSummary.textContent = 'No prior clinical history recorded.';
  $sWindow.textContent = '0 / 6 turns active';
  $sLatency.textContent = '—';
  $sBadge.innerHTML = '<span class="inference-badge idle">⏸ Idle</span>';

  // Clear chat
  $messagesArea.innerHTML = '';
  $messagesArea.appendChild($welcomeView);
  $welcomeView.style.display = 'flex';
}

// ── Sidebar Toggle ────────────────────────────────────────────────
function toggleSidebar() {
  $sidebar.classList.toggle('collapsed');
}

// ── Textarea Auto-Resize & Keyboard ──────────────────────────────
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

$userInput.addEventListener('input', () => autoResize($userInput));
$userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Enter key on URL field triggers connect
$apiUrl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') toggleConnection();
});

// ── Persist URL in localStorage ───────────────────────────────────
const STORAGE_KEY = 'vitalLamaApiUrl';
const savedUrl = localStorage.getItem(STORAGE_KEY);
if (savedUrl) $apiUrl.value = savedUrl;

$apiUrl.addEventListener('change', () => {
  localStorage.setItem(STORAGE_KEY, $apiUrl.value.trim());
});

// ── Initialise Session ID Display ─────────────────────────────────
$sSessionId.textContent = sessionId;

