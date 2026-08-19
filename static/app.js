/* ─── State ───────────────────────────────────────────────── */
const chatWindow  = document.getElementById('chatWindow');
const messages    = document.getElementById('messages');
const userInput   = document.getElementById('userInput');
const sendBtn     = document.getElementById('sendBtn');
const welcomeScr  = document.getElementById('welcomeScreen');
const statusDot   = document.getElementById('statusDot');
const voiceSelect = document.getElementById('voiceSelect');
const stopSpeechBtn = document.getElementById('stopSpeechBtn');
const voiceRate    = document.getElementById('voiceRate');
const rateValue   = document.getElementById('rateValue');
const sidebar     = document.getElementById('sidebar');
const voiceAssistantBtn = document.getElementById('voiceAssistantBtn');
const stopGenBtn  = document.getElementById('stopGenBtn');
const autoSpeakToggle = document.getElementById('autoSpeakToggle');
const voiceVisualizer = document.getElementById('voiceVisualizer');
const micBtn         = document.getElementById('micBtn');
const speakerToggleBtn = document.getElementById('speakerToggleBtn');
const speakerOnIcon  = document.getElementById('speakerOnIcon');
const speakerOffIcon = document.getElementById('speakerOffIcon');

let isWaiting = false;
let modelReady = false;
let isSpeaking = false;
let assistantActive = false;
let recognition = null;
let conversation = [];
let currentAbort = null;
let currentRequestId = null;
let currentStreamingText = "";
let autoSpeak = false;
let voiceArmed = false;
let voiceArmedTimer = null;
let browserVoices = [];
let synth = window.speechSynthesis;

function loadPrefs() {
  try {
    autoSpeak = localStorage.getItem('autoSpeak') === '1';
  } catch {}
  if (autoSpeakToggle) autoSpeakToggle.checked = autoSpeak;
}
function savePrefs() {
  try { localStorage.setItem('autoSpeak', autoSpeak ? '1' : '0'); } catch {}
}
loadPrefs();

function updateSpeakerUI() {
  if (autoSpeak) {
    speakerOnIcon.classList.remove('hidden');
    speakerOffIcon.classList.add('hidden');
    speakerToggleBtn.classList.add('active');
  } else {
    speakerOnIcon.classList.add('hidden');
    speakerOffIcon.classList.remove('hidden');
    speakerToggleBtn.classList.remove('active');
  }
  if (autoSpeakToggle) autoSpeakToggle.checked = autoSpeak;
}
updateSpeakerUI();

if (autoSpeakToggle) {
  autoSpeakToggle.addEventListener('change', () => {
    autoSpeak = !!autoSpeakToggle.checked;
    savePrefs();
    updateSpeakerUI();
  });
}

if (speakerToggleBtn) {
  speakerToggleBtn.addEventListener('click', () => {
    autoSpeak = !autoSpeak;
    savePrefs();
    updateSpeakerUI();
    if (!autoSpeak) stopSpeech();
  });
}

/* ─── Model Status Poll ───────────────────────────────────── */
async function checkModelStatus() {
  try {
    const r = await fetch('/health');
    const d = await r.json();
    if (d.model_loaded) {
      modelReady = true;
      statusDot.className = 'status-dot ready';
      statusDot.title = 'Model ready';
      userInput.placeholder = 'Ask Ting Ling Ling anything...';
    } else {
      statusDot.className = 'status-dot loading';
      statusDot.title = 'Loading model...';
      userInput.placeholder = 'Model loading, please wait...';
      setTimeout(checkModelStatus, 2500);
    }
  } catch {
    setTimeout(checkModelStatus, 3000);
  }
}
checkModelStatus();

/* ─── Sidebar Toggle ──────────────────────────────────────── */
document.getElementById('toggleSidebar').addEventListener('click', () => {
  sidebar.classList.toggle('collapsed');
});
document.getElementById('newChatBtn').addEventListener('click', () => {
  messages.innerHTML = '';
  welcomeScr.classList.remove('hidden');
  userInput.value = '';
  conversation = [];
  autoResize();
});

/* ─── Topic Buttons ───────────────────────────────────────── */
document.querySelectorAll('.topic-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.topic-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    userInput.value = btn.dataset.q;
    autoResize();
    sendMessage();
  });
});

/* ─── Send on Enter (Shift+Enter for newline) ─────────────── */
userInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
sendBtn.addEventListener('click', sendMessage);

/* ─── Auto-resize textarea ────────────────────────────────── */
function autoResize() {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 180) + 'px';
  
  // Toggle send button state
  if (sendBtn) {
    const hasContent = userInput.value.trim().length > 0;
    sendBtn.disabled = !hasContent || isWaiting;
  }
}
userInput.addEventListener('input', () => {
  if (isSpeaking || (synth && synth.speaking)) {
    stopSpeech();
  }
  autoResize();
});
// Initialize button state
if (sendBtn) sendBtn.disabled = true;

/* ─── Rate Slider ─────────────────────────────────────────── */
voiceRate.addEventListener('input', () => {
  rateValue.textContent = voiceRate.value;
});

/* ─── Stop Speech ─────────────────────────────────────────── */
/* ─── TTS Logic (Browser-side) ────────────────────────────── */
function initSpeechSynthesis() {
  if (!synth) return;

  function loadVoices() {
    browserVoices = synth.getVoices();
    voiceSelect.innerHTML = '';
    
    // Filter for common high-quality voices or just show all
    browserVoices.forEach((voice, i) => {
      const option = document.createElement('option');
      option.value = i;
      option.textContent = `${voice.name} (${voice.lang})`;
      if (voice.default) option.selected = true;
      voiceSelect.appendChild(option);
    });
  }

  loadVoices();
  if (synth.onvoiceschanged !== undefined) {
    synth.onvoiceschanged = loadVoices;
  }
}
initSpeechSynthesis();

function setVisualizer(active) {
  if (active) {
    voiceVisualizer.classList.add('active');
    statusDot.classList.add('speaking');
  } else {
    voiceVisualizer.classList.remove('active');
    statusDot.classList.remove('speaking');
  }
}

async function stopSpeech() {
  if (synth) {
    synth.cancel();
  }
  isSpeaking = false;
  if (stopSpeechBtn) stopSpeechBtn.classList.add('hidden');
  setVisualizer(false);
  try {
    fetch('/stop', { method: 'POST' });
  } catch (e) {}
}
stopSpeechBtn.addEventListener('click', stopSpeech);

/* ─── Voice Interaction (STT) ──────────────────────────────── */
function initVoiceAssistant() {
  if (!('webkitSpeechRecognition' in window)) {
    if (voiceAssistantBtn) voiceAssistantBtn.style.display = 'none';
    if (micBtn) micBtn.style.display = 'none';
    return;
  }
  recognition = new webkitSpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    statusDot.classList.add('listening');
    if (voiceAssistantBtn) voiceAssistantBtn.classList.add('listening');
    if (micBtn) micBtn.classList.add('listening');
    setVisualizer(true);
  };

  recognition.onspeechstart = () => {
    // Cut off AI speech immediately when user starts speaking
    stopSpeech();
  };

  recognition.onsoundstart = () => {
    if (isSpeaking || (synth && synth.speaking)) {
      stopSpeech();
    }
  };
  
  recognition.onend = () => { 
    statusDot.classList.remove('listening');
    if (voiceAssistantBtn) voiceAssistantBtn.classList.remove('listening');
    if (micBtn) micBtn.classList.remove('listening');
    setVisualizer(false);
    
    // Auto-restart if the global assistant is active
    if (assistantActive) {
      try { recognition.start(); } catch {}
    }
  };

  recognition.onerror = (event) => {
    console.error('Speech recognition error:', event.error);
    statusDot.classList.remove('listening');
    if (voiceAssistantBtn) voiceAssistantBtn.classList.remove('listening');
    if (micBtn) micBtn.classList.remove('listening');
    setVisualizer(false);
    
    if (event.error === 'not-allowed') {
      alert('Microphone access denied. Please enable microphone permissions in your browser.');
    }
  };

  recognition.onresult = (event) => {
    let interimTranscript = '';
    let finalTranscript = '';

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript;
      } else {
        interimTranscript += event.results[i][0].transcript;
      }
    }

    // Stop AI speech immediately if user speech is detected
    if (interimTranscript || finalTranscript) {
      if (isSpeaking || (synth && synth.speaking)) {
        stopSpeech();
      }
    }

    if (finalTranscript) {
      const result = finalTranscript.toLowerCase().trim();
      
      // Wake phrase logic (only for global assistant)
      if (assistantActive && !voiceArmed && !isWaiting && !isSpeaking) {
        if (result.includes('hi ting ling ling') || result.includes('ting ling ling')) {
          voiceArmed = true;
          if (voiceArmedTimer) clearTimeout(voiceArmedTimer);
          voiceArmedTimer = setTimeout(() => { voiceArmed = false; }, 8000);
          appendMessage('ai', "Listening. Say your question.", 'local');
          if (autoSpeak) speakNow("I'm listening. How can I help?");
          return;
        }
      }

      // If we are actively listening (via micBtn or armed assistant)
      if ((micBtn && micBtn.classList.contains('listening')) || voiceArmed) {
        userInput.value = finalTranscript;
        autoResize();
        
        // One-shot listening: stop after getting a final result
        if (micBtn && micBtn.classList.contains('listening')) {
          recognition.stop(); // Stop first
          setTimeout(() => { sendMessage(); }, 100); // Small delay to let it stop
        } else if (voiceArmed) {
          voiceArmed = false;
          sendMessage();
        }
      }
    } else if (interimTranscript && ((micBtn && micBtn.classList.contains('listening')) || voiceArmed)) {
      userInput.value = interimTranscript;
      autoResize();
    }
  };
}

function toggleMic() {
  stopSpeech();
  if (!recognition) initVoiceAssistant();
  
  if (micBtn.classList.contains('listening')) {
    recognition.stop();
  } else {
    // If global assistant is on, stop it first to reset
    if (assistantActive) {
      assistantActive = false;
      recognition.stop();
      setTimeout(() => {
        recognition.start();
      }, 100);
    } else {
      recognition.start();
    }
  }
}

if (micBtn) micBtn.addEventListener('click', toggleMic);

function toggleVoiceAssistant() {
  stopSpeech();
  if (!recognition) initVoiceAssistant();
  assistantActive = !assistantActive;
  if (assistantActive) {
    if (voiceAssistantBtn) voiceAssistantBtn.classList.add('active');
    recognition.start();
  } else {
    if (voiceAssistantBtn) voiceAssistantBtn.classList.remove('active');
    recognition.stop();
  }
}
if (voiceAssistantBtn) voiceAssistantBtn.addEventListener('click', toggleVoiceAssistant);

function playStatusSound(type) {
  statusDot.classList.add('loading');
  setTimeout(() => statusDot.classList.remove('loading'), 1000);
}

function sendSuggestion(text) {
  userInput.value = text;
  autoResize();
  sendMessage();
}
window.sendSuggestion = sendSuggestion;

/* ─── Main Send Logic ─────────────────────────────────────── */
async function sendMessage() {
  stopSpeech();
  const text = userInput.value.trim();

  if (!text || isWaiting) return;

  welcomeScr.classList.add('hidden');
  appendMessage('user', text);
  conversation.push({ role: 'user', content: text });
  userInput.value = '';
  autoResize();

  const typingEl = appendTyping();
  isWaiting = true;
  sendBtn.disabled = true;
  if (stopGenBtn) stopGenBtn.classList.remove('hidden');

  const brain_mode = 'cloud';

  try {
    // Prefer streaming if the browser supports it.
    if (window.ReadableStream) {
      currentStreamingText = "";
      currentRequestId = null;
      if (currentAbort) {
        try { currentAbort.abort(); } catch {}
      }
      currentAbort = new AbortController();

      // Replace typing indicator with a live-updating AI message.
      typingEl.remove();
      const aiEl = appendStreamingAI();

      const res = await fetch('/chat_stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          brain_mode: brain_mode,
          history: conversation.slice(-12)
        }),
        signal: currentAbort.signal
      });

      if (!res.ok || !res.body) throw new Error('Streaming failed');

      await readSSE(res.body, async (msg) => {
        if (msg.type === 'meta') {
          currentRequestId = msg.request_id || null;
          if (msg.source) setStreamingSource(aiEl, msg.source);
          return;
        }
        if (msg.type === 'chunk') {
          currentStreamingText += (msg.text || "");
          setStreamingText(aiEl, currentStreamingText);
          return;
        }
        if (msg.type === 'done') {
          const finalSource = msg.source || null;
          finalizeStreamingAI(aiEl, currentStreamingText, finalSource);
          conversation.push({ role: 'assistant', content: currentStreamingText });
          if (autoSpeak && currentStreamingText) {
            await speakNow(currentStreamingText);
          }
          return;
        }
        if (msg.type === 'error') {
          finalizeStreamingAI(aiEl, msg.error || 'Error', 'error');
          conversation.push({ role: 'assistant', content: msg.error || 'Error' });
        }
      });

      scrollBottom();
      return;
    }

    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        message: text,
        brain_mode: brain_mode,
        history: conversation.slice(-12)
      })
    });
    const data = await res.json();
    typingEl.remove();

    const reply = data.reply || data.error || 'Sorry, something went wrong.';
    appendMessage('ai', reply, data.source);
    conversation.push({ role: 'assistant', content: reply });
    if (autoSpeak) speakNow(reply);
    scrollBottom();
  } catch (err) {
    if (err && err.name === 'AbortError') {
      appendMessage('ai', '(Stopped.)', 'local');
      conversation.push({ role: 'assistant', content: '(Stopped.)' });
      return;
    }
    try { typingEl.remove(); } catch {}
    appendMessage('ai', 'Error: Could not reach the server.');
  } finally {
    isWaiting = false;
    if (sendBtn) {
      sendBtn.disabled = userInput.value.trim().length === 0;
    }
    if (stopGenBtn) stopGenBtn.classList.add('hidden');
  }
}

async function readSSE(body, onMessage) {
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const lines = raw.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const payload = line.slice(6);
          try { onMessage(JSON.parse(payload)); } catch {}
        }
      }
    }
  }
}

async function stopGenerating() {
  if (!isWaiting) return;
  if (currentAbort) {
    try { currentAbort.abort(); } catch {}
  }
  if (currentRequestId) {
    try {
      await fetch('/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ request_id: currentRequestId })
      });
    } catch {}
  }
  if (stopGenBtn) stopGenBtn.classList.add('hidden');
}
if (stopGenBtn) stopGenBtn.addEventListener('click', stopGenerating);

/* ─── Markdown & Code Rendering ───────────────────────────── */
function renderMarkdown(text) {
  if (!text) return "";
  
  const placeholders = [];
  
  // 1. Protect Code Blocks
  text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (match) => {
    const id = `__CODE_BLOCK_${placeholders.length}__`;
    placeholders.push({ id, type: 'code', content: match });
    return id;
  });

  // 2. Protect Block Math \[ ... \]
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, (match, math) => {
    const id = `__BLOCK_MATH_${placeholders.length}__`;
    placeholders.push({ id, type: 'math', content: `<div class="math-block-render" data-expr="${escapeAttr(math.trim())}"></div>` });
    return id;
  });

  // 3. Protect Inline Math \( ... \)
  text = text.replace(/\\\(([\s\S]*?)\\\)/g, (match, math) => {
    const id = `__INLINE_MATH_${placeholders.length}__`;
    placeholders.push({ id, type: 'math', content: `<span class="math-inline-render" data-expr="${escapeAttr(math.trim())}"></span>` });
    return id;
  });

  // 4. Do general Markdown (Bold, Italic, etc.)
  let html = text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^[•\-\*] (.+)$/gm, '<li>$1</li>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br/>');

  // 5. Restore placeholders
  placeholders.forEach(p => {
    let content = p.content;
    if (p.type === 'code') {
      content = content.replace(/```(\w*)\n?([\s\S]*?)```/g, (m, lang, code) => {
        const language = lang.trim().toLowerCase() || 'code';
        const cleanCode = code.trim();
        if (language === 'chartjson') {
          return `<div class="chart-container-wrapper" style="position:relative; width:100%; max-width:640px; margin:16px 0; background:rgba(15,23,42,0.8); padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,0.1);"><canvas class="interactive-chart-canvas" data-config="${escapeAttr(cleanCode)}"></canvas></div>`;
        }
        if (language === 'mermaid') {
          return `<div class="mermaid-diagram-wrapper" style="margin:16px 0; background:rgba(15,23,42,0.8); padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,0.1);"><div class="mermaid">${escapeHtml(cleanCode)}</div></div>`;
        }
        return `<div class="code-wrapper">
                  <div class="code-header">
                    <span>${language}</span>
                    <button class="code-copy-btn" onclick="copyRawCode(this)">Copy</button>
                  </div>
                  <pre data-lang="${language}"><code class="language-${language}">${escapeHtml(cleanCode)}</code></pre>
                </div>`;
      });
    }
    html = html.replace(p.id, content);
  });

  return `<p>${html}</p>`;
}

function copyRawCode(btn) {
  const wrapper = btn.closest('.code-wrapper');
  const code = wrapper.querySelector('code').textContent;
  navigator.clipboard.writeText(code).then(() => {
    btn.textContent = 'Copied!';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.textContent = 'Copy';
      btn.classList.remove('copied');
    }, 2000);
  });
}
window.copyRawCode = copyRawCode;

function appendMessage(role, text) {
  const el = document.createElement('div');
  el.className = `message ${role}`;

  if (role === 'user') {
    el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  } else {
    const rendered = renderMarkdown(text);
    el.innerHTML = `
      <div class="ai-avatar">TLL</div>
      <div class="ai-content">
        <div class="ai-text">${rendered}${renderManusPreviewBtn(text)}</div>
        <div class="ai-actions">
          <button class="action-btn" onclick="copyText(this)" data-text="${escapeAttr(text)}" title="Copy">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            <span>Copy</span>
          </button>
          <button class="action-btn" onclick="speakText(this)" data-text="${escapeAttr(text)}" title="Speak">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
            <span>Speak</span>
          </button>
        </div>
      </div>`;
  }
  
  messages.appendChild(el);
  
  // Apply Prism highlighting
  if (window.Prism) {
    Prism.highlightAllUnder(el);
  }

  // Apply KaTeX Manually
  if (window.katex) {
    el.querySelectorAll('.math-block-render').forEach(m => {
      try { katex.render(m.dataset.expr, m, { displayMode: true, throwOnError: false }); } catch(e) { console.error(e); }
    });
    el.querySelectorAll('.math-inline-render').forEach(m => {
      try { katex.render(m.dataset.expr, m, { displayMode: false, throwOnError: false }); } catch(e) { console.error(e); }
    });
  }

  // Apply Visualizations (Charts & Mermaid Diagrams)
  renderVisualizations(el);

  scrollBottom();
  return el;
}

function appendStreamingAI(source = null) {
  const el = document.createElement('div');
  el.className = 'message ai';
  const sourceBadge = source ? `<span class="source-tag ${source.toLowerCase()}">${source}</span>` : '';
  el.innerHTML = `
    <div class="ai-avatar">TLL</div>
    <div class="ai-content">
      <div class="ai-text"><span class="streaming-text"></span>${sourceBadge}</div>
      <div class="ai-actions">
        <button class="action-btn" onclick="copyText(this)" data-text="" title="Copy">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
          <span>Copy</span>
        </button>
        <button class="action-btn" onclick="speakText(this)" data-text="" title="Speak">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
          <span>Speak</span>
        </button>
      </div>
    </div>`;
  messages.appendChild(el);
  scrollBottom();
  return el;
}

function setStreamingText(el, text) {
  const span = el.querySelector('.streaming-text');
  if (span) span.textContent = text;
  // Keep buttons in sync with the latest text.
  el.querySelectorAll('.action-btn').forEach(btn => { btn.dataset.text = text; });
  scrollBottom();
}

function setStreamingSource(el, source) {
  const aiText = el.querySelector('.ai-text');
  if (!aiText) return;
  const existing = aiText.querySelector('.source-tag');
  if (existing) existing.remove();
  if (!source) return;
  const badge = document.createElement('span');
  badge.className = `source-tag ${String(source).toLowerCase()}`;
  badge.textContent = source;
  aiText.appendChild(badge);
}

function finalizeStreamingAI(el, text, source = null) {
  const aiText = el.querySelector('.ai-text');
  if (!aiText) return;
  const rendered = renderMarkdown(text || '');
  const sourceBadge = source ? `<span class="source-tag ${String(source).toLowerCase()}">${source}</span>` : '';
  const previewBtn = renderManusPreviewBtn(text || '');
  aiText.innerHTML = `${rendered}${sourceBadge}${previewBtn}`;

  // Re-apply highlight and math after final render.
  if (window.Prism) {
    Prism.highlightAllUnder(el);
  }
  if (window.katex) {
    el.querySelectorAll('.math-block-render').forEach(m => {
      try { katex.render(m.dataset.expr, m, { displayMode: true, throwOnError: false }); } catch(e) { console.error(e); }
    });
    el.querySelectorAll('.math-inline-render').forEach(m => {
      try { katex.render(m.dataset.expr, m, { displayMode: false, throwOnError: false }); } catch(e) { console.error(e); }
    });
  }

  // Apply Visualizations (Charts & Mermaid Diagrams)
  renderVisualizations(el);

  el.querySelectorAll('.action-btn').forEach(btn => { btn.dataset.text = text || ''; });
}

/* ─── Manus AI Live UI Sandbox Modal Helper ─────────────────── */
function renderManusPreviewBtn(text) {
  if (!text) return '';
  const match = text.match(/generated_uis\/([a-zA-Z0-9_\-]+)/i) || text.match(/Project:\s*['"]?([a-zA-Z0-9_\-]+)['"]?/i);
  if (!match) return '';
  const slug = match[1];
  return `<br/><button class="manus-preview-btn" onclick="openManusModal('/generated_uis/${slug}/index.html', '${slug}')">✨ Open Manus AI Live UI Sandbox</button>`;
}

function openManusModal(url, projectTitle) {
  const modal = document.getElementById('manusUiModal');
  const frame = document.getElementById('manusUiFrame');
  const title = document.getElementById('manusUiProjectTitle');
  const extLink = document.getElementById('manusUiExternalLink');

  if (title) title.textContent = (projectTitle || 'Manus AI Live Preview').replace('_', ' ').toUpperCase();
  if (frame) frame.src = url;
  if (extLink) extLink.href = url;
  if (modal) modal.classList.remove('hidden');
}

function closeManusModal() {
  const modal = document.getElementById('manusUiModal');
  const frame = document.getElementById('manusUiFrame');
  if (modal) modal.classList.add('hidden');
  if (frame) frame.src = 'about:blank';
}
window.openManusModal = openManusModal;
window.closeManusModal = closeManusModal;

/* ─── Interactive Charts & Mermaid Visualization Renderer ───── */
function renderVisualizations(el) {
  if (!el) return;
  if (window.Chart) {
    el.querySelectorAll('.interactive-chart-canvas').forEach(canvas => {
      if (canvas.dataset.rendered) return;
      canvas.dataset.rendered = "true";
      try {
        const configStr = canvas.dataset.config;
        if (configStr) {
          const config = JSON.parse(configStr);
          new Chart(canvas.getContext('2d'), config);
        }
      } catch(e) {
        console.error("Chart render error:", e);
      }
    });
  }
  if (window.mermaid) {
    try {
      const nodes = el.querySelectorAll('.mermaid');
      if (nodes && nodes.length > 0) {
        mermaid.init(undefined, nodes);
      }
    } catch(e) {
      console.error("Mermaid render error:", e);
    }
  }
}
window.renderVisualizations = renderVisualizations;

function appendTyping() {
  const el = document.createElement('div');
  el.className = 'message ai';
  el.innerHTML = `<div class="ai-avatar">TLL</div><div class="ai-content"><div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>`;
  messages.appendChild(el);
  scrollBottom();
  return el;
}

function copyText(btn) {
  const text = btn.dataset.text;
  navigator.clipboard.writeText(text).then(() => {
    const originalHTML = btn.innerHTML;
    btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';
    setTimeout(() => { btn.innerHTML = originalHTML; }, 2000);
  });
}
window.copyText = copyText;

async function speakText(btn) {
  const text = btn.dataset.text;
  speakNow(text);
}
window.speakText = speakText;

function speakNow(text) {
  if (!synth || !text) return;
  
  // Stop current speech
  synth.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  const voiceIdx = voiceSelect.value;
  
  if (browserVoices[voiceIdx]) {
    utterance.voice = browserVoices[voiceIdx];
  }
  
  // Speed mapping: macOS 'say' (100-350) -> Browser rate (0.1-10)
  // 175 is roughly 1.0 rate
  utterance.rate = parseFloat(voiceRate.value) / 175;
  utterance.pitch = 1.0;
  utterance.volume = 1.0;

  utterance.onstart = () => {
    isSpeaking = true;
    stopSpeechBtn.classList.remove('hidden');
    setVisualizer(true);
  };

  utterance.onend = () => {
    isSpeaking = false;
    stopSpeechBtn.classList.add('hidden');
    setVisualizer(false);
  };

  utterance.onerror = () => {
    isSpeaking = false;
    stopSpeechBtn.classList.add('hidden');
    setVisualizer(false);
  };

  synth.speak(utterance);
}

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function escapeAttr(t) {
  return t.replace(/"/g,'&quot;').replace(/'/g,'&#39;').replace(/\n/g,' ');
}
function scrollBottom() {
  chatWindow.scrollTo({ top: chatWindow.scrollHeight, behavior: 'smooth' });
}

/* ─── Themes & Features Logic ──────────────────────────────── */
const themeToggleBtn = document.getElementById('themeToggleBtn');
const themes = ['default', 'theme-light', 'theme-cyberpunk'];
let currentThemeIdx = 0;

try {
  const savedTheme = localStorage.getItem('themePreference');
  if (savedTheme) {
    document.body.className = savedTheme;
    if (savedTheme === 'theme-light') {
      document.body.setAttribute('data-theme', 'light');
    }
    currentThemeIdx = themes.indexOf(savedTheme) !== -1 ? themes.indexOf(savedTheme) : 0;
  }
} catch {}

if (themeToggleBtn) {
  themeToggleBtn.addEventListener('click', () => {
    currentThemeIdx = (currentThemeIdx + 1) % themes.length;
    const newTheme = themes[currentThemeIdx];
    document.body.className = newTheme === 'default' ? '' : newTheme;
    if (newTheme === 'theme-light') {
      document.body.setAttribute('data-theme', 'light');
    } else {
      document.body.removeAttribute('data-theme');
    }
    try { localStorage.setItem('themePreference', document.body.className); } catch {}
  });
}

/* ─── Export & Import Chat ────────────────────────────────── */
const exportChatBtn = document.getElementById('exportChatBtn');
const importChatBtn = document.getElementById('importChatBtn');
const importFileInput = document.getElementById('importFileInput');

if (exportChatBtn) {
  exportChatBtn.addEventListener('click', async () => {
    if (conversation.length === 0) {
      alert("No messages to export yet.");
      return;
    }
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify({
      title: "Ting Ling Ling Chat Session",
      timestamp: new Date().toISOString(),
      messages: conversation
    }, null, 2));
    const dlAnchor = document.createElement('a');
    dlAnchor.setAttribute("href", dataStr);
    dlAnchor.setAttribute("download", `ting_ling_ling_chat_${Date.now()}.json`);
    document.body.appendChild(dlAnchor);
    dlAnchor.click();
    dlAnchor.remove();
  });
}

if (importChatBtn && importFileInput) {
  importChatBtn.addEventListener('click', () => importFileInput.click());
  importFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const json = JSON.parse(evt.target.result);
        const msgs = json.messages || json;
        if (Array.isArray(msgs)) {
          conversation = msgs;
          messages.innerHTML = '';
          welcomeScr.classList.add('hidden');
          msgs.forEach(m => {
            appendMessage(m.role === 'user' ? 'user' : 'assistant', m.content);
          });
          scrollBottom();
        } else {
          alert("Invalid chat file format.");
        }
      } catch (err) {
        alert("Failed to parse JSON file: " + err.message);
      }
    };
    reader.readAsText(file);
  });
}

/* ─── Settings Modal & Voices ─────────────────────────────── */
const settingsModal = document.getElementById('settingsModal');
const openSettingsBtn = document.getElementById('openSettingsBtn');
const modalSystemPrompt = document.getElementById('modalSystemPrompt');
const saveSystemPromptBtn = document.getElementById('saveSystemPromptBtn');
const modalVoiceSelect = document.getElementById('modalVoiceSelect');
const modalRateSlider = document.getElementById('modalRateSlider');
const modalRateValue = document.getElementById('modalRateValue');

function openSettingsModal() {
  if (settingsModal) {
    settingsModal.classList.remove('hidden');
    loadSystemPrompt();
    loadVoicesList();
  }
}
function closeSettingsModal() {
  if (settingsModal) settingsModal.classList.add('hidden');
}
window.closeSettingsModal = closeSettingsModal;

if (openSettingsBtn) openSettingsBtn.addEventListener('click', openSettingsModal);

async function loadSystemPrompt() {
  try {
    const res = await fetch('/api/system_prompt');
    const data = await res.json();
    if (modalSystemPrompt && data.system_prompt) {
      modalSystemPrompt.value = data.system_prompt;
    }
  } catch {}
}

if (saveSystemPromptBtn) {
  saveSystemPromptBtn.addEventListener('click', async () => {
    const promptText = modalSystemPrompt.value.trim();
    if (!promptText) return;
    try {
      const res = await fetch('/api/system_prompt', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ prompt: promptText })
      });
      const d = await res.json();
      if (d.status === 'ok') {
        alert("System Persona updated successfully!");
        closeSettingsModal();
      }
    } catch (e) {
      alert("Error updating prompt: " + e.message);
    }
  });
}

async function loadVoicesList() {
  try {
    const res = await fetch('/api/voices');
    const data = await res.json();
    if (modalVoiceSelect && data.voices) {
      modalVoiceSelect.innerHTML = '';
      data.voices.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        if (v.toLowerCase() === (data.current_voice || '').toLowerCase()) {
          opt.selected = true;
        }
        modalVoiceSelect.appendChild(opt);
      });
    }
  } catch {}
}

if (modalVoiceSelect) {
  modalVoiceSelect.addEventListener('change', async () => {
    const selectedVoice = modalVoiceSelect.value;
    fetch('/api/voice_settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ voice: selectedVoice })
    });
  });
}

if (modalRateSlider && modalRateValue) {
  modalRateSlider.addEventListener('input', () => {
    modalRateValue.textContent = modalRateSlider.value;
    fetch('/api/voice_settings', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ rate: parseInt(modalRateSlider.value) })
    });
  });
}

/* ─── Memory Inspector & Dataset Dashboard Logic ─────────── */
const memoryModal = document.getElementById('memoryModal');
const openMemoryBtn = document.getElementById('openMemoryBtn');
const memoryList = document.getElementById('memoryList');
const newMemoryKey = document.getElementById('newMemoryKey');
const newMemoryVal = document.getElementById('newMemoryVal');
const addMemoryBtn = document.getElementById('addMemoryBtn');

const trainingModal = document.getElementById('trainingModal');
const openTrainingBtn = document.getElementById('openTrainingBtn');
const sampleCountDisplay = document.getElementById('sampleCountDisplay');

function closeMemoryModal() {
  if (memoryModal) memoryModal.classList.add('hidden');
}
window.closeMemoryModal = closeMemoryModal;

function closeTrainingModal() {
  if (trainingModal) trainingModal.classList.add('hidden');
}
window.closeTrainingModal = closeTrainingModal;

if (openMemoryBtn) {
  openMemoryBtn.addEventListener('click', () => {
    if (memoryModal) {
      memoryModal.classList.remove('hidden');
      loadMemories();
    }
  });
}

if (openTrainingBtn) {
  openTrainingBtn.addEventListener('click', async () => {
    if (trainingModal) {
      trainingModal.classList.remove('hidden');
      try {
        const res = await fetch('/api/training_stats');
        const d = await res.json();
        if (sampleCountDisplay) sampleCountDisplay.textContent = d.sample_count || 0;
      } catch {}
    }
  });
}

async function loadMemories() {
  if (!memoryList) return;
  try {
    const res = await fetch('/api/memories');
    const d = await res.json();
    memoryList.innerHTML = '';
    const mems = d.memories || {};
    const keys = Object.keys(mems);
    if (keys.length === 0) {
      memoryList.innerHTML = '<div style="font-size:13px; color:var(--text-muted); padding:10px;">No memories stored yet.</div>';
      return;
    }
    keys.forEach(k => {
      const item = document.createElement('div');
      item.style.cssText = 'display:flex; justify-content:space-between; align-items:center; background:var(--glass-bg); padding:10px; border-radius:8px; margin-bottom:8px; border:1px solid var(--glass-border);';
      item.innerHTML = `
        <div style="font-size:13px; word-break:break-all;">
          <strong style="color:var(--accent);">${escapeHtml(k)}:</strong> ${escapeHtml(mems[k])}
        </div>
        <button class="manus-icon-btn" onclick="deleteMemoryKey('${escapeAttr(k)}')" title="Delete memory">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
        </button>
      `;
      memoryList.appendChild(item);
    });
  } catch {}
}

async function deleteMemoryKey(key) {
  try {
    await fetch('/api/memories', {
      method: 'DELETE',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ key })
    });
    loadMemories();
  } catch {}
}
window.deleteMemoryKey = deleteMemoryKey;

if (addMemoryBtn) {
  addMemoryBtn.addEventListener('click', async () => {
    const key = newMemoryKey.value.trim();
    const val = newMemoryVal.value.trim();
    if (!key || !val) return;
    try {
      await fetch('/api/memories', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ key, value: val })
      });
      newMemoryKey.value = '';
      newMemoryVal.value = '';
      loadMemories();
    } catch {}
  });
}

/* ─── Code Canvas & File Upload Handlers ─────────────────── */
const codeCanvasModal = document.getElementById('codeCanvasModal');
const openCodeCanvasBtn = document.getElementById('openCodeCanvasBtn');
const canvasCodeInput = document.getElementById('canvasCodeInput');
const canvasTerminalOutput = document.getElementById('canvasTerminalOutput');
const runCanvasCodeBtn = document.getElementById('runCanvasCodeBtn');

const attachFileBtn = document.getElementById('attachFileBtn');
const fileUploadInput = document.getElementById('fileUploadInput');

function closeCodeCanvasModal() {
  if (codeCanvasModal) codeCanvasModal.classList.add('hidden');
}
window.closeCodeCanvasModal = closeCodeCanvasModal;

if (openCodeCanvasBtn) {
  openCodeCanvasBtn.addEventListener('click', () => {
    if (codeCanvasModal) codeCanvasModal.classList.remove('hidden');
  });
}

if (runCanvasCodeBtn) {
  runCanvasCodeBtn.addEventListener('click', async () => {
    const code = canvasCodeInput.value.trim();
    if (!code) return;
    canvasTerminalOutput.textContent = 'Running code in Python sandbox...';
    try {
      const res = await fetch('/api/run_code', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ code })
      });
      const d = await res.json();
      canvasTerminalOutput.textContent = d.output || d.error || 'Execution finished.';
    } catch (e) {
      canvasTerminalOutput.textContent = 'Execution error: ' + e.message;
    }
  });
}

if (attachFileBtn && fileUploadInput) {
  attachFileBtn.addEventListener('click', () => fileUploadInput.click());
  fileUploadInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const d = await res.json();
      if (d.status === 'ok') {
        const attachNotice = `[Attached File: ${d.filename}]\nPath: ${d.saved_path}\n` + 
          (d.content_preview ? `Content Preview:\n\`\`\`\n${d.content_preview}\n\`\`\`\n` : '');
        userInput.value = (userInput.value ? userInput.value + '\n\n' : '') + attachNotice;
        userInput.focus();
      } else {
        alert("Upload failed: " + (d.error || 'Unknown error'));
      }
    } catch (err) {
      alert("File upload error: " + err.message);
    }
  });
}

/* ─── Hardware Health Polling & Prompt Polish ────────────── */
const sysCpuLabel = document.getElementById('sysCpuLabel');
const sysRamLabel = document.getElementById('sysRamLabel');
const enhancePromptBtn = document.getElementById('enhancePromptBtn');

async function updateSystemHealth() {
  try {
    const res = await fetch('/api/system_health');
    const d = await res.json();
    if (d.status === 'ok' && d.health) {
      if (sysCpuLabel && d.health.cpu_usage) sysCpuLabel.textContent = `CPU ${d.health.cpu_usage}`;
      if (sysRamLabel && d.health.ram_utilization) sysRamLabel.textContent = `RAM ${d.health.ram_utilization}`;
    }
  } catch {}
}
setInterval(updateSystemHealth, 5000);
updateSystemHealth();

if (enhancePromptBtn) {
  enhancePromptBtn.addEventListener('click', async () => {
    const promptText = userInput.value.trim();
    if (!promptText) return;
    try {
      const res = await fetch('/api/enhance_prompt', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ prompt: promptText })
      });
      const d = await res.json();
      if (d.status === 'ok' && d.enhanced) {
        userInput.value = d.enhanced;
        userInput.focus();
      }
    } catch {}
  });
}

/* ─── Visual Web Inspector Handlers ─────────────────────── */
const browserInspectorModal = document.getElementById('browserInspectorModal');
const inspectorUrlInput = document.getElementById('inspectorUrlInput');
const inspectUrlBtn = document.getElementById('inspectUrlBtn');
const inspectorContentOutput = document.getElementById('inspectorContentOutput');

function closeBrowserInspectorModal() {
  if (browserInspectorModal) browserInspectorModal.classList.add('hidden');
}
window.closeBrowserInspectorModal = closeBrowserInspectorModal;

if (inspectUrlBtn && inspectorUrlInput) {
  inspectUrlBtn.addEventListener('click', async () => {
    const url = inspectorUrlInput.value.trim();
    if (!url) return;
    inspectorContentOutput.textContent = `Browsing ${url}...`;
    try {
      const res = await fetch('/api/browse_page', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ url })
      });
      const d = await res.json();
      inspectorContentOutput.textContent = d.content || d.error || 'Failed to fetch content.';
    } catch (e) {
      inspectorContentOutput.textContent = 'Error fetching URL: ' + e.message;
    }
  });
}
