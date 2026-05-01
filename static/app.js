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
}
userInput.addEventListener('input', autoResize);

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
    isSpeaking = false;
    stopSpeechBtn.classList.add('hidden');
    setVisualizer(false);
  }
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
    sendBtn.disabled = false;
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
        const language = lang.trim() || 'code';
        const cleanCode = code.trim();
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
        <div class="ai-text">${rendered}</div>
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
  aiText.innerHTML = `${rendered}${sourceBadge}`;

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

  el.querySelectorAll('.action-btn').forEach(btn => { btn.dataset.text = text || ''; });
}

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
