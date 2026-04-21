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
const brainModeToggle = document.getElementById('brainModeToggle');
const modeLocalLabel = document.getElementById('modeLocalLabel');
const modeCloudLabel = document.getElementById('modeCloudLabel');

let isWaiting = false;
let modelReady = false;
let isSpeaking = false;
let assistantActive = false;
let recognition = null;
let conversation = [];

/* ─── Brain Mode Toggle ────────────────────────────────────── */
function updateModeLabels() {
  if (brainModeToggle.checked) {
    modeCloudLabel.classList.add('active');
    modeLocalLabel.classList.remove('active');
  } else {
    modeCloudLabel.classList.remove('active');
    modeLocalLabel.classList.add('active');
  }
}
brainModeToggle.addEventListener('change', updateModeLabels);
updateModeLabels();

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
async function stopSpeech() {
  try {
    await fetch('/stop', { method: 'POST' });
    isSpeaking = false;
    stopSpeechBtn.classList.add('hidden');
  } catch (err) {
    console.error('Error stopping speech:', err);
  }
}
stopSpeechBtn.addEventListener('click', stopSpeech);

/* ─── Voice Assistant ──────────────────────────────────────── */
function initVoiceAssistant() {
  if (!('webkitSpeechRecognition' in window)) {
    voiceAssistantBtn.style.display = 'none';
    return;
  }
  recognition = new webkitSpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onresult = (event) => {
    const result = event.results[event.results.length - 1][0].transcript.toLowerCase().trim();
    if (!isWaiting && !isSpeaking) {
      if (result.includes('hi ting ling ling') || result.includes('ting ling ling')) {
        playStatusSound('listen');
        appendMessage('ai', "I'm listening...");
      } else if (assistantActive) {
        userInput.value = result;
        sendMessage();
      }
    }
  };
  recognition.onend = () => { if (assistantActive) recognition.start(); };
}

function toggleVoiceAssistant() {
  if (!recognition) initVoiceAssistant();
  assistantActive = !assistantActive;
  if (assistantActive) {
    voiceAssistantBtn.classList.add('active');
    recognition.start();
  } else {
    voiceAssistantBtn.classList.remove('active');
    recognition.stop();
  }
}
voiceAssistantBtn.addEventListener('click', toggleVoiceAssistant);

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

  const brain_mode = brainModeToggle.checked ? 'cloud' : 'local';

  try {
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
    scrollBottom();
  } catch (err) {
    typingEl.remove();
    appendMessage('ai', '❌ Could not reach the server.');
  } finally {
    isWaiting = false;
    sendBtn.disabled = false;
  }
}

/* ─── Markdown & Code Rendering ───────────────────────────── */
function renderMarkdown(text) {
  // Pass 1: Code blocks with language detection
  text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (match, lang, code) => {
    const language = lang.trim() || 'python';
    return `<pre class="line-numbers"><code class="language-${language}">${escapeHtml(code.trim())}</code></pre>`;
  });

  // Pass 2: Bold, Italic, Inline Code
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^[•\-\*] (.+)$/gm, '<li>$1</li>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br/>');
}

function appendMessage(role, text, source = null) {
  const el = document.createElement('div');
  el.className = `message ${role}`;

  if (role === 'user') {
    el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  } else {
    const rendered = renderMarkdown(text);
    const sourceBadge = source ? `<span class="source-tag ${source.toLowerCase()}">${source}</span>` : '';
    el.innerHTML = `
      <div class="ai-avatar">TL</div>
      <div class="ai-content">
        <div class="ai-text">${rendered}${sourceBadge}</div>
        <div class="ai-actions">
          <button class="action-btn" onclick="copyText(this)" data-text="${escapeAttr(text)}">Copy</button>
          <button class="action-btn" onclick="speakText(this)" data-text="${escapeAttr(text)}">Speak</button>
        </div>
      </div>`;
  }
  
  messages.appendChild(el);
  
  // Apply Prism highlighting
  if (window.Prism) {
    Prism.highlightAllUnder(el);
  }

  // Apply KaTeX
  if (window.renderMathInElement) {
    renderMathInElement(el, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false}
      ],
      throwOnError : false
    });
  }

  scrollBottom();
  return el;
}

function appendTyping() {
  const el = document.createElement('div');
  el.className = 'message ai';
  el.innerHTML = `<div class="ai-avatar">TL</div><div class="ai-content"><div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>`;
  messages.appendChild(el);
  scrollBottom();
  return el;
}

function copyText(btn) {
  const text = btn.dataset.text;
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = '✓ Copied';
    setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
  });
}
window.copyText = copyText;

async function speakText(btn) {
  const text = btn.dataset.text;
  const voice = voiceSelect.value;
  const rate = voiceRate.value;
  isSpeaking = true;
  stopSpeechBtn.classList.remove('hidden');
  try {
    await fetch('/speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice, rate })
    });
  } finally {
    setTimeout(() => { isSpeaking = false; }, 2000);
  }
}
window.speakText = speakText;

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function escapeAttr(t) {
  return t.replace(/"/g,'&quot;').replace(/'/g,'&#39;').replace(/\n/g,' ');
}
function scrollBottom() {
  chatWindow.scrollTo({ top: chatWindow.scrollHeight, behavior: 'smooth' });
}
