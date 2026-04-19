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

let isWaiting = false;
let modelReady = false;
let isSpeaking = false;
let assistantActive = false;
let recognition = null;

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

/* ─── Voice Assistant (Wake Word & Loop) ───────────────────── */
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
    console.log('[Voice] Heard:', result);

    if (!isWaiting && !isSpeaking) {
      if (result.includes('hi ting ling ling') || result.includes('ting ling ling')) {
        // Wake word detected!
        playStatusSound('listen');
        appendMessage('ai', "I'm listening...");
      } else if (assistantActive) {
        // Already active, treat as question
        userInput.value = result;
        sendMessage();
      }
    }
  };

  recognition.onend = () => {
    if (assistantActive) recognition.start();
  };

  recognition.onerror = (err) => {
    console.error('[Voice] Error:', err.error);
    if (err.error === 'not-allowed') {
      assistantActive = false;
      voiceAssistantBtn.classList.remove('active');
      alert('Microphone access denied. Please allow it to use Voice Assistant.');
    }
  };
}

function toggleVoiceAssistant() {
  if (!recognition) initVoiceAssistant();

  assistantActive = !assistantActive;
  if (assistantActive) {
    voiceAssistantBtn.classList.add('active');
    recognition.start();
    appendMessage('ai', "Voice Assistant active. Say 'Hi Ting Ling Ling' to start a conversation!");
  } else {
    voiceAssistantBtn.classList.remove('active');
    recognition.stop();
  }
}

voiceAssistantBtn.addEventListener('click', toggleVoiceAssistant);

function playStatusSound(type) {
  // Simple heuristic: change UI state
  statusDot.classList.add('loading');
  setTimeout(() => statusDot.classList.remove('loading'), 1000);
}

/* ─── Send a Suggestion Chip ──────────────────────────────── */
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

  // Hide welcome, show messages
  welcomeScr.classList.add('hidden');

  // Append user bubble
  appendMessage('user', text);
  userInput.value = '';
  autoResize();

  // Show typing indicator
  const typingEl = appendTyping();
  isWaiting = true;
  sendBtn.disabled = true;

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    });
    const data = await res.json();
    typingEl.remove();

    const reply = data.reply || data.error || 'Sorry, something went wrong.';
    const el = appendMessage('ai', reply);
    scrollBottom();

    // Auto-speak if in Voice Assistant mode
    if (assistantActive) {
      const speakBtn = el.querySelector('.action-btn[onclick*="speakText"]');
      if (speakBtn) speakText(speakBtn);
    }
  } catch (err) {
    typingEl.remove();
    appendMessage('ai', '❌ Could not reach the server. Make sure `app.py` is running.');
  } finally {
    isWaiting = false;
    sendBtn.disabled = false;
  }
}

/* ─── Append User Message ─────────────────────────────────── */
function appendMessage(role, text) {
  const el = document.createElement('div');
  el.className = `message ${role}`;

  if (role === 'user') {
    el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  } else {
    const rendered = renderMarkdown(text);
    el.innerHTML = `
      <div class="ai-avatar">TL</div>
      <div class="ai-content">
        <div class="ai-text">${rendered}</div>
        <div class="ai-actions">
          <button class="action-btn" onclick="copyText(this)" data-text="${escapeAttr(text)}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
            </svg>
            Copy
          </button>
          <button class="action-btn" onclick="speakText(this)" data-text="${escapeAttr(text)}">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
              <path d="M19.07 4.93a10 10 0 010 14.14M15.54 8.46a5 5 0 010 7.07"/>
            </svg>
            Speak
          </button>
        </div>
      </div>`;
  }
  messages.appendChild(el);

  if (window.renderMathInElement) {
    renderMathInElement(el, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false},
        {left: '\\(', right: '\\)', display: false},
        {left: '\\[', right: '\\]', display: true}
      ],
      throwOnError : false
    });
  }

  scrollBottom();
  return el;
}

/* ─── Typing Dots ─────────────────────────────────────────── */
function appendTyping() {
  const el = document.createElement('div');
  el.className = 'message ai';
  el.innerHTML = `
    <div class="ai-avatar">TL</div>
    <div class="ai-content">
      <div class="typing">
        <div class="dot"></div><div class="dot"></div><div class="dot"></div>
      </div>
    </div>`;
  messages.appendChild(el);
  scrollBottom();
  return el;
}

/* ─── Actions ─────────────────────────────────────────────── */
function copyText(btn) {
  const text = btn.dataset.text;
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = '✓ Copied';
    setTimeout(() => { btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg> Copy`; }, 2000);
  });
}
window.copyText = copyText;

async function speakText(btn) {
  const text = btn.dataset.text;
  const voice = voiceSelect.value;
  const rate = voiceRate.value;
  
  isSpeaking = true;
  stopSpeechBtn.classList.remove('hidden');
  btn.textContent = '🎙️ Speaking...';
  
  try {
    await fetch('/speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice, rate })
    });
  } finally {
    // We don't immediately hide the stop button because the speech is still playing background
    // But we restore the button text
    setTimeout(() => { 
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 010 14.14M15.54 8.46a5 5 0 010 7.07"/></svg> Speak`; 
    }, 2000);
  }
}
window.speakText = speakText;

/* ─── Markdown Renderer (lightweight) ────────────────────── */
function renderMarkdown(text) {
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    // code blocks
    .replace(/```[\w]*\n?([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    // inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // headings
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // unordered list
    .replace(/^[•\-\*] (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
    // ordered list
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    // blockquote
    .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
    // line breaks
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br/>');
}

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function escapeAttr(t) {
  return t.replace(/"/g,'&quot;').replace(/'/g,'&#39;').replace(/\n/g,' ');
}

/* ─── Scroll ──────────────────────────────────────────────── */
function scrollBottom() {
  chatWindow.scrollTo({ top: chatWindow.scrollHeight, behavior: 'smooth' });
}
