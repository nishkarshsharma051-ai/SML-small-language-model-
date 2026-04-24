# 🏮 Ting Ling Ling — Small Language Model (SLM) Assistant

**Ting Ling Ling** is a high-performance, privacy-focused Small Language Model (SLM) designed for speed, efficiency, and a seamless "stealth" user experience. It features a minimal, premium dark-mode interface with integrated real-time voice interaction.

![License: MIT](https://img.shields.io/badge/License-MIT-white.svg)
![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Status: Active](https://img.shields.io/badge/Status-Active-green.svg)

---

## 🎙️ Voice Interaction Update
Ting Ling Ling now supports full **Bi-Directional Voice Interaction**:
- **Speech-to-Text (STT)**: Talk to the assistant directly via the microphone.
- **Text-to-Speech (TTS)**: The assistant responds with natural spoken replies.
- **Dynamic Visualizer**: Real-time visual feedback for listening and speaking states.

## ✨ Key Features
- **SLM Architecture**: Optimized for low latency and high efficiency, capable of running on consumer hardware.
- **Stealth Mode**: A brand-neutral identity that focuses solely on being a helpful assistant, without revealing underlying architectures (Gemini/Groq fallbacks).
- **Mathematical & Scholarly Expertise**: Fine-tuned for academic support, including LaTeX rendering for complex equations.
- **Minimalist Design**: A premium, "glassmorphic" UI designed for maximum focus.
- **Hybrid Intelligence**: Smart switching between local inference and cloud-powered engines for the best response quality.

## 🛠️ Technology Stack
- **Backend**: Python, Flask
- **Core AI**: PyTorch, Transformers
- **Frontend**: Vanilla JS (ES6+), CSS3 (Modern Glassmorphism)
- **Voice**: Web Speech API (STT & TTS)
- **Mathematical Rendering**: KaTeX

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- macOS (for native `say` command support) or a modern browser for Web Speech API.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/nishkarshsharma051-ai/SML-small-language-model-.git
   cd SML-small-language-model-
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   Create a `.env` file and add your configuration (API keys, etc.).

4. Run the application:
   ```bash
   python app.py
   ```
5. Open your browser at `http://127.0.0.1:5001`.

## 📜 Credits
Developed and maintained by **Nishkarsh Sharma**.

---
*Ting Ling Ling may make mistakes. Verify information.*
