"""
app.py — Flask server for Ting Ling Ling Study AI
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import threading
import subprocess
import torch
from flask import Flask, request, jsonify, render_template

# Force single threading to avoid mutex issues on Mac
torch.set_num_threads(1)

from brain import brain

app = Flask(__name__)

# Load synchronously to avoid thread conflicts during init
try:
    print("[App] Initializing brain...")
    brain.load()
except Exception as e:
    print(f"[App] Brain init error: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"error": "Empty message"}), 400

    try:
        answer = brain.ask(question)
        return jsonify({"reply": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text  = data.get("text", "")
    voice = data.get("voice", "daniel")
    if not text:
        return jsonify({"status": "empty"}), 400

    # Use the native Mac 'say' command for voice output
    # It's much more stable than pyttsx3 in this environment
    voice_map = {
        "daniel": "Daniel",
        "reed": "Reed",
        "rocko": "Rocko",
        "grandpa": "Grandpa",
        "eddy": "Eddy",
        "fred": "Fred"
    }
    os_voice = voice_map.get(voice, "Daniel")
    
    # Clean text for shell command
    clean_text = text[:800].replace('"', '').replace("'", "").replace("\n", " ")
    
    # Run in background via subprocess
    subprocess.Popen(["say", "-v", os_voice, clean_text])
    
    return jsonify({"status": "speaking"})


def _voice_id(name: str) -> str:
    voices = {
        "daniel":  "com.apple.voice.compact.en-GB.Daniel",
        "reed":    "com.apple.eloquence.en-US.Reed",
        "rocko":   "com.apple.eloquence.en-US.Rocko",
        "grandpa": "com.apple.eloquence.en-US.Grandpa",
        "eddy":    "com.apple.eloquence.en-US.Eddy",
        "fred":    "com.apple.speech.synthesis.voice.Fred",
    }
    return voices.get(name, voices["daniel"])


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": brain._loaded})


if __name__ == "__main__":
    print("═" * 55)
    print("   🤖  Ting Ling Ling — Study AI")
    print("   🌐  http://localhost:5001")
    print("═" * 55)
    app.run(host="0.0.0.0", port=5001, debug=False)
