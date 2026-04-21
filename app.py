"""
app.py — Flask server for Ting Ling Ling General Assistant
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
from voice_model import VoiceModel

app = Flask(__name__)

# Initialize voice model
voice_engine = VoiceModel()

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
    history = data.get("history", [])
    # Accept 'brain_mode' (cloud or local)
    brain_mode = data.get("brain_mode", "cloud")
    force_local = (brain_mode == "local")
    
    if not question:
        return jsonify({"error": "Empty message"}), 400

    try:
        answer = brain.ask(question, force_local=force_local, history=history)
        return jsonify({
            "reply": answer,
            "source": brain.source
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text  = data.get("text", "")
    voice = data.get("voice", "daniel")
    rate  = int(data.get("rate", 175))
    if not text:
        return jsonify({"status": "empty"}), 400

    voice_engine.set_voice(voice)
    voice_engine.rate = rate
    voice_engine.speak(text)
    
    return jsonify({"status": "speaking"})


@app.route("/stop", methods=["POST"])
def stop_speak():
    voice_engine.stop()
    return jsonify({"status": "stopped"})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": brain._loaded})


if __name__ == "__main__":
    print("═" * 55)
    print("   🤖  Ting Ling Ling — General Assistant")
    print("   🌐  http://localhost:5001")
    print("═" * 55)
    app.run(host="0.0.0.0", port=5001, debug=False)
