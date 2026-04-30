"""
app.py — Flask server for Ting Ling Ling General Assistant
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import json
import uuid
from flask import Flask, request, jsonify, render_template
from flask import Response, stream_with_context

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
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        data = {}
    question = str(data.get("message", "")).strip()
    history = data.get("history", [])
    if not isinstance(history, list):
        history = []
    # Accept 'brain_mode' (cloud or local)
    brain_mode = str(data.get("brain_mode", "cloud")).strip().lower()
    force_local = (brain_mode == "local")
    
    if not question:
        return jsonify({"error": "Empty message"}), 400

    try:
        answer = brain.ask(question, force_local=force_local, history=history)
        return jsonify({
            "reply": answer,
            "source": brain.source,
        })
    except Exception as e:
        return jsonify({
            "reply": "I hit an unexpected error, but I’m still up. Please try again with a shorter prompt.",
            "error": str(e),
            "source": "error",
        }), 200


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        data = {}

    question = str(data.get("message", "")).strip()
    history = data.get("history", [])
    if not isinstance(history, list):
        history = []
    brain_mode = str(data.get("brain_mode", "cloud")).strip().lower()
    force_local = (brain_mode == "local")

    if not question:
        return Response("data: " + json.dumps({"type": "error", "error": "Empty message"}) + "\n\n", mimetype="text/event-stream")

    request_id = uuid.uuid4().hex

    @stream_with_context
    def generate():
        try:
            rid, gen = brain.ask_stream(question, force_local=force_local, history=history, request_id=request_id)
            # Ensure client knows the request id.
            yield "data: " + json.dumps({"type": "meta", "request_id": rid, "source": brain.source}) + "\n\n"
            for chunk in gen:
                if chunk:
                    yield "data: " + json.dumps({"type": "chunk", "text": chunk}) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "source": brain.source}) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    return Response(generate(), mimetype="text/event-stream", headers=headers)


@app.route("/cancel", methods=["POST"])
def cancel():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        data = {}
    rid = str(data.get("request_id", "")).strip()
    if not rid:
        return jsonify({"status": "missing"}), 400
    ok = brain.cancel(rid)
    return jsonify({"status": "ok" if ok else "unknown"})


@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        data = {}
    text  = data.get("text", "")
    voice = data.get("voice", "daniel")
    rate  = int(data.get("rate", 175))
    if not text:
        return jsonify({"status": "empty"}), 400

    try:
        voice_engine.set_voice(voice)
        voice_engine.rate = rate
        voice_engine.speak(text)
        return jsonify({"status": "speaking"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 200


@app.route("/stop", methods=["POST"])
def stop_speak():
    try:
        voice_engine.stop()
        return jsonify({"status": "stopped"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 200


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": brain._loaded,
        "hf_loaded": getattr(brain, "hf_loaded", False),
        "cloud_enabled": getattr(brain, "use_cloud_primary", False),
        "source": getattr(brain, "source", "unknown"),
    })


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5001"))
    print("═" * 55)
    print("   Ting Ling Ling — General Assistant")
    print(f"   🌐  http://{host}:{port}")
    print("═" * 55)
    app.run(host=host, port=port, debug=False, use_reloader=False)
