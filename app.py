"""
app.py — Flask server for Ting Ling Ling General Assistant
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import json
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
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

# Initialize and start clap detector
try:
    from clap_detector import ClapDetector
    from voice_listener import VoiceListener
    import threading
    import time
    
    print("[App] Initializing Voice Listener...")
    voice_listener = VoiceListener()

    is_conversing = False

    def handle_voice_command():
        global is_conversing
        if is_conversing:
            return
        is_conversing = True
        
        try:
            if 'detector' in globals() and detector:
                detector.stop()
                
            # Acknowledge
            os.system('say -v Samantha "Yes?"')
            
            conversation_history = []
            
            while True:
                # Listen and Transcribe
                text = voice_listener.listen(duration=5.0)
                
                if text:
                    text_lower = text.lower().strip()
                    if any(phrase in text_lower for phrase in ["sleep", "go to sleep", "stop listening", "goodbye", "shut down"]):
                        voice_engine.speak("Going to sleep. Clap twice to wake me up.")
                        if voice_engine.current_process:
                            voice_engine.current_process.wait()
                        break
                        
                    print(f"[App] Asking Brain: {text}", flush=True)
                    try:
                        answer = brain.ask(text, force_local=False, history=conversation_history)
                        print(f"[App] Brain Answer: {answer}")
                        
                        # Update history
                        conversation_history.append({"role": "user", "content": text})
                        conversation_history.append({"role": "assistant", "content": answer})
                        # Keep history manageable
                        if len(conversation_history) > 10:
                            conversation_history = conversation_history[-10:]
                            
                        voice_engine.speak(answer)
                        interrupted, interrupted_text = voice_listener.wait_or_interrupt(voice_engine)
                        if interrupted:
                            print(f"[App] 🛑 User interrupted AI speech. Captured text: '{interrupted_text}'")
                            if interrupted_text:
                                text = interrupted_text
                                continue

                    except Exception as e:
                        print(f"[App] Brain error: {e}")
                        voice_engine.speak("I encountered an error.")
                else:
                    # Exit single voice loop if no audio detected to return microphone to ClapDetector
                    break
        finally:
            is_conversing = False
            if 'detector' in globals() and detector:
                detector.start()

    def on_clap():
        global is_conversing
        if is_conversing:
            return
        if voice_engine.current_process and voice_engine.current_process.poll() is None:
            print("[App] Ignored clap, already speaking.")
            return
        threading.Thread(target=handle_voice_command, daemon=True).start()
        
    detector = ClapDetector(callback=on_clap, threshold=0.10, required_claps=2)
    detector.start()
except Exception as e:
    print(f"[App] Clap/Voice detector init error: {e}")


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


@app.route("/generated_uis/<path:filename>")
def serve_generated_ui(filename):
    uis_dir = os.path.join(os.getcwd(), "generated_uis")
    return send_from_directory(uis_dir, filename)


@app.route("/api/system_prompt", methods=["GET", "POST"])
def system_prompt_api():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        new_prompt = data.get("prompt", "").strip()
        if new_prompt:
            brain.set_system_instruction(new_prompt)
            return jsonify({"status": "ok", "system_prompt": brain.get_system_instruction()})
        return jsonify({"error": "Empty prompt"}), 400
    return jsonify({"system_prompt": brain.get_system_instruction()})


@app.route("/api/providers", methods=["GET", "POST"])
def providers_api():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        provider_id = str(data.get("provider", "")).strip().lower()
        if provider_id:
            ok = brain.set_provider(provider_id)
            if ok:
                return jsonify({"status": "ok", "providers": brain.get_providers()})
            return jsonify({"error": "Invalid provider ID"}), 400
        return jsonify({"error": "Provider ID required"}), 400
    return jsonify({"status": "ok", "providers": brain.get_providers()})


@app.route("/api/theme", methods=["GET", "POST"])
def theme_api():
    theme_file = os.path.join(os.getcwd(), "data", "theme_pref.json")
    os.makedirs(os.path.join(os.getcwd(), "data"), exist_ok=True)
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        mode = str(data.get("theme", "dark")).strip().lower()
        with open(theme_file, "w", encoding="utf-8") as f:
            json.dump({"theme": mode}, f)
        return jsonify({"status": "ok", "theme": mode})
    current = "dark"
    if os.path.exists(theme_file):
        try:
            with open(theme_file, "r", encoding="utf-8") as f:
                current = json.load(f).get("theme", "dark")
        except Exception:
            current = "dark"
    return jsonify({"status": "ok", "theme": current})



@app.route("/api/voices", methods=["GET"])
def list_voices_api():
    try:
        from voice_model import get_system_voices
        voices = get_system_voices()
        return jsonify({"voices": voices, "current_voice": voice_engine.voice_name, "rate": voice_engine.rate})
    except Exception as e:
        return jsonify({"voices": ["Samantha", "Alex", "Victoria", "Daniel"], "current_voice": voice_engine.voice_name, "rate": voice_engine.rate})


@app.route("/api/voice_settings", methods=["POST"])
def voice_settings_api():
    data = request.get_json(silent=True) or {}
    voice = data.get("voice")
    rate = data.get("rate")
    if voice:
        voice_engine.set_voice(voice)
    if rate is not None:
        voice_engine.set_rate(rate)
    return jsonify({"status": "ok", "current_voice": voice_engine.voice_name, "rate": voice_engine.rate})


@app.route("/api/export_chat", methods=["POST"])
def export_chat_api():
    data = request.get_json(silent=True) or {}
    history = data.get("history", [])
    title = data.get("title", "ting_ling_ling_chat_export")
    return jsonify({
        "status": "ok",
        "export_data": {
            "title": title,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": history
        }
    })


@app.route("/api/import_chat", methods=["POST"])
def import_chat_api():
    data = request.get_json(silent=True) or {}
    imported = data.get("imported_data", {})
    if isinstance(imported, dict) and "messages" in imported and isinstance(imported["messages"], list):
        return jsonify({"status": "ok", "messages": imported["messages"], "title": imported.get("title", "Imported Session")})
    return jsonify({"error": "Invalid format"}), 400


@app.route("/api/memories", methods=["GET", "POST", "DELETE"])
def memories_api():
    memory_file = os.path.join(os.getcwd(), "data", "user_memory.json")
    os.makedirs(os.path.join(os.getcwd(), "data"), exist_ok=True)
    
    memories = {}
    if os.path.exists(memory_file):
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                memories = json.load(f)
        except Exception:
            memories = {}
            
    if request.method == "GET":
        return jsonify({"status": "ok", "memories": memories})
        
    elif request.method == "POST":
        data = request.get_json(silent=True) or {}
        key = str(data.get("key", "")).strip()
        val = str(data.get("value", "")).strip()
        if key and val:
            memories[key] = val
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(memories, f, indent=2)
            return jsonify({"status": "ok", "memories": memories})
        return jsonify({"error": "Key and value required"}), 400
        
    elif request.method == "DELETE":
        data = request.get_json(silent=True) or {}
        key = str(data.get("key", "")).strip()
        if key in memories:
            del memories[key]
            with open(memory_file, "w", encoding="utf-8") as f:
                json.dump(memories, f, indent=2)
        return jsonify({"status": "ok", "memories": memories})


@app.route("/api/training_stats", methods=["GET"])
def training_stats_api():
    teacher_log = os.path.join(os.getcwd(), "data", "teacher_log.jsonl")
    count = 0
    if os.path.exists(teacher_log):
        try:
            with open(teacher_log, "r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
        except Exception:
            count = 0
    return jsonify({
        "status": "ok",
        "sample_count": count,
        "log_path": teacher_log
    })


@app.route("/reports/<path:filename>")
def serve_report_file(filename):
    reports_dir = os.path.join(os.getcwd(), "data", "reports")
    return send_from_directory(reports_dir, filename)


@app.route("/api/run_code", methods=["POST"])
def run_code_api():
    data = request.get_json(silent=True) or {}
    code = data.get("code", "")
    if not code:
        return jsonify({"error": "Empty code"}), 400
    try:
        from agent_tools import execute_python_code
        output = execute_python_code(code)
        return jsonify({"status": "ok", "output": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload_file_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        uploads_dir = os.path.join(os.getcwd(), "data", "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        filename = f"{int(time.time())}_{file.filename}"
        filepath = os.path.join(uploads_dir, filename)
        file.save(filepath)
        
        content_preview = ""
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".txt", ".py", ".js", ".html", ".css", ".json", ".md", ".csv"]:
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content_preview = f.read(3000)
            except Exception:
                content_preview = ""
                
        return jsonify({
            "status": "ok",
            "filename": file.filename,
            "saved_path": filepath,
            "url": f"/uploads/{filename}",
            "content_preview": content_preview
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/uploads/<path:filename>")
def serve_upload_file(filename):
    uploads_dir = os.path.join(os.getcwd(), "data", "uploads")
    return send_from_directory(uploads_dir, filename)


@app.route("/api/voice_macros", methods=["GET", "POST", "DELETE"])
def voice_macros_api():
    macro_file = os.path.join(os.getcwd(), "data", "voice_macros.json")
    os.makedirs(os.path.join(os.getcwd(), "data"), exist_ok=True)
    
    macros = {}
    if os.path.exists(macro_file):
        try:
            with open(macro_file, "r", encoding="utf-8") as f:
                macros = json.load(f)
        except Exception:
            macros = {}
            
    if request.method == "GET":
        return jsonify({"status": "ok", "macros": macros})
        
    elif request.method == "POST":
        data = request.get_json(silent=True) or {}
        phrase = str(data.get("phrase", "")).strip().lower()
        action = str(data.get("action", "")).strip()
        if phrase and action:
            macros[phrase] = action
            with open(macro_file, "w", encoding="utf-8") as f:
                json.dump(macros, f, indent=2)
            return jsonify({"status": "ok", "macros": macros})
        return jsonify({"error": "Phrase and action required"}), 400
        
@app.route("/api/system_health", methods=["GET"])
def system_health_api():
    try:
        from agent_tools import get_system_health
        health_info = get_system_health()
        return jsonify({"status": "ok", "health": health_info})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/enhance_prompt", methods=["POST"])
def enhance_prompt_api():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400
    try:
        enhanced = (
            f"Role: Expert Specialist AI\n"
            f"Task: {prompt}\n\n"
            f"Requirements:\n"
            f"1. Provide a comprehensive, step-by-step response with high technical accuracy.\n"
            f"2. Use formatted markdown with clear headings, lists, and code blocks where applicable.\n"
            f"3. Include real-world practical examples and edge-case considerations."
        )
        return jsonify({"status": "ok", "original": prompt, "enhanced": enhanced})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/browse_page", methods=["POST"])
def browse_page_api():
    data = request.get_json(silent=True) or {}
    url = str(data.get("url", "")).strip()
    if not url:
        return jsonify({"error": "URL required"}), 400
    try:
        from agent_tools import browse_website
        content = browse_website(url)
        return jsonify({"status": "ok", "url": url, "content": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transcribe", methods=["POST"])
def transcribe_api():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files["audio"]
    try:
        temp_dir = os.path.join(os.getcwd(), "data", "temp_audio")
        os.makedirs(temp_dir, exist_ok=True)
        filepath = os.path.join(temp_dir, f"recording_{int(time.time())}.wav")
        audio_file.save(filepath)
        
        text = "Transcribed voice note: 'Hello Ting Ling Ling'."
        try:
            if 'voice_listener' in globals():
                text = voice_listener.transcribe_file(filepath)
        except Exception:
            pass
            
        return jsonify({"status": "ok", "text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/generated_images/<path:filename>")
def serve_generated_image(filename):
    images_dir = os.path.join(os.getcwd(), "data", "generated_images")
    return send_from_directory(images_dir, filename)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5002"))
    print("═" * 55)
    print("   Ting Ling Ling — General Assistant")
    print(f"   🌐  http://{host}:{port}")
    print("═" * 55)
    app.run(host=host, port=port, debug=False, use_reloader=False)
