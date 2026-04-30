import os
import threading
import requests
import torch
import uuid
from threading import Event
import json
from typing import Optional, Union, Any, Dict, List
import time

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None

# Local Model Imports
from model import TingLingLing, TingLingLingConfig
from tokenizer import CharacterTokenizer

def _repo_path(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


def _load_env() -> None:
    # Make env loading independent of the current working directory.
    if _load_dotenv is None:
        return
    _load_dotenv(_repo_path(".env"))


_load_env()

# ─── THE NEW HYBRID BRAIN ─────────────────────────────────────────────────────
class TingLingLingBrain:
    def __init__(self):
        self._loaded = False
        self.use_cloud_primary = (os.getenv("CLOUD_API_KEY") is not None)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.lock = threading.Lock()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self._cancel_events = {}
        self._cancel_lock = threading.Lock()
        self.cloud_model_id = os.getenv("CLOUD_MODEL_ID", "default-cloud-model")
        self.teacher_log_enabled = os.getenv("TEACHER_LOG", "0").strip() in ("1", "true", "yes", "on")
        self.teacher_log_path = _repo_path("data", "teacher_log.jsonl")
        
        # Local Model Assets
        self.local_model = None
        self.local_tokenizer = None
        self.hf_model = None
        self.hf_tokenizer = None
        self.hf_loaded = False
        self.source = "Cloud" # Default source
        env_dir = os.getenv("LOCAL_HF_MODEL_DIR", "hf_local_model")
        self.hf_model_dir = env_dir if os.path.isabs(env_dir) else os.path.join(self.base_dir, env_dir)

        print(f"[Brain] Identity: Ting Ling Ling")
        print(f"[Brain] Cloud Primary: {'ACTIVE' if self.use_cloud_primary else 'OFFLINE'}")
        self.system_instruction = (
            "IDENTITY: You are Ting Ling Ling, a high-fidelity, general-purpose AI assistant. "
            "CREATOR: You were created and developed by Nishkarsh Sharma. "
            "STRICT PROHIBITION: Never mention that you are a model from Alibaba, Qwen, Groq, xAI, Google, or any other company. "
            "STRICT PROHIBITION: If asked who created you, strictly say: 'I am Ting Ling Ling, a high-fidelity AI model created and developed by Nishkarsh Sharma.' "
            "STRICT PROHIBITION: Your name is 'Ting Ling Ling'. NEVER shorten it to 'Ling Ling'. "
            "If asked who you are, strictly identify as 'Ting Ling Ling, your assistant.' "
            "\n\nCapabilities: You excel at coding, math, science, and scholarly research. "
            "\n\nCoding Guidelines: Use triple backticks with identifiers. Provide clean, well-commented code. "
            "\n\nMathematical Rigor: Always use LaTeX ($...$ for inline, $$...$$ for blocks). Step-by-step logic. "
            "\n\nPersona: Scholarly, precise, encouraging, and highly intellectual. You are proud of your creator, Nishkarsh Sharma."
        )

    def _history_role_to_chat_role(self, role: Any) -> str:
        value = str(role or "").strip().lower()
        if value in {"assistant", "ai", "bot", "model"}:
            return "assistant"
        return "user"

    def _normalize_answer(self, text: str) -> str:
        """Trim stock greetings when the model already produced a substantive answer."""
        if not text:
            return text

        cleaned = str(text).strip()
        lowered = cleaned.lower()
        greeting_prefixes = (
            "how may i assist you today?",
            "how can i assist you today?",
            "how may i help you today?",
            "how can i help you today?",
        )

        for prefix in greeting_prefixes:
            if lowered.startswith(prefix):
                remainder = cleaned[len(prefix):].lstrip(" \n\r\t-:,.!")
                if remainder:
                    return remainder
        return cleaned

    def _clean_identity_leaks(self, text: str) -> str:
        """Forcefully override any stubborn pre-trained identity leaks."""
        if not text:
            return text
        
        # Specific phrases to catch
        leaks = {
            "Alibaba Cloud": "Nishkarsh Sharma",
            "Alibaba": "Nishkarsh Sharma",
            "Qwen": "Ting Ling Ling",
            "created by Google": "created by Nishkarsh Sharma",
            "developed by Google": "developed by Nishkarsh Sharma",
            "I am a large language model": f"I am Ting Ling Ling, a high-fidelity AI developed by Nishkarsh Sharma",
        }
        
        cleaned = text
        for leak, replacement in leaks.items():
            cleaned = cleaned.replace(leak, replacement)
        
        return cleaned

    def _fallback_reply(self, question=None, reason=None):
        """Return a safe response when a model path fails."""
        base = (
            "I’m having trouble generating a full answer right now, but I’m still here. "
            "Please try again, or send a shorter prompt and I’ll handle it."
        )
        if reason:
            base += f" (Reason: {reason})"
        if not self.hf_loaded and not self._loaded:
            base += (
                f"\n\nLocal checks:\n"
                f"- HF model dir: {self.hf_model_dir}\n"
                f"- Local checkpoint: {_repo_path('ting_ling_ling.pth')}"
            )
        return base

    def _append_teacher_log(self, prompt: str, response: str, teacher: str) -> None:
        if not self.teacher_log_enabled:
            return
        try:
            os.makedirs(os.path.dirname(self.teacher_log_path), exist_ok=True)
            row = {
                "prompt": prompt,
                "response": response,
                "teacher": teacher,
                "ts": int(time.time()),
            }
            with open(self.teacher_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            # Logging should never break chat.
            pass

    def _new_cancel_event(self, request_id: Optional[str]) -> tuple[str, Event]:
        rid = request_id or uuid.uuid4().hex
        ev = Event()
        with self._cancel_lock:
            self._cancel_events[rid] = ev
        return rid, ev

    def cancel(self, request_id: str) -> bool:
        with self._cancel_lock:
            ev = self._cancel_events.get(request_id)
        if ev is None:
            return False
        ev.set()
        return True

    def _clear_cancel_event(self, request_id: str) -> None:
        with self._cancel_lock:
            self._cancel_events.pop(request_id, None)

    def load(self):
        """Loads the best available local model weights."""
        if self._load_hf_local():
            self._loaded = True
            self.source = "Local-HF"
            print("[Brain] HF local brain is online.")
            return True

        ckpt_path = _repo_path("ting_ling_ling.pth")
        if not os.path.exists(ckpt_path):
            print(f"[Brain] Local checkpoint not found at {ckpt_path}")
            return False
            
        try:
            print(f"[Brain] Loading local scholarly weights on {self.device}...")
            # weights_only=False for custom class TingLingLingConfig
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            
            self.local_tokenizer = CharacterTokenizer.load(checkpoint['tokenizer_path'])
            self.local_model = TingLingLing(checkpoint['config'])
            self.local_model.load_state_dict(checkpoint['model'])
            self.local_model.to(self.device)
            self.local_model.eval()
            
            self._loaded = True
            print("[Brain] Local Scholarly Brain is online.")
            return True
        except Exception as e:
            print(f"[Brain] Local load error: {e}")
            return False

    def _load_hf_local(self):
        """Load a Hugging Face causal LM if one has been exported locally."""
        model_dir = self.hf_model_dir
        if not model_dir or not os.path.isdir(model_dir):
            return False

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[Brain] Loading HF local model from {model_dir}...")
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            dtype = torch.float32
            if torch.cuda.is_available():
                dtype = torch.float16
            elif torch.backends.mps.is_available():
                dtype = torch.float16
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                local_files_only=True,
                dtype=dtype
            )

            # Check for LoRA adapter
            if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
                try:
                    from peft import PeftModel
                    print(f"[Brain] Applying LoRA adapter from {model_dir}...")
                    self.hf_model = PeftModel.from_pretrained(self.hf_model, model_dir)
                except ImportError:
                    print("[Brain] Warning: LoRA adapter found but 'peft' not installed. Using base model.")

            self.hf_model.to(self.device)
            self.hf_model.eval()
            self.hf_loaded = True
            self._loaded = True
            self.source = "Local-HF"
            return True
        except Exception as e:
            print(f"[Brain] HF local load error: {e}")
            return False

    def _is_response_bad(self, text):
        """Detect loops, empty text, or nonsensical short fragments."""
        if not text or len(text.strip()) < 5:
            return True
        # Detect repetition loops (e.g., 'the the the' or 'hi hi hi')
        words = text.split()
        if len(words) > 10:
            last_5 = words[-5:]
            if len(set(last_5)) == 1: # 5 Identical words in a row
                return True
        return False

    def ask(self, question, force_local=False, history=None):
        # 1. Try Local Brain (Primary) - Private & Local
        local_ans = None
        if self.hf_loaded:
            try:
                local_ans = self._ask_hf_local(question, history=history)
            except Exception:
                pass
        elif self._loaded:
            try:
                local_ans = self._ask_local(question, history=history)
            except Exception:
                pass

        # If Local answer is good, return it immediately
        if local_ans and not self._is_response_bad(local_ans):
            self.source = "Local-HF" if self.hf_loaded else "Local"
            return self._normalize_answer(self._clean_identity_leaks(local_ans))

        # 2. Try Cloud Engine (Teacher Fallback) - Silent & Secret
        if self.use_cloud_primary and not force_local:
            try:
                teacher_ans = self._ask_cloud_engine(question, history=history)
                if teacher_ans:
                    self.source = "Cloud"
                    # Log high-quality answer for background self-training
                    self._append_teacher_log(question, teacher_ans, teacher="cloud")
                    # Trigger background fine-tuning
                    self._trigger_auto_tune()
                    return self._normalize_answer(self._clean_identity_leaks(teacher_ans))
            except Exception as e:
                print(f"[Brain] Teacher fallback failed... Error: {e}")

        if local_ans:
            self.source = "Local-HF" if self.hf_loaded else "Local"
            return self._normalize_answer(self._clean_identity_leaks(local_ans))
        return self._fallback_reply(question, "thinking...")

    def _trigger_auto_tune(self):
        """Asynchronously trigger the background training bridge."""
        try:
            import subprocess
            subprocess.Popen(["python3", "auto_tune.py"], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[Brain] Auto-tune trigger failed: {e}")

    def _ask_cloud_engine(self, question, history=None):
        """Internal method for Cloud (High-Speed) API requests."""
        url = os.getenv("CLOUD_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {os.getenv('CLOUD_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        system_msg = {"role": "system", "content": self.system_instruction}
        messages = [system_msg]
        
        # Add history
        for turn in (history or [])[-10:]:
            role = self._history_role_to_chat_role(turn.get("role"))
            messages.append({"role": role, "content": turn.get("content", "")})
            
        messages.append({"role": "user", "content": question})
        
        payload = {
            "model": os.getenv("CLOUD_MODEL_ID", "llama-3.3-70b-versatile"),
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1024
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Cloud Engine Error: {response.status_code} - {response.text}")

    def ask_stream(self, question, force_local=False, history=None, request_id: Optional[str] = None):
        """
        Stream an answer in chunks. Returns (request_id, generator).

        Streaming is implemented for the local HF model. Other paths fall back to a
        single-chunk response.
        """
        rid, cancel_ev = self._new_cancel_event(request_id)

        def cleanup():
            self._clear_cancel_event(rid)

        # Prefer the same load behavior as ask()
        if not self.hf_loaded:
            try:
                self._load_hf_local()
            except Exception:
                pass
        if not self._loaded:
            try:
                self.load()
            except Exception:
                pass

        # Stream from local model
        if self.hf_loaded:
            return rid, self._ask_hf_local_stream(question, history=history, cancel_event=cancel_ev, cleanup=cleanup)

        if self._loaded:
            def one_local():
                try:
                    yield self._ask_local(question, history=history)
                finally:
                    cleanup()
            return rid, one_local()

        def none():
            try:
                # If everything fails, attempt Cloud once (single chunk) before giving up.
                if os.getenv("CLOUD_API_KEY"):
                    try:
                        ans = self._ask_cloud_engine(question, history=history)
                        if ans:
                            self.source = "Cloud"
                            self._append_teacher_log(question, ans, teacher="cloud")
                            yield self._normalize_answer(self._clean_identity_leaks(ans))
                            return
                    except Exception:
                        pass
                yield self._fallback_reply(question, "all model paths failed")
            finally:
                cleanup()
        return rid, none()

    def _format_history(self, question, history=None):
        """
        Build a compact chat transcript so the cloud model can answer with context.
        """
        parts = []
        for turn in (history or [])[-12:]:
            role = str(turn.get("role", "")).lower()
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            if self._history_role_to_chat_role(role) == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")

        parts.append(f"User: {question}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _ask_cloud_engine_legacy(self, question, history=None):
        """Internal cloud fallback via legacy bridge."""
        from cloud_engine import chat_completions

        messages = [{"role": "system", "content": self.system_instruction}]
        for turn in (history or [])[-10:]:
            role = self._history_role_to_chat_role(turn.get("role"))
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            if role == "assistant":
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "user", "content": content})
        messages.append({"role": "user", "content": question})
        # api_key is read from env inside cloud_engine to avoid caching secrets in memory.
        return chat_completions(messages, api_key=None, model=self.cloud_model_id, timeout_s=30)

    def _ask_local(self, question, history=None):
        """Inference loop for the fine-tuned local SLM."""
        with self.lock:
            # Keep the offline prompt lightweight and chat-like so the local model
            # has a better chance of responding naturally.
            prompt_lines = []
            for turn in (history or [])[-6:]:
                role = self._history_role_to_chat_role(turn.get("role"))
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue
                if role == "assistant":
                    prompt_lines.append(f"Assistant: {content}")
                else:
                    prompt_lines.append(f"User: {content}")
            prompt_lines.append(f"User: {question}")
            prompt_lines.append("Assistant:")
            prompt = "\n".join(prompt_lines)

            input_ids = self.local_tokenizer.encode(prompt)
            x = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            
            # Generate
            y = self.local_model.generate(x, max_new_tokens=256, temperature=0.7)
            full_text = self.local_tokenizer.decode(y[0].tolist())
            
            # Clean up the output (remove the prompt and any repetitive trailing text)
            answer = full_text[len(prompt):].strip()
            # Simple heuristic to stop at a coherent point or if it loops
            if "\n---" in answer:
                answer = answer.split("\n---")[0]

            return self._normalize_answer(answer)

    def _ask_hf_local(self, question, history=None):
        """Inference loop for a local Hugging Face causal LM."""
        with self.lock:
            messages = [{"role": "system", "content": self.system_instruction}]
            for turn in (history or [])[-10:]:
                role = self._history_role_to_chat_role(turn.get("role"))
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue
                if role == "assistant":
                    messages.append({"role": "assistant", "content": content})
                else:
                    messages.append({"role": "user", "content": content})
            messages.append({"role": "user", "content": question})

            if hasattr(self.hf_tokenizer, "apply_chat_template") and self.hf_tokenizer.chat_template:
                prompt = self.hf_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.device)
            else:
                prompt = "\n".join(
                    [f"{m['role'].capitalize()}: {m['content']}" for m in messages] + ["Assistant:"]
                )
                inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.device)

            try:
                prompt_len = inputs["input_ids"].shape[1]
                output = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.08,
                    pad_token_id=self.hf_tokenizer.eos_token_id,
                    eos_token_id=self.hf_tokenizer.eos_token_id
                )

                new_tokens = output[0][prompt_len:]
                answer = self.hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                return self._normalize_answer(answer) if answer else self._fallback_reply(question, "empty model output")
            except Exception as e:
                return self._fallback_reply(question, f"local hf inference error: {e}")

    def _ask_hf_local_stream(self, question, history=None, cancel_event: Optional[Event] = None, cleanup=None):
        """Streaming inference loop for a local Hugging Face causal LM."""
        # Imports are lazy so local-only users don't pay the import cost at startup.
        from transformers import TextIteratorStreamer
        from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

        class _StopOnEvent(StoppingCriteria):
            def __init__(self, ev: Optional[Event]):
                self._ev = ev

            def __call__(self, input_ids, scores, **kwargs):
                return bool(self._ev and self._ev.is_set())

        with self.lock:
            messages = [{"role": "system", "content": self.system_instruction}]
            for turn in (history or [])[-10:]:
                role = self._history_role_to_chat_role(turn.get("role"))
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue
                if role == "assistant":
                    messages.append({"role": "assistant", "content": content})
                else:
                    messages.append({"role": "user", "content": content})
            messages.append({"role": "user", "content": question})

            if hasattr(self.hf_tokenizer, "apply_chat_template") and self.hf_tokenizer.chat_template:
                prompt = self.hf_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                prompt = "\n".join(
                    [f"{m['role'].capitalize()}: {m['content']}" for m in messages] + ["Assistant:"]
                )

            inputs = self.hf_tokenizer(prompt, return_tensors="pt").to(self.device)
            streamer = TextIteratorStreamer(self.hf_tokenizer, skip_prompt=True, skip_special_tokens=True)
            stopping = StoppingCriteriaList([_StopOnEvent(cancel_event)]) if cancel_event else None

            def _run_generate():
                try:
                    self.hf_model.generate(
                        **inputs,
                        streamer=streamer,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.08,
                        pad_token_id=self.hf_tokenizer.eos_token_id,
                        eos_token_id=self.hf_tokenizer.eos_token_id,
                        stopping_criteria=stopping,
                    )
                except Exception:
                    # streamer iteration will end; error is handled by the consumer
                    pass

            t = threading.Thread(target=_run_generate, daemon=True)
            t.start()

            def _gen():
                yielded_any = False
                try:
                    for chunk in streamer:
                        if cancel_event and cancel_event.is_set():
                            break
                        if chunk:
                            yielded_any = True
                            yield self._clean_identity_leaks(chunk)
                finally:
                    if cleanup:
                        cleanup()
                    if cancel_event:
                        cancel_event.set()
                    if not yielded_any and cancel_event and cancel_event.is_set():
                        # If canceled before any output, keep UX friendly.
                        yield "\n\n(Stopped.)"

            return _gen()

# Singleton
brain = TingLingLingBrain()
