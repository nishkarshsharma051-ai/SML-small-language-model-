import os
import threading
import torch

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

# Local Model Imports
from model import TingLingLing, TingLingLingConfig
from tokenizer import CharacterTokenizer

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# ─── THE NEW HYBRID BRAIN ─────────────────────────────────────────────────────
class TingLingLingBrain:
    def __init__(self):
        self._loaded = False
        self.use_cloud = (API_KEY is not None)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.lock = threading.Lock()
        
        # Local Model Assets
        self.local_model = None
        self.local_tokenizer = None
        self.hf_model = None
        self.hf_tokenizer = None
        self.hf_loaded = False
        self.source = "Cloud" # Default source
        self.hf_model_dir = os.getenv("LOCAL_HF_MODEL_DIR", "hf_local_model")

        print(f"[Brain] Cloud Mode: {'ACTIVE' if self.use_cloud else 'INACTIVE (No API Key)'}")
        if self.use_cloud:
            import google.generativeai as genai
            genai.configure(api_key=API_KEY)
            self.system_instruction = (
                "You are Ting Ling Ling, a helpful, reliable, general-purpose GPT-style assistant. "
                "Answer a wide range of user questions naturally and directly, across everyday topics, technical topics, creative tasks, and tutoring. "
                "If a question is ambiguous, ask a concise clarifying question. If you do not know something, say so plainly instead of inventing details. "
                "Be friendly, concise when the user wants brevity, and detailed when the user asks for depth. "
                "\n\nCoding Guidelines:\n"
                "1. Always use triple backticks with the correct language identifier for code blocks (e.g., ```python).\n"
                "2. Provide clean, well-commented code following industry best practices and explain the reasoning behind the solution.\n"
                "3. For advanced coding questions, include complexity notes, edge cases, and a short example usage.\n"
                "\n\nMathematical Rigor:\n"
                "1. Always use LaTeX for ALL mathematical expressions. "
                "Use single dollar signs for inline math (e.g., $E=mc^2$) and double dollar signs for block math equations.\n"
                "2. Prefer universal mathematical notation and symbols where appropriate, such as ∑, ∫, ∂, ⇒, ∀, ∃, ≈, ≤, ≥, and ∴.\n"
                "3. When teaching advanced math, show formulas step by step and keep notation precise.\n"
                "\n\nScholarly Guidelines:\n"
                "1. Introduce yourself as Ting Ling Ling only when it fits naturally.\n"
                "2. Use structured Markdown with bold headers and bullet points.\n"
            )

    def _fallback_reply(self, question=None, reason=None):
        """Return a safe response when a model path fails."""
        base = (
            "I’m having trouble generating a full answer right now, but I’m still here. "
            "Please try again, or send a shorter prompt and I’ll handle it."
        )
        if reason:
            base += f" (Reason: {reason})"
        return base

    def load(self):
        """Loads the best available local model weights."""
        if self._load_hf_local():
            self._loaded = True
            self.source = "Local-HF"
            print("[Brain] HF local brain is online.")
            return True

        ckpt_path = 'ting_ling_ling.pth'
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
            self.hf_model.to(self.device)
            self.hf_model.eval()
            self.hf_loaded = True
            return True
        except Exception as e:
            print(f"[Brain] HF local load error: {e}")
            return False

    def ask(self, question, force_local=False, history=None):
        # 1. Try Cloud first (unless force_local is True)
        if self.use_cloud and not force_local:
            try:
                res = self._ask_cloud(question, history=history)
                if res:
                    self.source = "Cloud"
                    return res
            except Exception as e:
                print(f"[Brain] Cloud failed, falling back... Error: {e}")

        # 2. Fallback to Local Brain (or forced Local)
        if self.hf_loaded:
            try:
                self.source = "Local-HF"
                return self._ask_hf_local(question, history=history)
            except Exception as e:
                print(f"[Brain] HF local failed, falling back... Error: {e}")

        if self._loaded:
            try:
                self.source = "Local"
                return self._ask_local(question, history=history)
            except Exception as e:
                print(f"[Brain] Local fallback failed, falling back... Error: {e}")

        return self._fallback_reply(question, "all model paths failed")

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
            if role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")

        parts.append(f"User: {question}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _ask_cloud(self, question, history=None):
        """Internal method for Gemini API requests."""
        try:
            # Math pre-solver
            from math_solver import math_solver
            math_res = math_solver.solve_request(question)
        except Exception as e:
            print(f"[Brain] Math pre-solver failed: {e}")
            math_res = None
        
        prompt = self._format_history(question, history=history)
        if math_res:
            lx_res = math_res.get('latex_result', math_res['result'])
            prompt = (
                f"{prompt}\n"
                f"The user asked a math question: '{question}'\n"
                f"VERIFIED MATHEMATICAL DERIVATION: $${lx_res}$$\n"
                "Use the verified derivation if relevant, then explain clearly and naturally."
            )

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=self.system_instruction
        )
        response = model.generate_content(prompt)
        return response.text if response else None

    def _ask_local(self, question, history=None):
        """Inference loop for the fine-tuned local SLM."""
        with self.lock:
            # Keep the offline prompt lightweight and chat-like so the local model
            # has a better chance of responding naturally.
            prompt_lines = []
            for turn in (history or [])[-6:]:
                role = str(turn.get("role", "")).lower()
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

            prefix = "*(Offline Mode - Local Brain)*\n\n"
            return prefix + answer

    def _ask_hf_local(self, question, history=None):
        """Inference loop for a local Hugging Face causal LM."""
        with self.lock:
            system_text = (
                "You are Ting Ling Ling, a helpful, reliable, general-purpose assistant. "
                "Answer coding, math, English, history, science, and everyday questions clearly. "
                "For science and astronomy questions, answer directly instead of refusing."
            )

            messages = [{"role": "system", "content": system_text}]
            for turn in (history or [])[-10:]:
                role = str(turn.get("role", "")).lower()
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
                prefix = "*(Offline Mode - Local HF Brain)*\n\n"
                return prefix + answer if answer else self._fallback_reply(question, "empty model output")
            except Exception as e:
                return self._fallback_reply(question, f"local hf inference error: {e}")

# Singleton
brain = TingLingLingBrain()
