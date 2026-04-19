import os
import threading
import torch
import google.generativeai as genai
from dotenv import load_dotenv

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
        self.source = "Cloud" # Default source

        print(f"[Brain] Cloud Mode: {'ACTIVE' if self.use_cloud else 'INACTIVE (No API Key)'}")
        if self.use_cloud:
            genai.configure(api_key=API_KEY)
            self.system_instruction = (
                "You are 'Ting Ling Ling', a premier Study AI Assistant specialized in rigorous Academia. "
                "Your persona is scholarly, precise, and highly intellectual, yet encouraging and mentor-like. "
                "\n\nMathematical Rigor:\n"
                "1. Always use LaTeX for ALL mathematical expressions. "
                "Use single dollar signs for inline math (e.g., $E=mc^2$) and double dollar signs for block math equations.\n"
                "2. When explaining calculus or algebra, provide the underlying intuition and real-world physical meaning.\n"
                "3. Ensure all fractions, exponents, and integrals are properly typeset in LaTeX.\n"
                "\n\nScholarly Guidelines:\n"
                "1. Introduce yourself as Ting Ling Ling, your scholarly companion.\n"
                "2. Use structured Markdown with bold headers and bullet points.\n"
            )

    def load(self):
        """Loads the local fine-tuned model weights."""
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

    def ask(self, question):
        # 1. Always try Cloud first if key exists
        if self.use_cloud:
            try:
                res = self._ask_cloud(question)
                if res:
                    self.source = "Cloud"
                    return res
            except Exception as e:
                print(f"[Brain] Cloud failed, falling back... Error: {e}")

        # 2. Fallback to Local Brain
        if self._loaded:
            self.source = "Local"
            return self._ask_local(question)

        return "I'm having trouble connecting to both my cloud and local brains. ☁️ Offline mode requires `ting_ling_ling.pth`."

    def _ask_cloud(self, question):
        """Internal method for Gemini API requests."""
        # Math pre-solver
        from math_solver import math_solver
        math_res = math_solver.solve_request(question)
        
        prompt = question
        if math_res:
            lx_res = math_res.get('latex_result', math_res['result'])
            prompt = (
                f"The user asked: '{question}'\n"
                f"VERIFIED MATHEMATICAL DERIVATION: $${lx_res}$$\n"
                "Please explain this using your Ting Ling Ling scholarly persona."
            )

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=self.system_instruction
        )
        response = model.generate_content(prompt)
        return response.text if response else None

    def _ask_local(self, question):
        """Inference loop for the fine-tuned local SLM."""
        with self.lock:
            # Prepare scholarly prompt format
            prompt = f"{question.upper()}:"
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
            
            prefix = "*(Offline Mode — Scholarly Local Brain)*\n\n"
            return prefix + answer

# Singleton
brain = TingLingLingBrain()
rd time connecting to my cloud brain right now. ☁️\n\n"
            "This usually happens if the API quota is full. "
            "Please try again in a few moments!"
        )

# Singleton
brain = TingLingLingBrain()
