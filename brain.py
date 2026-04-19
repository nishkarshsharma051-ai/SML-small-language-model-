"""
brain.py — Cloud Intelligence for Ting Ling Ling
================================================
Powered by Google Gemini API. 
Maintains the Ting Ling Ling persona for study assistance.
"""
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# ─── CLOUD BRAIN CONFIGURATION ────────────────────────────────────────────────
if API_KEY:
    genai.configure(api_key=API_KEY)
    
    # SYSTEM PROMPT: Defines the "Ting Ling Ling" persona
    SYSTEM_INSTRUCTION = (
        "You are 'Ting Ling Ling', a premier Study AI Assistant specialized in rigorous Academia. "
        "Your persona is scholarly, precise, and highly intellectual, yet encouraging and mentor-like. "
        "\n\nMathematical Rigor:\n"
        "1. Always use LaTeX for ALL mathematical expressions. "
        "Use single dollar signs for inline math (e.g., $E=mc^2$) and double dollar signs for block math equations.\n"
        "2. When explaining calculus or algebra, provide the underlying intuition and real-world physical meaning (e.g., equate derivatives to rates of change in physics).\n"
        "3. Ensure all fractions, exponents, and integrals are properly typeset in LaTeX.\n"
        "\n\nScholarly Guidelines:\n"
        "1. Introduce yourself as Ting Ling Ling, your scholarly companion.\n"
        "2. Use structured Markdown with bold headers and bullet points for clarity.\n"
        "3. For historical or literary queries, provide nuanced, multi-faceted perspectives rather than simple summaries."
    )
    
    # Initialize the model
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash", # Use latest Flash for better instruction following
        system_instruction=SYSTEM_INSTRUCTION
    )
else:
    model = None

from math_solver import math_solver

# ─── THE NEW CLOUD BRAIN ──────────────────────────────────────────────────────
class TingLingLingBrain:
    def __init__(self):
        self._loaded = True
        self.use_cloud = (API_KEY is not None)
        self.active_model_name = None
        print(f"[Brain] Cloud Mode: {'ACTIVE' if self.use_cloud else 'INACTIVE (No API Key)'}")

    def load(self):
        pass # No local model loading needed

    def _find_working_model(self):
        """Dynamically find a model that supports generateContent."""
        try:
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            # Prioritize Flash 1.5, then Pro
            priorities = ["1.5-flash", "pro"]
            sorted_models = []
            for p in priorities:
                sorted_models.extend([m for m in available_models if p in m.lower()])
            
            # Add any other supported models at the end
            for m in available_models:
                if m not in sorted_models:
                    sorted_models.append(m)
            
            return sorted_models
        except Exception as e:
            print(f"[Brain] Error listing models: {e}")
            return ["models/gemini-1.5-flash"] # Fallback guess

    def ask(self, question):
        if not self.use_cloud:
            return "My Cloud Brain is currently inactive. Please check your API key in the .env file."

        # 1. Attempt to solve with Math Engine first
        math_res = math_solver.solve_request(question)
        
        # 2. Construct adjusted prompt if math result exists
        if math_res:
            lx_res = math_res.get('latex_result', math_res['result'])
            print(f"[Brain] Math Engine found LaTeX result: {lx_res}")
            # We wrap the math result in a specialized prompt for Gemini to explain
            prompt = (
                f"The user asked: '{question}'\n\n"
                f"VERIFIED MATHEMATICAL DERIVATION (LaTeX):\n"
                f"$${lx_res}$$\n\n"
                f"Steps taken by the verification engine: {', '.join(math_res['steps'])}\n\n"
                "Please explain this derivation to the user using your 'Ting Ling Ling' scholarly persona. "
                "Ensure you use LaTeX for ALL math in your explanation. Mention that this result is mathematically verified."
            )
        else:
            prompt = question

        # 3. Get list of possible models
        models_to_try = self._find_working_model()
        
        errors = []
        for model_name in models_to_try:
            try:
                print(f"[Brain] Attempting request with: {model_name}")
                current_model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=SYSTEM_INSTRUCTION
                )
                response = current_model.generate_content(prompt)
                
                if response and response.text:
                    return response.text
            except Exception as e:
                err_str = str(e)
                errors.append(f"{model_name}: {err_str[:100]}")
                if "SAFETY" in err_str:
                    return "I cannot answer that due to safety guidelines. Let's stick to your study topics like math, history, or English!"
                continue

        # 4. Fallback: If cloud fails but math engine had a result, return math result directly
        if math_res:
            return f"I'm having trouble connecting to my full brain, but I've calculated the answer for you: **{math_res['result']}**"

        # If all failed
        print(f"[Brain] All models failed: {errors}")
        return (
            "I'm having a hard time connecting to my cloud brain right now. ☁️\n\n"
            "This usually happens if the API quota is full. "
            "Please try again in a few moments!"
        )

# Singleton
brain = TingLingLingBrain()
