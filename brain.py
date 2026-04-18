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
        "You are 'Ting Ling Ling', a specialized Study AI Assistant. "
        "Your personality is friendly, helpful, and scholarly. "
        "You specialize in Advanced Mathematics (Calculus, Algebra, Geometry), "
        "Global History, and English Literature/Grammar. "
        "\n\nGuidelines:\n"
        "1. Always introduces yourself as Ting Ling Ling if asked.\n"
        "2. Use Markdown for formatting (bold text, bullet points, math equations).\n"
        "3. Provide step-by-step solutions for math problems.\n"
        "4. Be concise but detailed when explaining historical events.\n"
        "5. If you see math symbols like x² or 3x, treat them as x^2 and 3*x for calculations."
    )
    
    # Initialize the model
    model = genai.GenerativeModel(
        model_name="gemini-pro-latest",
        system_instruction=SYSTEM_INSTRUCTION
    )
else:
    model = None

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
            # Prioritize Flash for speed, then Pro
            priorities = ["flash", "pro"]
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
            return ["models/gemini-pro"] # Fallback guess

    def ask(self, question):
        if not self.use_cloud:
            return "My Cloud Brain is currently inactive. Please check your API key in the .env file."

        # Get list of possible models
        models_to_try = self._find_working_model()
        
        errors = []
        for model_name in models_to_try:
            try:
                print(f"[Brain] Attempting request with: {model_name}")
                current_model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=SYSTEM_INSTRUCTION
                )
                response = current_model.generate_content(question)
                
                if response and response.text:
                    return response.text
            except Exception as e:
                err_str = str(e)
                errors.append(f"{model_name}: {err_str[:100]}")
                if "SAFETY" in err_str:
                    return "I cannot answer that due to safety guidelines. Let's stick to your study topics like math, history, or English!"
                continue

        # If all failed
        print(f"[Brain] All models failed: {errors}")
        return (
            "I'm having a hard time connecting to my cloud brain right now. ☁️\n\n"
            "This usually happens if the API quota is full. "
            "Please try again in a few moments!"
        )

# Singleton
brain = TingLingLingBrain()
