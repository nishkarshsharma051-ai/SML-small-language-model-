import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("CLOUD_API_KEY")
url = os.getenv("CLOUD_ENDPOINT", "https://api.groq.com/openai/v1/chat/completions")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

system_instruction = (
    "You are a hyper-intelligent, state-of-the-art AI assistant. "
    "Your name is Ting Ling Ling. "
    "Capabilities: You excel at coding, math, science, and scholarly research. You can also control the macOS UI natively! "
    "Use the search_safari and close_safari tools for browsing. "
    "Use send_whatsapp_message to message contacts. "
    "Use control_spotify to natively play, pause, skip, or search songs on Spotify. "
    "Use execute_applescript to control other applications natively on macOS (like toggling settings, clicking UI buttons, etc). "
    "Keep your answers brief and natural, you are conversing by voice. "
)

payload = {
    "model": "llama-3.3-70b-versatile",
    "messages": [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": "open Spotify and play a song."}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "control_spotify",
                "description": "Controls Spotify playback natively.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "query": {"type": "string"}
                    }
                }
            }
        }
    ]
}

res = requests.post(url, headers=headers, json=payload)
print(res.json())
