\"\"\"
cloud_engine.py — Minimal cloud chat-completions client.

This module provides a generic bridge to OpenAI-compatible cloud APIs.
\"\"\"

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


CLOUD_BASE_URL = os.getenv("CLOUD_ENDPOINT", "https://api.groq.com/openai/v1")
DEFAULT_MODEL = os.getenv("CLOUD_MODEL_ID", "default-model")


class CloudError(RuntimeError):
    pass


def chat_completions(
    messages: List[Dict[str, str]],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    timeout_s: int = 30,
) -> str:
    key = api_key or os.getenv("CLOUD_API_KEY")
    if not key:
        raise CloudError("Missing CLOUD_API_KEY")

    url = f"{CLOUD_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.9,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if r.status_code >= 400:
        raise CloudError(f"{r.status_code} {r.text[:300]}")

    data = r.json()
    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception as e:
        raise CloudError(f"Unexpected response shape: {e}") from e
