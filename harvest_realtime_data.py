import os
import json
import time
from duckduckgo_search import DDGS
from cloud_engine import chat_completions
import subprocess
from dotenv import load_dotenv

load_dotenv()

TOPICS = [
    "Latest AI breakthroughs 2026",
    "Latest global technology news July 2026",
    "Global market finance news today 2026",
    "Space exploration news 2026",
    "Latest science discoveries 2026"
]

def harvest():
    print("[Harvester] Starting real-time data harvest...")
    all_snippets = []
    
    with DDGS() as ddgs:
        for topic in TOPICS:
            print(f"[Harvester] Searching for: {topic}")
            try:
                results = list(ddgs.text(topic, max_results=5))
                for res in results:
                    all_snippets.append(res['body'])
            except Exception as e:
                print(f"[Harvester] Search error: {e}")
                
    if not all_snippets:
        print("[Harvester] No data found.")
        return
        
    chunk_size = 5
    chunks = [all_snippets[i:i + chunk_size] for i in range(0, len(all_snippets), chunk_size)]
    
    qa_pairs = []
    
    for chunk in chunks:
        context = "\n".join(chunk)
        prompt = f"""
You are an expert AI data synthesizer. Based ONLY on the following real-time data snippets from July 2026, generate 3 highly intelligent, conversational Question-Answer pairs.
The questions should ask about the facts in the text. The answers should be accurate and helpful.
Output ONLY a raw JSON array of objects with "prompt" and "response" keys. Do not include markdown blocks like ```json.

Snippets:
{context}
"""
        try:
            print("[Harvester] Generating QA pairs via Cloud LLM...")
            messages = [{"role": "user", "content": prompt}]
            response_text = chat_completions(messages, model="llama-3.3-70b-versatile", timeout_s=30)
            
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
                
            parsed = json.loads(cleaned.strip())
            if isinstance(parsed, list):
                qa_pairs.extend(parsed)
        except Exception as e:
            print(f"[Harvester] Cloud LLM Error: {e}")
            
    if not qa_pairs:
        print("[Harvester] Failed to generate any QA pairs.")
        return
        
    print(f"[Harvester] Successfully generated {len(qa_pairs)} high-quality real-time QA pairs.")
    
    dataset_path = "data/teacher_log.jsonl"
    os.makedirs("data", exist_ok=True)
    with open(dataset_path, "a", encoding="utf-8") as f:
        for pair in qa_pairs:
            entry = {
                "prompt": pair["prompt"],
                "response": pair["response"],
                "teacher": "cloud_harvester"
            }
            f.write(json.dumps(entry) + "\n")
            
    print("[Harvester] Triggering real-time micro-training...")
    subprocess.run(["venv/bin/python3", "auto_tune.py"])
    print("[Harvester] Done!")

if __name__ == "__main__":
    harvest()
