"""
test_isolated_brain.py
Tests if loading the model in a separate process avoids the mutex crash.
"""
import multiprocessing as mp
import torch
import os

def load_and_inference(q, pipe_conn):
    try:
        # Import inside the process to avoid inheriting parent's locks
        from transformers import pipeline
        print(f"[Worker] Loading model on cpu...")
        pipe = pipeline(
            "text-generation", 
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            device="cpu"
        )
        print(f"[Worker] Inference for: {q}")
        res = pipe(q, max_new_tokens=50)[0]['generated_text']
        pipe_conn.send({"status": "ok", "result": res})
    except Exception as e:
        pipe_conn.send({"status": "error", "message": str(e)})

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=load_and_inference, args=("Explain the Pythagorean theorem.", child_conn))
    p.start()
    
    print("Waiting for result...")
    if parent_conn.poll(60): # 60s timeout
        print("Result:", parent_conn.recv())
    else:
        print("Timeout or worker crashed.")
    p.join()
