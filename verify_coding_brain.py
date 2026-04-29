import os
import torch
from brain import TingLingLingBrain

def verify_advanced_coding():
    print("Initializing Ting Ling Ling Brain...")
    brain = TingLingLingBrain()
    brain.load()
    
    questions = [
        "Explain Dynamic Programming.",
        "How does Dijkstra's Algorithm work?",
        "Explain the Python GIL.",
        "What is a Trie data structure?",
        "Explain the difference between vertical and horizontal scaling.",
        "What is a Fenwick Tree?",
        "What are consistency patterns in system design?",
        "What is database sharding?"
    ]
    
    print("\n--- Testing Advanced Coding Knowledge ---\n")
    for q in questions:
        print(f"Question: {q}")
        # Force local to test the SLM specifically
        answer = brain.ask(q, force_local=True)
        print(f"Answer: {answer}\n")
        print("-" * 50)

if __name__ == "__main__":
    verify_advanced_coding()
