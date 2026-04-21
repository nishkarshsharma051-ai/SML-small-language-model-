import torch
from brain import TingLingLingBrain

def verify_coding_brain():
    brain = TingLingLingBrain()
    brain.use_cloud = False # Force local mode
    
    print("\n--- Verifying Coding Logic in Local Brain ---")
    if not brain.load():
        print("❌ Failed to load local model!")
        return

    # Questions based on new coding data
    questions = [
        "How do you define a function in Python?",
        "What are JavaScript promises?",
        "Explain CSS Flexbox centering."
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer = brain.ask(q)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    verify_coding_brain()
