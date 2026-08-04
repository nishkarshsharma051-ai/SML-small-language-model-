import torch
from brain import TingLingLingBrain

def verify_brain():
    brain = TingLingLingBrain()
    brain.use_cloud = False # Force local mode
    
    print("\n--- Verifying Local Brain ---")
    if not brain.load():
        print("❌ Failed to load local model!")
        return

    # Prompt from scholarly data: History (Napoleonic Wars)
    questions = [
        "What was the Battle of Waterloo?",
        "Explain the Law of Conservation of Energy."
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer = brain.ask(q)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    verify_brain()
