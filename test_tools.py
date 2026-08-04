import time
from brain import brain

print("Waiting for model to load...")
while not brain._loaded:
    time.sleep(1)
print("Model loaded! Asking question...")
ans = brain.ask("Please open Spotify", force_local=True)
print("ANSWER:", ans)
