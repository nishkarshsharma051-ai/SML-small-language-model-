import torch
from model import TingLingLing, TingLingLingConfig
from tokenizer import CharacterTokenizer

# --- Configuration ---
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
ckpt_path = 'ting_ling_ling.pth'
# ---------------------

# load checkpoint 
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
config = checkpoint['config']
tokenizer = CharacterTokenizer.load(checkpoint['tokenizer_path'])
model = TingLingLing(config)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# encoding the start context
start_text = "PYTHAGOREAN THEOREM:"
start_ids = tokenizer.encode(start_text)
x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

print(f"Generating from prompt: {start_text}")
print("-" * 30)
y = model.generate(x, max_new_tokens=500, temperature=0.8, top_k=10)
print(tokenizer.decode(y[0].tolist()))
print("-" * 30)
