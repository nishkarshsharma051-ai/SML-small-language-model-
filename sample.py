import torch
from model import TingLingLing, TingLingLingConfig
from tokenizer import CharacterTokenizer

# check if we can use MPS (Mac GPU)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# load the model
checkpoint = torch.load('ting_ling_ling.pth', map_location=device)
config = checkpoint['config']
model = TingLingLing(config)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# load tokenizer
tokenizer = CharacterTokenizer.load(checkpoint['tokenizer_path'])

# generate!
prompt = "TING LING LING:"
start_ids = tokenizer.encode(prompt)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

print(f"Generating from prompt: {prompt}")
print("-" * 30)
y = model.generate(x, max_new_tokens=500, temperature=0.8, top_k=10)
print(tokenizer.decode(y[0].tolist()))
print("-" * 30)
