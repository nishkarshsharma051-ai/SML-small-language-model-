import torch
from transformers import GPT2Config, GPT2LMHeadModel
import numpy as np

# Settings
device = "mps" if torch.backends.mps.is_available() else "cpu"
block_size = 64
batch_size = 4

# Load model ONLY from HF
config = GPT2Config(vocab_size=65, n_positions=block_size, n_embd=128, n_layer=2, n_head=2)
model = GPT2LMHeadModel(config)
model.to(device)

# Load data manually (like in the original working train.py)
data = np.fromfile('data/train.bin', dtype=np.uint16)

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print(f"Testing HF model with manual data loading on {device}...")
model.train()
xb, yb = get_batch()
outputs = model(xb, labels=yb)
loss = outputs.loss
loss.backward()
optimizer.step()
print(f"Success! Loss: {loss.item()}")
