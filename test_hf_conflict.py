import torch
from model import TingLingLing, TingLingLingConfig
import transformers # just import to see if it causes mutex crash
import numpy as np

device = "mps" if torch.backends.mps.is_available() else "cpu"
block_size = 64
batch_size = 4

config = TingLingLingConfig(vocab_size=65, block_size=block_size)
model = TingLingLing(config)
model.to(device)

data = np.fromfile('data/train.bin', dtype=np.uint16)

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

print(f"Testing custom model with HF imported on {device}...")
model.train()
xb, yb = get_batch()
logits, loss = model(xb, yb)
loss.backward()
optimizer.step()
print(f"Success! Loss: {loss.item()}")
