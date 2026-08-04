import os
import time
import math
import numpy as np
import torch
from model import TingLingLing, TingLingLingConfig
from tokenizer import CharacterTokenizer

# --- Hyperparameters ---
batch_size = 32
block_size = 128
max_iters = 2000
learning_rate = 1e-3
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_interval = 250
eval_iters = 200
log_interval = 10
# -----------------------

# load tokenizer for vocab_size
tokenizer_path = 'data/tokenizer.json'
tokenizer = CharacterTokenizer.load(tokenizer_path)
vocab_size = tokenizer.vocab_size

# data loading logic
train_data = np.fromfile('data/train.bin', dtype=np.uint16)
val_data = np.fromfile('data/val.bin', dtype=np.uint16)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# model setup
config = TingLingLingConfig(vocab_size=vocab_size, block_size=block_size)
model = TingLingLing(config)
model.to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Starting training on {device}...")
t0 = time.time()

for iter in range(max_iters):
    # evaluate loss once in a while
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch
    xb, yb = get_batch('train')

    # forward pass
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % log_interval == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"iter {iter}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")

# save the model
checkpoint = {
    'model': model.state_dict(),
    'config': config,
    'tokenizer_path': tokenizer_path
}
torch.save(checkpoint, 'ting_ling_ling.pth')
print("Training complete. Model saved to ting_ling_ling.pth")
