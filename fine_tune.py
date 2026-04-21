import os
import time
import math
import numpy as np
import torch
from model import TingLingLing, TingLingLingConfig
from tokenizer import CharacterTokenizer

# --- Hyperparameters for Fine-tuning ---
batch_size = 32
block_size = 128
max_iters = 1000  # Fine-tuning for coding skills needs more steps
learning_rate = 1e-4 # Lower LR for fine-tuning
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_interval = 100
eval_iters = 100
log_interval = 10
# ---------------------------------------

# 1. Load tokenizer
tokenizer_path = 'data/tokenizer_scholarly.json'
tokenizer = CharacterTokenizer.load(tokenizer_path)
vocab_size = tokenizer.vocab_size

# 2. Load scholarly data
train_data = np.fromfile('data/scholarly_train.bin', dtype=np.uint16)
val_data = np.fromfile('data/scholarly_val.bin', dtype=np.uint16)

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

# 3. Load the expanded base model
ckpt_path = 'ting_ling_ling_expanded.pth'
if not os.path.exists(ckpt_path):
    print(f"Error: {ckpt_path} not found. Run expand_vocab.py first.")
    exit(1)

print(f"Loading base model from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model = TingLingLing(checkpoint['config'])
model.load_state_dict(checkpoint['model'])
model.to(device)

# 4. Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Starting fine-tuning on {device}...")
t0 = time.time()

for iter in range(max_iters):
    # evaluate loss once in a while
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {iter}: scholarly train loss {losses['train']:.4f}, scholarly val loss {losses['val']:.4f}")

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

# 5. Save the fine-tuned model
# We save it as ting_ling_ling.pth to replace the Shakespeare version for the app
final_ckpt = {
    'model': model.state_dict(),
    'config': model.config,
    'tokenizer_path': tokenizer_path
}
torch.save(final_ckpt, 'ting_ling_ling.pth')
print("Fine-tuning complete. Model saved to ting_ling_ling.pth")
