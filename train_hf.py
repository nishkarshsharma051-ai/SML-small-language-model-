import os
import time
import torch
import numpy as np
import json
from model import TingLingLing, TingLingLingConfig
from tokenizer import CharacterTokenizer

# --- Settings ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
output_dir = "./ting_ling_ling_checkpoints"
batch_size = 32
block_size = 128
max_steps = 1000
learning_rate = 5e-4
eval_interval = 250
# ----------------

# 1. Load tokenizer (Original version)
with open('data/tokenizer.json', 'r') as f:
    tokenizer_data = json.load(f)
tokenizer = CharacterTokenizer(tokenizer_data['chars'])

# 2. Prepare Model (Pure PyTorch version for stability)
config = TingLingLingConfig(
    vocab_size=tokenizer.vocab_size,
    n_embd=384,
    n_layer=6,
    n_head=6,
    block_size=block_size
)
model = TingLingLing(config)
model.to(device)

# 3. Data Loading
train_data = np.fromfile('data/train.bin', dtype=np.uint16)
val_data = np.fromfile('data/val.bin', dtype=np.uint16)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

# 4. Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 5. Stable Training Loop
print(f"Starting Stable Training (Preparation for HF Export) on {device}...")
os.makedirs(output_dir, exist_ok=True)
t0 = time.time()

for step in range(max_steps):
    # Eval
    if step % eval_interval == 0 or step == max_steps - 1:
        model.eval()
        with torch.no_grad():
            xb, yb = get_batch('val')
            logits, loss = model(xb, yb)
            print(f"Step {step}: Val Loss {loss.item():.4f}")
        model.train()

    # Training step
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"Step {step}: Train Loss {loss.item():.4f}, Time {dt*1000:.2f}ms")

# 6. Save checkpoint for export
print("Training complete. Saving checkpoint...")
checkpoint = {
    'model': model.state_dict(),
    'config': {
        'vocab_size': config.vocab_size,
        'n_embd': config.n_embd,
        'n_layer': config.n_layer,
        'n_head': config.n_head,
        'block_size': config.block_size
    },
    'tokenizer_chars': tokenizer_data['chars']
}
torch.save(checkpoint, os.path.join(output_dir, 'stable_checkpoint.pth'))
print(f"Stable checkpoint saved to {output_dir}/stable_checkpoint.pth")
