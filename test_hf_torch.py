import torch
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(vocab_size=100, n_layer=2, n_embd=64, n_head=2)
model = GPT2LMHeadModel(config)
x = torch.randint(0, 100, (1, 10))
y = model(x)
print("Forward pass successful!")
