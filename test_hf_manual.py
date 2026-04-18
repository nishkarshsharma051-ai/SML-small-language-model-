import torch
from transformers import GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling
from tokenizer_hf import TingLingLingTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset
import json

# Settings
device = "mps" if torch.backends.mps.is_available() else "cpu"
block_size = 64
batch_size = 4

# Load tokenizer
with open('data/tokenizer.json', 'r') as f:
    tokenizer_data = json.load(f)
tokenizer = TingLingLingTokenizer(chars=tokenizer_data['chars'], model_max_length=block_size)

# Load model
config = GPT2Config(vocab_size=tokenizer.vocab_size, n_positions=block_size, n_embd=128, n_layer=2, n_head=2)
model = GPT2LMHeadModel(config)
model.to(device)

# Simple data
text = "TING LING LING " * 10
train_dataset = Dataset.from_dict({"text": [text[i:i+block_size] for i in range(0, len(text)-block_size, block_size)]})
train_dataset = train_dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=block_size), batched=True)
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# One step test
print(f"Testing manual HF loop on {device}...")
model.train()
for batch in loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Step successful! Loss: {loss.item()}")
    break

print("Manual HF loop test passed!")
