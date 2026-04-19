import torch
import os
import json
from tokenizer import CharacterTokenizer
from model import TingLingLing, TingLingLingConfig

def expand_vocab():
    # 1. Load original tokenizer to see what we have
    old_tokenizer_path = 'data/tokenizer.json'
    old_tokenizer = CharacterTokenizer.load(old_tokenizer_path)
    old_chars = old_tokenizer.chars
    print(f"Old vocab size: {len(old_chars)}")

    # 2. Define new scholarly characters
    # Instead of guessing, we scan the actual scholarly dataset
    scholarly_input_path = 'data/scholarly_input.txt'
    if os.path.exists(scholarly_input_path):
        with open(scholarly_input_path, 'r') as f:
            scholarly_data = f.read()
        scholarly_chars = set(scholarly_data)
        print(f"Found {len(scholarly_chars)} unique characters in scholarly dataset.")
    else:
        scholarly_chars = set()
        print("Scholarly dataset not found, using default additions.")
    
    # Combined unique chars, sorted for consistency
    all_chars = sorted(list(set(old_chars) | scholarly_chars))
    print(f"New total vocab size: {len(all_chars)}")

    # 3. Save new tokenizer
    new_tokenizer = CharacterTokenizer("".join(all_chars))
    new_tokenizer_path = 'data/tokenizer_scholarly.json'
    new_tokenizer.save(new_tokenizer_path)
    print(f"Saved scholarly tokenizer to {new_tokenizer_path}")

    # 4. Load the original model weights
    ckpt_path = 'ting_ling_ling.pth'
    # Use weights_only=False for PyTorch 2.6+ to load custom classes like TingLingLingConfig
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    old_state_dict = checkpoint['model']
    old_config = checkpoint['config']
    
    # 5. Create a new model with expanded config
    new_config = TingLingLingConfig(
        vocab_size=len(all_chars),
        n_embd=old_config.n_embd,
        n_layer=old_config.n_layer,
        n_head=old_config.n_head,
        block_size=old_config.block_size
    )
    new_model = TingLingLing(new_config)
    new_state_dict = new_model.state_dict()

    # 6. Weight Transfer
    # Create mapping from old index to new index
    old_to_new = {old_tokenizer.stoi[c]: new_tokenizer.stoi[c] for c in old_chars}
    
    for name, param in old_state_dict.items():
        if 'transformer.wte.weight' in name or 'lm_head.weight' in name:
            # Transfer existing embeddings / head weights
            for old_idx, new_idx in old_to_new.items():
                new_state_dict[name][new_idx] = param[old_idx]
            print(f"Transferred {len(old_to_new)} weights for {name}")
        elif 'bias' in name and 'attn' in name:
            # Skip attention mask bias (it's a fixed buffer)
            continue
        elif name in new_state_dict:
            # Non-vocab dependent layers (Attention, MLP, Layernorm weights/biases, positional embeddings)
            if new_state_dict[name].shape == param.shape:
                new_state_dict[name].copy_(param)
            else:
                print(f"Skipping {name} due to shape mismatch: {param.shape} -> {new_state_dict[name].shape}")
        else:
            print(f"Key {name} not found in new model.")

    # 7. Save the "Seed" model for fine-tuning
    new_checkpoint = {
        'model': new_state_dict,
        'config': new_config,
        'tokenizer_path': new_tokenizer_path
    }
    seed_pth = 'ting_ling_ling_expanded.pth'
    torch.save(new_checkpoint, seed_pth)
    print(f"Saved expanded base model to {seed_pth}")

if __name__ == "__main__":
    expand_vocab()
