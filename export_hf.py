import os
import torch
import json

def export():
    checkpoint_path = 'ting_ling_ling_checkpoints/stable_checkpoint.pth'
    export_dir = './ting-ling-ling-hf-final'
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found. Please run train_hf.py first.")
        return

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 1. Create HF config manually (No class inheritance here)
    print("Creating HF config manually...")
    config_data = checkpoint['config']
    hf_config = {
        "architectures": ["TingLingLingModelHF"],
        "model_type": "ting_ling_ling",
        "vocab_size": config_data['vocab_size'],
        "n_embd": config_data['n_embd'],
        "n_layer": config_data['n_layer'],
        "n_head": config_data['n_head'],
        "block_size": config_data['block_size'],
        "transformers_version": "4.35.0"
    }
    
    # 2. Prepare state dict
    # We want to save the state dict in a way that TingLingLingModelHF can load it.
    # Our wrapper saves the model in self.model
    # So the keys in the state dict should be prefixed with 'model.'
    state_dict = checkpoint['model']
    hf_state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    
    # 3. Save everything manually
    print(f"Saving HF format files to {export_dir}...")
    os.makedirs(export_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(export_dir, 'config.json'), 'w') as f:
        json.dump(hf_config, f, indent=2)
        
    # Save weights
    torch.save(hf_state_dict, os.path.join(export_dir, 'pytorch_model.bin'))
    
    # Save tokenizer files
    with open(os.path.join(export_dir, 'vocab.json'), 'w') as f:
        json.dump(checkpoint['tokenizer_chars'], f, indent=2)
    
    # Minimal tokenizer config to make HF happy
    tokenizer_config = {
        "model_max_length": config_data['block_size'],
        "tokenizer_class": "TingLingLingTokenizer"
    }
    with open(os.path.join(export_dir, 'tokenizer_config.json'), 'w') as f:
        json.dump(tokenizer_config, f, indent=2)

    print("-" * 30)
    print(f"SUCCESS! Model exported to {export_dir}")
    print("Files created: config.json, pytorch_model.bin, vocab.json, tokenizer_config.json")
    print("-" * 30)

if __name__ == "__main__":
    export()
