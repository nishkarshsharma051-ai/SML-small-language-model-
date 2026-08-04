#!/usr/bin/env python3
"""
Ting Ling Ling — HF Inference + Male Voice Output
===================================================
Run: python3 sample_hf_stable.py [--voice daniel|reed|rocko|grandpa|eddy|fred]
"""

import torch
import json
import os
import argparse
from voice_model import VoiceModel, speak_text

# ─── Arg Parsing ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Ting Ling Ling Text Generator + Voice")
parser.add_argument("--voice", type=str, default="daniel",
                    choices=["daniel", "reed", "rocko", "grandpa", "eddy", "fred"],
                    help="Male voice to use for TTS output (default: daniel)")
parser.add_argument("--tokens", type=int, default=300, help="Number of tokens to generate (default: 300)")
parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
parser.add_argument("--topk", type=int, default=10, help="Top-K sampling (default: 10)")
parser.add_argument("--prompt", type=str, default="TING LING LING:", help="Prompt to start generation")
parser.add_argument("--no-voice", action="store_true", help="Disable voice output (text only)")
args = parser.parse_args()

from model import TingLingLing, TingLingLingConfig
from tokenizer import CharacterTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_dir = "./ting-ling-ling-hf-final"


def sample():
    try:
        print("═" * 55)
        print("   🤖  Ting Ling Ling — Small Language Model (SLM)")
        print("═" * 55)

        # 1. Load config
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config_data = json.load(f)

        # 2. Load vocab
        with open(os.path.join(model_dir, 'vocab.json'), 'r') as f:
            chars = json.load(f)

        # 3. Initialize stable model components
        tokenizer = CharacterTokenizer(chars)
        config = TingLingLingConfig(
            vocab_size=config_data['vocab_size'],
            n_embd=config_data['n_embd'],
            n_layer=config_data['n_layer'],
            n_head=config_data['n_head'],
            block_size=config_data['block_size']
        )
        model = TingLingLing(config)

        # 4. Load HF-formatted weights
        print(f"   Loading model weights from {model_dir}...")
        state_dict = torch.load(os.path.join(model_dir, 'pytorch_model.bin'), map_location=device)
        new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()
        print(f"   Model loaded on: {device.upper()}")

        # 5. Generate text
        print(f"\n   Prompt  : {args.prompt}")
        print(f"   Tokens  : {args.tokens}  |  Temp: {args.temp}  |  Top-K: {args.topk}")
        print("─" * 55)

        input_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=args.tokens,
                                        temperature=args.temp, top_k=args.topk)

        generated_text = tokenizer.decode(output_ids[0].tolist())
        print(generated_text)
        print("─" * 55)

        # 6. Voice output
        if not args.no_voice:
            print(f"\n🎙️  Voice Model: {args.voice.capitalize()}")
            vm = VoiceModel(voice_name=args.voice, rate=155)
            vm.list_voices()
            vm.speak(generated_text, label="Ting Ling Ling")
            print("\n✅ Voice output complete!")
        else:
            print("\n[Voice output disabled]")

    except Exception as e:
        import traceback
        print(f"\n❌ Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    sample()
