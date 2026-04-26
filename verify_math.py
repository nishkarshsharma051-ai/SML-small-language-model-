import torch
from model import TingLingLing, TingLingLingConfig
from tokenizer import CharacterTokenizer

def test_math():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    ckpt_path = 'ting_ling_ling.pth'
    
    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TingLingLing(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    tokenizer_path = checkpoint.get('tokenizer_path', 'data/tokenizer_scholarly.json')
    tokenizer = CharacterTokenizer.load(tokenizer_path)
    
    prompts = [
        "ADVANCED CALCULUS: Stokes Theorem",
        "LINEAR ALGEBRA: Eigenvalues",
        "MATH PROBLEMS: Solve the first-order linear differential equation: dy/dx + 2y = e^x.",
        "DISCRETE MATH: Graph Theory"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated = model.generate(context, max_new_tokens=200)[0].tolist()
        print(f"Response: {tokenizer.decode(generated)}")

if __name__ == "__main__":
    test_math()
