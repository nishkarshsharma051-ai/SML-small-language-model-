import torch
from model_hf import TingLingLingModelHF
from tokenizer_hf import TingLingLingTokenizer

# device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load the saved HF model and tokenizer from the FINAL export directory
model_dir = "./ting-ling-ling-hf-final"

try:
    print(f"Loading final HF model from {model_dir}...")
    tokenizer = TingLingLingTokenizer.from_pretrained(model_dir)
    model = TingLingLingModelHF.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Generate!
    prompt = "TING LING LING:"
    print(f"Generating from prompt: {prompt}")
    print("-" * 30)
    
    # Tokenize input
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        # Generate new tokens
        # Note: temperature and top_k control the "creativity" of the Ting Ling Ling
        output_ids = model.generate(input_ids, max_new_tokens=200, temperature=0.8, top_k=10)
    
    print(tokenizer.decode(output_ids[0].tolist()))
    print("-" * 30)

except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have run 'python3 train_hf.py' followed by 'python3 export_hf.py'.")
