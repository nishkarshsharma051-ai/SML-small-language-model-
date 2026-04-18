from tokenizer_hf import TingLingLingTokenizer
import json

with open('data/tokenizer.json', 'r') as f:
    data = json.load(f)

t = TingLingLingTokenizer(chars=data['chars'])
text = "TING LING LING"
ids = t.encode(text)
decoded = t.decode(ids)

print(f"Original: {text}")
print(f"IDs: {ids}")
print(f"Decoded: {decoded}")

assert text == decoded
print("Tokenizer test passed!")
