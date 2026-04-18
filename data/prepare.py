import os
import sys
import requests
import numpy as np

# add parent directory to path to import tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import CharacterTokenizer

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    print("Downloading Tiny Shakespeare...")
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# create the tokenizer
tokenizer = CharacterTokenizer(data)
print(f"vocab size: {tokenizer.vocab_size}")
print(f"chars: {''.join(tokenizer.chars)}")

# save the tokenizer metadata
tokenizer.save(os.path.join(os.path.dirname(__file__), 'tokenizer.json'))

# encode the data
train_data = data[:int(len(data)*0.9)]
val_data = data[int(len(data)*0.9):]

train_ids = tokenizer.encode(train_data)
val_ids = tokenizer.encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print("Done! train.bin and val.bin created.")
