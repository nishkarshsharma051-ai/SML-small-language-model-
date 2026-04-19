import os
import sys
import numpy as np

# add parent directory to path to import tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import CharacterTokenizer

# input file path
input_file_path = os.path.join(os.path.dirname(__file__), 'scholarly_input.txt')

if not os.path.exists(input_file_path):
    print(f"Error: {input_file_path} not found.")
    sys.exit(1)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of scholarly dataset in characters: {len(data):,}")

# create the tokenizer (re-using the same character set is safer)
# for fine-tuning we MUST use the expanded tokenizer
tokenizer_json = os.path.join(os.path.dirname(__file__), 'tokenizer_scholarly.json')
if os.path.exists(tokenizer_json):
    print("Loading scholarly tokenizer...")
    tokenizer = CharacterTokenizer.load(tokenizer_json)
else:
    print(f"Error: {tokenizer_json} not found. Run expand_vocab.py first.")
    sys.exit(1)

# encode the data
train_data = data[:int(len(data)*0.9)]
val_data = data[int(len(data)*0.9):]

train_ids = tokenizer.encode(train_data)
val_ids = tokenizer.encode(val_data)
print(f"scholarly train has {len(train_ids):,} tokens")
print(f"scholarly val has {len(val_ids):,} tokens")

# export to bin files - distinct from original tiny shakespeare ones
# so we don't accidentally trash them
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'scholarly_train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'scholarly_val.bin'))

print("Done! scholarly_train.bin and scholarly_val.bin created.")
