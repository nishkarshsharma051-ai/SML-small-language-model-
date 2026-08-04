#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

echo "--- PHASE 1: Data Generation ---"
python3 generate_scholarly_data.py
python3 data_builder.py

echo "--- PHASE 2: Tokenizer & Base Model Update ---"
python3 expand_vocab.py
python3 data/prepare_scholarly.py

echo "--- PHASE 3: Custom SLM Fine-Tuning ---"
# We'll run this for a shorter duration if we just want a quick update
# but the fine_tune.py is already running in the background.
# If you want to run it fully, uncomment the next line.
# python3 fine_tune.py

echo "--- PHASE 4: Hugging Face High-Fidelity Fine-Tuning ---"
# This will train the Qwen 0.5B model on the new coding data
python3 train_core.py --epochs 5 --batch-size 1 --grad-accum 16 --use-lora --output-dir hf_local_model

echo "--- Training Pipeline Complete ---"
