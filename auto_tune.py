import os
import json
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset
from hf_train import tokenize_example, make_collator, SYSTEM_PROMPT

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "data/hf_sft_train.jsonl"
LOG_FILE = "data/teacher_log.jsonl"
OUTPUT_DIR = "hf_local_model"
MAX_STEPS = 10 # Extremely fast fine-tuning for single/few samples

def main():
    if not os.path.exists(LOG_FILE):
        print("[Auto-Tune] No teacher log found. Nothing to learn.")
        return

    # 1. Sync Teacher Log to Training Dataset
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        log_entries = [json.loads(line) for line in f]
    
    # Convert log entries to training format and append to main training file
    with open(TRAIN_FILE, 'a', encoding='utf-8') as f:
        for entry in log_entries:
            # We use prompt/response keys expected by hf_train
            train_entry = {
                "prompt": entry["prompt"],
                "response": entry["response"]
            }
            f.write(json.dumps(train_entry, ensure_ascii=False) + "\n")
    
    # Clear the log so we don't re-process the same entries next time
    os.remove(LOG_FILE)
    print(f"[Auto-Tune] Merged {len(log_entries)} new samples into training set.")

    # 2. Run Micro-Training
    print("[Auto-Tune] Starting micro-training session...")
    
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR) # Start from current local model
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
    
    # Only train on the newest data if possible, or just the whole set for a few steps
    dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    
    # Map tokenization
    dataset = dataset.map(lambda x: tokenize_example(x, tokenizer, 512), remove_columns=dataset.column_names)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        learning_rate=2e-5,
        logging_steps=1,
        save_strategy="no",
        fp16=torch.backends.mps.is_available() or torch.cuda.is_available(),
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=make_collator(tokenizer)
    )
    
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[Auto-Tune] Local brain evolved successfully.")

if __name__ == "__main__":
    main()
