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

SYSTEM_PROMPT = "You are Ting Ling Ling, a highly intelligent and helpful AI assistant."

def tokenize_example(example, tokenizer, max_length):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]}
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        prompt_text = f"System: {SYSTEM_PROMPT}\nUser: {example['prompt']}\nAssistant: {example['response']}"
        
    return tokenizer(prompt_text, truncation=True, max_length=max_length, padding="max_length")

def make_collator(tokenizer):
    from transformers import DataCollatorForLanguageModeling
    return DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAIN_FILE = "data/hf_sft_train.jsonl"
LOG_FILE = "data/teacher_log.jsonl"
OUTPUT_DIR = "hf_local_model"
NUM_EPOCHS = 3 # Train for 3 full passes over new data

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
        
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    from peft import PeftModel, get_peft_model, LoraConfig
    if os.path.exists(os.path.join(OUTPUT_DIR, "adapter_config.json")):
        print("[Auto-Tune] Resuming LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR, is_trainable=True)
    else:
        print("[Auto-Tune] Initializing new LoRA adapter...")
        peft_config = LoraConfig(task_type="CAUSAL_LM", r=8, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(base_model, peft_config)
    
    # Enable gradient checkpointing for memory efficiency if needed
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    # Only train on the newest data if possible, or just the whole set for a few steps
    dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
    
    # Map tokenization
    dataset = dataset.map(lambda x: tokenize_example(x, tokenizer, 512), remove_columns=dataset.column_names)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=1,
        learning_rate=2e-5,
        logging_steps=5,
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
