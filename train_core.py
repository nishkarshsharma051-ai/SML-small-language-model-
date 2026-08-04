"""
Fine-tune a pretrained Hugging Face causal LM on Ting Ling Ling instruction data.

This script turns the local assistant into a more GPT-like model by starting
from a pretrained foundation model and adapting it with the small instruction
dataset generated from the repo's knowledge base.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None
    TaskType = None
    get_peft_model = None


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "hf_local_model"
SYSTEM_PROMPT = (
    "You are Ting Ling Ling, a helpful, reliable, general-purpose assistant. "
    "Answer coding, math, English, history, science, and everyday questions clearly. "
    "For science and astronomy questions, answer directly instead of refusing."
)


def build_prompt(instruction: str) -> str:
    return (
        f"System: {SYSTEM_PROMPT}\n"
        f"User: {instruction.strip()}\n"
        "Assistant:"
    )


def tokenize_example(example: Dict[str, str], tokenizer, max_length: int):
    prompt = example["prompt"].strip()
    response = example["response"].strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = build_prompt(prompt)
        full_text = f"{prompt_text} {response}{tokenizer.eos_token or ''}"

    prompt_ids = tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"]
    encoded = tokenizer(full_text, truncation=True, max_length=max_length)

    labels = list(encoded["input_ids"])
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    encoded["labels"] = labels
    return encoded


def make_collator(tokenizer):
    def collate(features: List[Dict[str, List[int]]]):
        labels = [feature.pop("labels") for feature in features]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = [
            label + [-100] * (max_len - len(label))
            for label in labels
        ]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch

    return collate


def maybe_apply_lora(model, model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    if LoraConfig is None or get_peft_model is None:
        raise RuntimeError("peft is not installed. Install it or run without --use-lora.")

    target_modules = None
    lowered = model_name.lower()
    if "gpt2" in lowered:
        target_modules = ["c_attn", "c_proj"]
    elif "qwen" in lowered or "llama" in lowered or "mistral" in lowered or "phi" in lowered:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if target_modules is None:
        raise ValueError(
            f"Don't know which layers to adapt for base model '{model_name}'. "
            "Use a supported model name or edit the target_modules list."
        )

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    return get_peft_model(model, config)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Ting Ling Ling with Hugging Face")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="Hugging Face base model to fine-tune.")
    parser.add_argument("--train-file", default="data/hf_sft_train.jsonl")
    parser.add_argument("--eval-file", default="data/hf_sft_val.jsonl")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Optional hard cap on training steps for short sanity runs.")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--dataset-cache-dir", default=None,
                        help="Optional datasets cache directory for sandbox-friendly local runs.")
    parser.add_argument("--local-only", action="store_true",
                        help="Load model/tokenizer only from local files.")
    parser.add_argument("--use-cpu", action="store_true",
                        help="Force CPU training for environments where accelerator training is unstable.")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true",
                        help="Disable gradient checkpointing for simpler short sanity runs.")
    parser.add_argument("--use-lora", action="store_true",
                        help="Use LoRA adapters instead of full fine-tuning.")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push the final model to the Hugging Face Hub if authenticated.")
    parser.add_argument("--hub-model-id", default=None,
                        help="Optional Hub repo id, e.g. your-name/ting-ling-ling.")
    args = parser.parse_args()

    if not os.path.exists(args.train_file):
        raise FileNotFoundError(
            f"{args.train_file} not found. Run `python3 data_builder.py` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=args.local_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=args.local_only,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if not args.disable_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.use_lora:
        model = maybe_apply_lora(model, args.base_model, lora_r=8, lora_alpha=16, lora_dropout=0.05)

    train_ds = load_dataset("json", data_files=args.train_file, split="train", cache_dir=args.dataset_cache_dir)
    eval_ds = (
        load_dataset("json", data_files=args.eval_file, split="train", cache_dir=args.dataset_cache_dir)
        if os.path.exists(args.eval_file) else None
    )

    def map_fn(example):
        return tokenize_example(example, tokenizer, args.max_length)

    train_ds = train_ds.map(map_fn, remove_columns=train_ds.column_names)
    if eval_ds is not None and len(eval_ds) > 0:
        eval_ds = eval_ds.map(map_fn, remove_columns=eval_ds.column_names)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None and len(eval_ds) > 0 else "no",
        eval_steps=args.eval_steps if eval_ds is not None and len(eval_ds) > 0 else None,
        fp16=torch.cuda.is_available(),
        bf16=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        do_train=True,
        do_eval=eval_ds is not None and len(eval_ds) > 0,
        use_cpu=args.use_cpu,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(tokenizer),
    )

    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
