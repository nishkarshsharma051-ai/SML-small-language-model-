"""
Quick local evaluation harness for the fine-tuned Hugging Face model.

This is not a full benchmark suite. It is a repeatable smoke test that checks
whether the local HF model can answer across coding, math, writing, and general
conversation in a ChatGPT-like way.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_DIR = "hf_local_model"


@dataclass
class EvalCase:
    name: str
    prompt: str
    kind: str
    checks: List[Callable[[str], bool]]


def _has_code(text: str) -> bool:
    return "```" in text


def _has_math_symbol(text: str) -> bool:
    return any(sym in text for sym in ["∑", "∫", "∂", "⇒", "≤", "≥", "≈", "$", "\\("])


def _has_explanation(text: str) -> bool:
    return len(text.split()) >= 40


def _has_palindrome_code(text: str) -> bool:
    return "palindrome" in text.lower() and _has_code(text)


def _has_stepwise_math(text: str) -> bool:
    lower = text.lower()
    return (
        _has_math_symbol(text)
        and ("step" in lower or "therefore" in lower or "because" in lower or "simpler terms" in lower)
    ) or ("f'(g(x))" in lower and "g'(x)" in lower) or ("h'(x)" in lower and "f'(g(x))" in lower)


def _has_general_answer(text: str) -> bool:
    return len(text.split()) >= 30 and not text.strip().startswith("I don't know")


def build_cases() -> List[EvalCase]:
    return [
        EvalCase(
            name="coding_palindrome",
            prompt="Write a Python function that checks whether a string is a palindrome.",
            kind="code",
            checks=[_has_code, _has_palindrome_code],
        ),
        EvalCase(
            name="coding_oop",
            prompt="Explain object-oriented programming with a Python example.",
            kind="code",
            checks=[_has_code, _has_explanation],
        ),
        EvalCase(
            name="math_chain_rule",
            prompt="Explain the chain rule in calculus and why it works.",
            kind="math",
            checks=[_has_stepwise_math, _has_explanation],
        ),
        EvalCase(
            name="math_linear_algebra",
            prompt="Explain matrices and vectors in linear algebra using symbols.",
            kind="math",
            checks=[_has_math_symbol, _has_explanation],
        ),
        EvalCase(
            name="writing_email",
            prompt="Help me write a polite professional email asking for a project update.",
            kind="writing",
            checks=[_has_general_answer],
        ),
        EvalCase(
            name="general_tesla",
            prompt="Who was Nikola Tesla and why is he important?",
            kind="general",
            checks=[_has_general_answer],
        ),
        EvalCase(
            name="science_blackholes",
            prompt="How do black holes form?",
            kind="general",
            checks=[_has_general_answer],
        ),
    ]


def generate_answer(model, tokenizer, prompt: str, device: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are Ting Ling Ling, a helpful general-purpose assistant. "
                "Answer science and astronomy questions directly instead of refusing."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = (
            "System: You are a helpful general-purpose assistant. "
            "Answer science and astronomy questions directly instead of refusing.\n"
            f"User: {prompt}\nAssistant:"
        )

    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate the local HF model.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(
            f"{args.model_dir} does not exist. Train the model first with hf_train.py."
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        local_files_only=True,
        dtype=dtype,
    ).to(device)
    model.eval()

    cases = build_cases()
    passed = 0

    print(f"Evaluating model at: {args.model_dir}")
    print(f"Device: {device}")
    print("-" * 72)

    for case in cases:
        answer = generate_answer(model, tokenizer, case.prompt, device)
        ok = all(check(answer) for check in case.checks)
        passed += int(ok)
        print(f"[{ 'PASS' if ok else 'FAIL' }] {case.name}")
        print(f"Prompt: {case.prompt}")
        print(answer[:500].replace("\n", " "))
        print("-" * 72)

    total = len(cases)
    print(f"Score: {passed}/{total}")


if __name__ == "__main__":
    main()
