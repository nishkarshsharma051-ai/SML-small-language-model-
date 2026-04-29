"""
Reusable local evaluation harness for the fine-tuned Hugging Face model.

The eval set lives on disk so runs are comparable over time, and each run writes
its results to eval/results/.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_DIR = "hf_local_model"
DEFAULT_EVAL_SET = os.path.join("eval", "eval_set.jsonl")
DEFAULT_RESULTS_DIR = os.path.join("eval", "results")


@dataclass
class EvalCase:
    name: str
    prompt: str
    kind: str
    checks: List[str]


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


def _not_refusal(text: str) -> bool:
    lower = text.strip().lower()
    refusal_markers = [
        "i'm sorry, but i can't assist with that",
        "i cannot help with that",
        "i can't help with that",
        "i'm sorry, but i cannot",
    ]
    return not any(marker in lower for marker in refusal_markers)


def _not_too_short(text: str) -> bool:
    return len(text.strip()) >= 20


def _has_step_markers(text: str) -> bool:
    lower = text.lower()
    markers = ["step", "first", "then", "next", "finally"]
    return any(marker in lower for marker in markers)


def _solves_simple_linear_equation(text: str) -> bool:
    normalized = text.replace(" ", "").lower()
    return "x=6" in normalized or "x = 6" in text.lower()


def _is_professional_rewrite(text: str) -> bool:
    lower = text.strip().lower()
    positive_markers = [
        "please",
        "could you",
        "would you",
        "earliest convenience",
        "kindly",
    ]
    blunt_phrases = [
        "i need it now",
        "send it now",
        "need the file now",
    ]
    return any(marker in lower for marker in positive_markers) and not any(
        phrase in lower for phrase in blunt_phrases
    )


CHECKS: Dict[str, Callable[[str], bool]] = {
    "has_code": _has_code,
    "has_math_symbol": _has_math_symbol,
    "has_explanation": _has_explanation,
    "has_palindrome_code": _has_palindrome_code,
    "has_stepwise_math": _has_stepwise_math,
    "has_general_answer": _has_general_answer,
    "not_refusal": _not_refusal,
    "not_too_short": _not_too_short,
    "has_step_markers": _has_step_markers,
    "solves_simple_linear_equation": _solves_simple_linear_equation,
    "is_professional_rewrite": _is_professional_rewrite,
}


def load_cases(path: str) -> List[EvalCase]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist. Create the eval set first."
        )

    cases: List[EvalCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [key for key in ("name", "prompt", "kind", "checks") if key not in row]
            if missing:
                raise ValueError(f"Eval row is missing keys: {missing}")
            for check_name in row["checks"]:
                if check_name not in CHECKS:
                    raise ValueError(f"Unknown eval check: {check_name}")
            cases.append(
                EvalCase(
                    name=str(row["name"]),
                    prompt=str(row["prompt"]),
                    kind=str(row["kind"]),
                    checks=[str(item) for item in row["checks"]],
                )
            )
    return cases


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


def save_results(results_dir: str, payload: Dict[str, object]) -> str:
    os.makedirs(results_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(results_dir, f"{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="Evaluate the local HF model.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        raise FileNotFoundError(
            f"{args.model_dir} does not exist. Train the model first with train_core.py."
        )

    cases = load_cases(args.eval_set)
    if not cases:
        raise ValueError(f"{args.eval_set} is empty.")

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

    passed = 0
    case_results = []

    print(f"Evaluating model at: {args.model_dir}")
    print(f"Using eval set: {args.eval_set}")
    print(f"Device: {device}")
    print("-" * 72)

    for case in cases:
        answer = generate_answer(model, tokenizer, case.prompt, device)
        check_results = {
            check_name: CHECKS[check_name](answer)
            for check_name in case.checks
        }
        ok = all(check_results.values())
        passed += int(ok)
        case_results.append(
            {
                "name": case.name,
                "kind": case.kind,
                "prompt": case.prompt,
                "passed": ok,
                "checks": check_results,
                "answer": answer,
            }
        )

        print(f"[{'PASS' if ok else 'FAIL'}] {case.name} ({case.kind})")
        print(f"Prompt: {case.prompt}")
        print(answer[:500].replace("\n", " "))
        print("-" * 72)

    total = len(cases)
    score = passed / total
    results_payload = {
        "model_dir": args.model_dir,
        "eval_set": args.eval_set,
        "device": device,
        "passed": passed,
        "total": total,
        "score": score,
        "results": case_results,
    }
    results_path = save_results(args.results_dir, results_payload)

    print(f"Score: {passed}/{total} ({score:.1%})")
    print(f"Saved results to: {results_path}")


if __name__ == "__main__":
    main()
