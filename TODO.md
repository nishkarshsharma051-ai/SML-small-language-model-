# Ting Ling Ling Roadmap

This file turns the current repo state into a practical build plan.

## Current Stage

Estimated progress:
- Demoable prototype: `85%`
- Reliable local small assistant: `65-70%`

What is already working:
- Local Flask app with chat, streaming, cancel, and voice endpoints
- Custom decoder-only transformer training path
- Hugging Face fine-tuning path using a pretrained base model
- Instruction dataset generation
- Local HF model directory with a recent LoRA adapter
- Basic smoke-test style evaluation scripts

What is not solid yet:
- Answer quality is inconsistent
- Dataset is still small for instruction tuning
- Evaluation is not rigorous enough
- Repo has multiple overlapping training paths
- Docs and filenames are out of sync
- Local model packaging is not fully portable

## Recommended Direction

Primary path:
- Use the Hugging Face fine-tune path as the main product path

Secondary path:
- Keep the scratch character-level model as an experiment/learning path only

Reason:
- The HF path is much closer to a useful assistant than the from-scratch character model

## Urgent

### 1. Standardize the repo around one main model path

Status:
- Partially done

Why this matters:
- The repo currently mixes a scratch model path, a custom HF export path, and a pretrained HF fine-tune path
- That makes future training and debugging much harder

Tasks:
- Decide that `train_core.py` is the main training path for the product
- Mark `train.py`, `train_hf.py`, `export_hf.py`, and older experimental files as legacy or experimental
- Add a short section in the README explaining which script is the current source of truth
- Move older experiments into an `experiments/` folder later

Success criteria:
- A new contributor can tell in under 2 minutes which script to run for the real model

### 2. Fix documentation drift

Status:
- Not done

Why this matters:
- Current docs reference filenames that do not match the repo
- This creates training mistakes and wasted time

Tasks:
- Update `MODEL_TRAINING.md`
- Update `README.md`
- Replace outdated references like `hf_dataset_builder.py` and `hf_train.py` if they are no longer the intended path
- Add a minimal "train from scratch vs fine-tune HF" comparison section

Success criteria:
- The documented commands match the actual files in the repo

### 3. Build a proper evaluation workflow

Status:
- Partially done

Why this matters:
- Right now you can test outputs, but you do not yet have a stable way to measure progress across runs

Tasks:
- Keep `eval_core.py` as a base
- Add a saved eval set in JSON or JSONL
- Split evals into categories: coding, math, science, history, writing, general chat
- Record pass/fail results per run
- Save a simple results file such as `eval/results/<timestamp>.json`
- Track bad behaviors: refusal, repetition, hallucination, identity leakage

Success criteria:
- Every training run can be compared against the last one using saved metrics

### 4. Expand the instruction dataset

Status:
- Partially done

Current state:
- About `205` total SFT examples across train and val

Why this matters:
- This is enough for a small prototype, but not enough for consistent assistant behavior

Tasks:
- Expand to `1k-3k` high-quality examples first
- Add more coding examples with complete answers and fenced code blocks
- Add more math tutoring examples with step-by-step reasoning and symbols
- Add more general assistant tasks like rewriting, summarization, and email drafting
- Add multi-turn chat examples
- Add short, medium, and long answer variants
- Add "answer directly without refusing" examples for science/history/general knowledge

Success criteria:
- The model stops refusing many normal questions and becomes more stylistically consistent

## Important

### 5. Improve data quality, not just data size

Status:
- Partially done

Why this matters:
- Repeating the same scholarly corpus many times helps memorization, but it does not teach robust instruction following

Tasks:
- Reduce dependence on repeated corpus-style text
- Add paraphrases of the same fact in different styles
- Add explanation/problem/example triples
- Add adversarial prompts that test whether the model stays helpful
- Review and remove weak, overly similar, or noisy examples

Success criteria:
- Training data looks like real assistant conversations, not repeated notes

### 6. Make the local model reproducible on a fresh machine

Status:
- Partially done

Why this matters:
- The current local HF setup is a LoRA adapter and still depends on the base model being available

Tasks:
- Clearly document the base model dependency
- Add a setup note for downloading or caching the base model
- Consider merging the adapter into final weights for easier deployment
- Document where `hf_local_model/` is expected and how the app loads it

Success criteria:
- Another machine can run the local model with predictable setup steps

### 7. Tune inference behavior

Status:
- Partially done

Why this matters:
- The model loads, but answer quality is uneven

Tasks:
- Tune generation settings: `temperature`, `top_p`, `repetition_penalty`, `max_new_tokens`
- Add better fallback handling when the local model gives a bad answer
- Improve refusal handling
- Review the long system prompt and simplify if needed
- Add a few regression prompts you always test after changes

Success criteria:
- Fewer empty, repetitive, or unnecessary refusal answers

### 8. Add performance benchmarks

Status:
- Not done

Why this matters:
- If this is meant to be a small local model, speed and memory need to be measured, not assumed

Tasks:
- Measure load time
- Measure first-token latency
- Measure tokens per second
- Measure memory use on your target machine
- Compare local HF mode vs scratch model mode

Success criteria:
- You can defend the claim that the model is fast enough for local use

## Later

### 9. Reorganize the repo structure

Status:
- Not done

Suggested layout:
- `app/`
- `training/`
- `eval/`
- `data/`
- `models/`
- `experiments/`
- `docs/`

Tasks:
- Move legacy scripts into `experiments/`
- Group evaluation scripts together
- Group production app files together
- Keep root cleaner so the intended workflow is obvious

Success criteria:
- The project reads like a product repo instead of a lab bench

### 10. Strengthen safety and output guardrails

Status:
- Partially done

Why this matters:
- You already clean identity leaks, but there should be broader output quality checks

Tasks:
- Detect empty output
- Detect repetition loops
- Detect contradictory self-identity
- Detect excessive refusal behavior
- Add answer-length sanity checks where helpful

Success criteria:
- Runtime quality control catches more bad outputs before they reach the UI

### 11. Sharpen the product focus

Status:
- Not fully decided

Why this matters:
- Small models get much stronger when optimized for a narrower job

Options:
- Study assistant
- Math tutor
- Coding helper
- General offline assistant

Recommendation:
- Position it first as a study + coding assistant, then widen later

Success criteria:
- Data, prompts, evals, and UI all reflect one clear product goal

## Suggested Execution Order

Do these next in order:

1. Fix docs and declare the main training path
2. Expand the instruction dataset to at least `1k+` strong examples
3. Build a persistent eval set and saved score output
4. Retrain the HF model
5. Tune inference behavior using the eval results
6. Document or package the model for fresh-machine reproducibility
7. Clean and reorganize the repo

## Definition of "Good Enough" for v1

Call v1 successful when all of these are true:
- Local HF model loads reliably
- Normal coding, math, science, and writing prompts usually get helpful answers
- Refusal rate for harmless questions is low
- Evaluation results are saved and improving between runs
- A fresh machine can reproduce the setup
- The repo clearly documents the intended workflow

## Notes

Keep in mind:
- The scratch character-level path is useful for learning, but it is unlikely to beat the pretrained HF path for real assistant quality
- The most valuable next work is better data plus better evals, not more UI polish
