# Model Training Guide

This repo has two model paths, but only one of them should be treated as the main product workflow.

## Official Path

Use the Hugging Face fine-tune path for the real assistant:
- Build instruction data with `data_builder.py`
- Fine-tune a pretrained causal LM with `train_core.py`
- Evaluate with `eval_core.py`
- Run the app with the resulting `hf_local_model/`

Why this is the official path:
- It produces much stronger assistant behavior than the scratch character model
- It is the path the app prefers when `hf_local_model/` exists
- It is the best foundation for future dataset growth and evaluation

## Experimental Path

These files are still useful, but they are experimental or legacy:
- `train.py`
- `fine_tune.py`
- `train_hf.py`
- `export_hf.py`
- `sample.py`
- `sample_hf.py`

Use them for learning or comparison, not as the default product workflow.

## Recommended Workflow

### 1. Build the instruction dataset

```bash
python3 data_builder.py
```

This writes:
- `data/hf_sft_train.jsonl`
- `data/hf_sft_val.jsonl`

The builder can also include up to 200 examples from `data/teacher_log.jsonl` if teacher logging is enabled elsewhere in the app.

### 2. Fine-tune the local HF model

Recommended command:

```bash
python3 train_core.py --base-model Qwen/Qwen2.5-0.5B-Instruct --use-lora --output-dir hf_local_model
```

Useful notes:
- `--use-lora` is recommended for local hardware
- The default base model is `Qwen/Qwen2.5-0.5B-Instruct`
- You can change the output directory, but the app expects `hf_local_model/` by default

### 3. Evaluate the model

```bash
python3 eval_core.py --model-dir hf_local_model
```

This is currently a smoke-test style evaluation across:
- coding
- math
- writing
- general knowledge

It is useful for quick regression checks after training runs.

### 4. Start the app

```bash
python3 app.py
```

When `hf_local_model/` exists, the app tries to load that local HF model first.

## Model Dependency Note

If you train with LoRA, `hf_local_model/` contains adapter weights and metadata, not a fully standalone base model package.

That means:
- the correct base model must be available locally
- the adapter and tokenizer files must remain together
- fresh-machine setup should document how the base model is obtained

The current adapter is configured for:
- `Qwen/Qwen2.5-0.5B-Instruct`

## Minimal End-to-End Commands

```bash
python3 data_builder.py
python3 train_core.py --base-model Qwen/Qwen2.5-0.5B-Instruct --use-lora --output-dir hf_local_model
python3 eval_core.py --model-dir hf_local_model
python3 app.py
```

## Next Improvements

The most valuable next work is:
- expanding the instruction dataset
- improving evaluation quality and saved metrics
- documenting fresh-machine reproducibility
- reducing confusion from older overlapping scripts
