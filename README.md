# Ting Ling Ling

Ting Ling Ling is a local-first small assistant project with a Flask app, voice features, and two model paths:
- an official Hugging Face fine-tune path for the real assistant
- an experimental scratch-built transformer path for learning and comparison

## Current Status

This repo is best understood as a working prototype:
- the app runs locally
- a local HF model path is wired into the runtime
- dataset generation and evaluation scripts exist
- answer quality is still being improved

## Official Training Path

Use this path if you want to train or improve the assistant:

1. Build the instruction dataset:

```bash
python3 data_builder.py
```

2. Fine-tune the local HF model:

```bash
python3 train_core.py --base-model Qwen/Qwen2.5-0.5B-Instruct --use-lora --output-dir hf_local_model
```

3. Run the smoke-test evaluation:

```bash
python3 eval_core.py --model-dir hf_local_model
```

4. Start the app:

```bash
python3 app.py
```

The app will prefer `hf_local_model/` when it exists.

## Experimental Scripts

These scripts are still in the repo, but they should not be treated as the default product workflow:
- `train.py`
- `fine_tune.py`
- `train_hf.py`
- `export_hf.py`
- `sample.py`
- `sample_hf.py`

They are useful for experiments, not as the main training path.

## Project Layout

Important files:
- `app.py`: Flask app entry point
- `brain.py`: hybrid runtime that loads local HF first, then other fallbacks
- `data_builder.py`: instruction dataset generator
- `train_core.py`: main HF fine-tuning script
- `eval_core.py`: quick local evaluation harness
- `MODEL_TRAINING.md`: clearer training notes and workflow details
- `TODO.md`: prioritized roadmap for what to do next

## Setup

### Prerequisites

- Python 3.9+
- macOS is helpful for local voice support via `say`, though the web app can still run elsewhere

### Install

```bash
pip install -r requirements.txt
```

### Optional Environment Variables

Create a `.env` file if you want cloud fallback or teacher logging:
- `CLOUD_API_KEY`
- `CLOUD_MODEL_ID`
- `CLOUD_ENDPOINT`
- `LOCAL_HF_MODEL_DIR`
- `TEACHER_LOG`

## Model Notes

The current official local path is based on fine-tuning a pretrained model, not training a foundation model from scratch.

Default target:
- `Qwen/Qwen2.5-0.5B-Instruct`

If you train with LoRA:
- `hf_local_model/` contains adapter weights and tokenizer/config files
- the base model still needs to be available locally

## Voice and UI

The web app includes:
- chat
- streaming responses
- cancel support
- speech output
- browser-side speech input support

## Recommended Next Work

The highest-value next steps are:
- expand the instruction dataset
- improve evaluation and saved metrics
- clean up legacy scripts further
- make fresh-machine setup more reproducible

## Credits

Developed and maintained by Nishkarsh Sharma.
