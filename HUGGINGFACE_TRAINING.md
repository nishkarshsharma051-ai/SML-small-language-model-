# Hugging Face Training Path

This repo now supports a real local Hugging Face fine-tune flow.

## Hugging Face

Use the platform here: [Hugging Face](https://huggingface.co/)

## Recommended Flow

1. Build instruction data:

```bash
python3 hf_dataset_builder.py
```

2. Fine-tune a pretrained causal LM:

```bash
python3 hf_train.py --base-model Qwen/Qwen2.5-3B-Instruct --use-lora --output-dir hf_local_model
```

3. Start the app with the local HF model directory in place.

## Notes

- This is a fine-tuning workflow, not training a foundation model from scratch.
- The default training target is `Qwen/Qwen2.5-3B-Instruct`, which is instruction-tuned, multilingual, and designed for stronger coding/math/general chat behavior.
- If your hardware is tight, you can still swap to `Qwen/Qwen2.5-1.5B-Instruct`.
- If your machine is tight on memory, keep `--use-lora` enabled.
- The app will prefer the local Hugging Face model when `hf_local_model/` exists.
