#!/bin/zsh
set -euo pipefail

MODEL_ROOT="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
REF_FILE="$MODEL_ROOT/refs/main"

if [[ ! -f "$REF_FILE" ]]; then
  echo "Missing cached Qwen base model ref at: $REF_FILE"
  echo "Download or cache the base model first, then rerun this script."
  exit 1
fi

SNAPSHOT_ID="$(cat "$REF_FILE")"
SNAPSHOT_PATH="$MODEL_ROOT/snapshots/$SNAPSHOT_ID"

if [[ ! -d "$SNAPSHOT_PATH" ]]; then
  echo "Missing cached Qwen snapshot at: $SNAPSHOT_PATH"
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

./.venv/bin/python train_core.py \
  --base-model "$SNAPSHOT_PATH" \
  --local-only \
  --use-cpu \
  --use-lora \
  --disable-gradient-checkpointing \
  --dataset-cache-dir .hf_datasets_cache \
  --output-dir hf_local_model_sanity \
  --batch-size 1 \
  --grad-accum 2 \
  --max-steps 1 \
  --logging-steps 1 \
  --save-steps 1 \
  --eval-steps 1
