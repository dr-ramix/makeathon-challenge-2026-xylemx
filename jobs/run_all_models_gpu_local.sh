#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p jobs/logs output/train_runs output/predictions

if [ ! -x "./.venv/bin/python" ]; then
  echo "Missing .venv/bin/python. Create the environment first."
  exit 1
fi

if ! ./.venv/bin/python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 1)"; then
  echo "CUDA is not available to PyTorch in this environment."
  exit 1
fi

RUN_PREFIX="${RUN_PREFIX:-all_models_gpu_local}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PATCH_SIZE="${PATCH_SIZE:-128}"
STRIDE="${STRIDE:-128}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DROPOUT="${DROPOUT:-0.2}"
LOSS_NAME="${LOSS_NAME:-bce_dice}"
NUM_WORKERS="${NUM_WORKERS:-4}"
USE_AMP="${USE_AMP:-true}"
OVERSAMPLE_POSITIVES="${OVERSAMPLE_POSITIVES:-true}"
PREPROCESSING_DIR="${PREPROCESSING_DIR:-output/preprocessing}"
PREPROCESS_FIRST="${PREPROCESS_FIRST:-0}"
PREDICT_AFTER="${PREDICT_AFTER:-0}"
SPLIT_NAME="${SPLIT_NAME:-test}"
THRESHOLD="${THRESHOLD:-0.5}"

MODELS=(
  small_unet
  resnet18_unet
  vgg_unet
  convnext_unet
  convnextv2_unet
  coatnet_unet
)

echo "Running local GPU multi-model training on host: $(hostname)"
echo "Started at: $(date -u)"
echo "Run prefix: ${RUN_PREFIX}"
echo "Epochs per model: ${EPOCHS}"
echo "Models:"
printf ' - %s\n' "${MODELS[@]}"

./.venv/bin/pip install -e . --no-build-isolation

if [ "$PREPROCESS_FIRST" = "1" ]; then
  echo
  echo "============================================================"
  echo "Running preprocessing first"
  echo "============================================================"
  ./.venv/bin/python scripts/preprocess.py \
    --data-root data/makeathon-challenge \
    --output-dir "${PREPROCESSING_DIR}" \
    --val-ratio 0.2 \
    --split-seed 42 \
    --consensus-mode agreement \
    --min-positive-sources 2
fi

for MODEL in "${MODELS[@]}"; do
  OUTPUT_DIR="output/train_runs/${RUN_PREFIX}/${MODEL}"
  mkdir -p "$OUTPUT_DIR"

  echo
  echo "============================================================"
  echo "Starting local GPU training for model: ${MODEL}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "============================================================"

  ./.venv/bin/python scripts/train.py \
    model="${MODEL}" \
    batch_size="${BATCH_SIZE}" \
    patch_size="${PATCH_SIZE}" \
    stride="${STRIDE}" \
    epochs="${EPOCHS}" \
    lr="${LR}" \
    weight_decay="${WEIGHT_DECAY}" \
    dropout="${DROPOUT}" \
    num_workers="${NUM_WORKERS}" \
    use_amp="${USE_AMP}" \
    loss="${LOSS_NAME}" \
    oversample_positives="${OVERSAMPLE_POSITIVES}" \
    preprocessing_dir="${PREPROCESSING_DIR}" \
    train_split="${PREPROCESSING_DIR}/train_tiles.json" \
    val_split="${PREPROCESSING_DIR}/val_tiles.json" \
    data_root=data/makeathon-challenge \
    output_dir="${OUTPUT_DIR}"

  if [ "$PREDICT_AFTER" = "1" ]; then
    CHECKPOINT="${OUTPUT_DIR}/best.pt"
    PRED_OUTPUT_DIR="output/predictions/${RUN_PREFIX}/${MODEL}"
    mkdir -p "$PRED_OUTPUT_DIR"

    echo
    echo "Predicting for model: ${MODEL}"
    ./.venv/bin/python scripts/predict.py \
      checkpoint="${CHECKPOINT}" \
      split="${SPLIT_NAME}" \
      data_root=data/makeathon-challenge \
      preprocessing_dir="${PREPROCESSING_DIR}" \
      output_dir="${PRED_OUTPUT_DIR}" \
      threshold="${THRESHOLD}"
  fi

  echo "Finished model: ${MODEL} at $(date -u)"
done

echo "All local GPU model runs finished at: $(date -u)"
