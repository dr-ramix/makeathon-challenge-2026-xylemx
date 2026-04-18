#!/bin/bash
#SBATCH --job-name=xylemx-train-all
#SBATCH --output=jobs/logs/train-all-%j.out
#SBATCH --error=jobs/logs/train-all-%j.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p jobs/logs output/train_runs/all_models

echo "Running on host: $(hostname)"
echo "Started at: $(date -u)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"

if [ ! -x "./.venv/bin/python" ]; then
  echo "Missing .venv/bin/python. Create the environment first."
  exit 1
fi

./.venv/bin/pip install -e . --no-build-isolation

RUN_PREFIX="${RUN_PREFIX:-all_models}"
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

# Sequential model list taken from the current model registry in src/xylemx/models/baseline.py
MODELS=(
  small_unet
  resnet18_unet
  vgg_unet
  convnext_unet
  convnextv2_unet
  coatnet_unet
)

echo "Training models sequentially:"
printf ' - %s\n' "${MODELS[@]}"

for MODEL in "${MODELS[@]}"; do
  OUTPUT_DIR="output/train_runs/${RUN_PREFIX}/${MODEL}"
  mkdir -p "$OUTPUT_DIR"

  echo
  echo "============================================================"
  echo "Starting model: ${MODEL}"
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

  echo "Finished model: ${MODEL} at $(date -u)"
done

echo "All models finished at: $(date -u)"
