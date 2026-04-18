#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this script on a SLURM login node."
  exit 1
fi

mkdir -p jobs/logs

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
PREPROCESS_FIRST="${PREPROCESS_FIRST:-0}"
START_DEPENDENCY="${START_DEPENDENCY:-}"

MODELS=(
  small_unet
  resnet18_unet
  vgg_unet
  convnext_unet
  convnextv2_unet
  coatnet_unet
)

dependency="$START_DEPENDENCY"

if [ "$PREPROCESS_FIRST" = "1" ]; then
  preprocess_job="$(sbatch --parsable jobs/preprocess.slurm)"
  echo "Submitted preprocess job: ${preprocess_job}"
  dependency="afterok:${preprocess_job}"
fi

echo "Submitting models sequentially:"
printf ' - %s\n' "${MODELS[@]}"

for model in "${MODELS[@]}"; do
  run_name="${RUN_PREFIX}_${model}"
  export_args="ALL,MODEL=${model},RUN_NAME=${run_name},EPOCHS=${EPOCHS},BATCH_SIZE=${BATCH_SIZE},PATCH_SIZE=${PATCH_SIZE},STRIDE=${STRIDE},LR=${LR},WEIGHT_DECAY=${WEIGHT_DECAY},DROPOUT=${DROPOUT},LOSS_NAME=${LOSS_NAME},NUM_WORKERS=${NUM_WORKERS},USE_AMP=${USE_AMP},OVERSAMPLE_POSITIVES=${OVERSAMPLE_POSITIVES},PREPROCESSING_DIR=${PREPROCESSING_DIR}"

  if [ -n "$dependency" ]; then
    job_id="$(sbatch --parsable --dependency="$dependency" --export="$export_args" jobs/train_baseline.slurm)"
  else
    job_id="$(sbatch --parsable --export="$export_args" jobs/train_baseline.slurm)"
  fi

  echo "Submitted ${model}: ${job_id}"
  dependency="afterok:${job_id}"
done

echo "Final dependency chain: ${dependency}"
echo "Tip: to chain prediction after the last model, run:"
echo "START_DEPENDENCY=${dependency} sbatch --dependency=${dependency} jobs/predict_all_models.slurm"
