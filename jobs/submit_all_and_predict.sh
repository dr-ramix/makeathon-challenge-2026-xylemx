#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this script on a SLURM login node."
  exit 1
fi

RUN_PREFIX="${RUN_PREFIX:-all_models}"
SPLIT_NAME="${SPLIT_NAME:-test}"
THRESHOLD="${THRESHOLD:-0.5}"

submit_output="$(
  RUN_PREFIX="${RUN_PREFIX}" \
  EPOCHS="${EPOCHS:-20}" \
  BATCH_SIZE="${BATCH_SIZE:-8}" \
  PATCH_SIZE="${PATCH_SIZE:-128}" \
  STRIDE="${STRIDE:-128}" \
  LR="${LR:-1e-3}" \
  WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}" \
  DROPOUT="${DROPOUT:-0.2}" \
  LOSS_NAME="${LOSS_NAME:-bce_dice}" \
  NUM_WORKERS="${NUM_WORKERS:-4}" \
  USE_AMP="${USE_AMP:-true}" \
  OVERSAMPLE_POSITIVES="${OVERSAMPLE_POSITIVES:-true}" \
  PREPROCESSING_DIR="${PREPROCESSING_DIR:-output/preprocessing}" \
  PREPROCESS_FIRST="${PREPROCESS_FIRST:-0}" \
  START_DEPENDENCY="${START_DEPENDENCY:-}" \
  bash jobs/submit_all_models.sh
)"

echo "$submit_output"

final_dependency="$(printf '%s\n' "$submit_output" | awk -F': ' '/Final dependency chain/ {print $2}' | tail -n 1)"

if [ -z "$final_dependency" ]; then
  echo "Could not determine final dependency from submit_all_models.sh output."
  exit 1
fi

predict_job="$(
  sbatch \
    --parsable \
    --dependency="${final_dependency}" \
    --export=ALL,RUN_PREFIX="${RUN_PREFIX}",SPLIT_NAME="${SPLIT_NAME}",THRESHOLD="${THRESHOLD}" \
    jobs/predict_all_models.slurm
)"

echo "Submitted predict_all_models job: ${predict_job}"
echo "Prediction dependency: ${final_dependency}"
