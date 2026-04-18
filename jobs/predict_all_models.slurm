#!/bin/bash
#SBATCH --job-name=xylemx-predict-all
#SBATCH --output=jobs/logs/predict-all-%j.out
#SBATCH --error=jobs/logs/predict-all-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p jobs/logs output/predictions

echo "Running on host: $(hostname)"
echo "Started at: $(date -u)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"

if [ ! -x "./.venv/bin/python" ]; then
  echo "Missing .venv/bin/python. Create the environment first."
  exit 1
fi

./.venv/bin/pip install -e . --no-build-isolation

RUN_PREFIX="${RUN_PREFIX:-all_models}"
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

echo "Generating predictions sequentially for models:"
printf ' - %s\n' "${MODELS[@]}"

for MODEL in "${MODELS[@]}"; do
  CHECKPOINT="output/train_runs/${RUN_PREFIX}/${MODEL}/best.pt"
  OUTPUT_DIR="output/predictions/${RUN_PREFIX}/${MODEL}"

  if [ ! -f "$CHECKPOINT" ]; then
    echo "Skipping ${MODEL}: missing checkpoint ${CHECKPOINT}"
    continue
  fi

  mkdir -p "$OUTPUT_DIR"

  echo
  echo "============================================================"
  echo "Predicting model: ${MODEL}"
  echo "Checkpoint: ${CHECKPOINT}"
  echo "Output dir: ${OUTPUT_DIR}"
  echo "============================================================"

  ./.venv/bin/python scripts/predict.py \
    checkpoint="${CHECKPOINT}" \
    split="${SPLIT_NAME}" \
    data_root=data/makeathon-challenge \
    preprocessing_dir=output/preprocessing \
    output_dir="${OUTPUT_DIR}" \
    threshold="${THRESHOLD}"

  echo "Finished predictions for ${MODEL} at $(date -u)"
done

echo "All prediction jobs finished at: $(date -u)"
