#!/bin/bash
#SBATCH --job-name=xylemx-predict
#SBATCH --output=jobs/logs/predict-%j.out
#SBATCH --error=jobs/logs/predict-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p jobs/logs output/predictions

echo "Running on host: $(hostname)"
echo "Started at: $(date -u)"

if [ ! -x "./.venv/bin/python" ]; then
  echo "Missing .venv/bin/python. Create the environment first."
  exit 1
fi

./.venv/bin/pip install -e . --no-build-isolation

RUN_NAME="${RUN_NAME:-debug}"
CHECKPOINT="${CHECKPOINT:-output/train_runs/${RUN_NAME}/best.pt}"
SPLIT_NAME="${SPLIT_NAME:-val}"
THRESHOLD="${THRESHOLD:-0.5}"
OUTPUT_DIR="${OUTPUT_DIR:-output/predictions/${RUN_NAME}}"

mkdir -p "$OUTPUT_DIR"

./.venv/bin/python scripts/predict.py \
  checkpoint="${CHECKPOINT}" \
  split="${SPLIT_NAME}" \
  data_root=data/makeathon-challenge \
  preprocessing_dir=output/preprocessing \
  output_dir="${OUTPUT_DIR}" \
  threshold="${THRESHOLD}"

echo "Finished at: $(date -u)"
