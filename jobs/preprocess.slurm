#!/bin/bash
#SBATCH --job-name=xylemx-preprocess
#SBATCH --output=jobs/logs/preprocess-%j.out
#SBATCH --error=jobs/logs/preprocess-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

mkdir -p jobs/logs output/preprocessing

echo "Running on host: $(hostname)"
echo "Started at: $(date -u)"
echo "Repo root: $REPO_ROOT"

if [ ! -x "./.venv/bin/python" ]; then
  echo "Missing .venv/bin/python. Create the environment first."
  exit 1
fi

# Editable install is helpful on cluster nodes where the environment is reused.
./.venv/bin/pip install -e . --no-build-isolation

./.venv/bin/python scripts/preprocess.py \
  --data-root data/makeathon-challenge \
  --output-dir output/preprocessing \
  --val-ratio 0.2 \
  --split-seed 42 \
  --consensus-mode agreement \
  --min-positive-sources 2

echo "Finished at: $(date -u)"
