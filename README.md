# XylemX - osapiens Makeathon 2026

## Deforestation Detection from Space

![Deforestation event example](content/deforestation.png)

This repository contains team **xylemx**'s solution for the osapiens Makeathon 2026 challenge.
We build geospatial ML pipelines that detect deforestation events from multimodal satellite data (Sentinel-2, Sentinel-1, AEF embeddings) with weak supervision.

## Core Idea: Systematic Experimentation

The project idea was to **systematically experiment with different ideas**, not just train one model once.

Our process is structured as repeatable experiment loops:

1. Define one hypothesis (feature representation, label fusion rule, model family, or training setting).
2. Run a consistent preprocessing + training + evaluation pipeline.
3. Compare runs using the same metrics and artifacts.
4. Keep what generalizes, drop what does not.

Main experimentation axes:

- Data representation: `snapshot_pair`, `snapshot_quad`, temporal windows, single-year snapshots.
- Supervision strategy: weak-label fusion modes (`consensus_2of3`, `union`, `soft_vote`, etc.).
- Architecture choice: U-Net/FPN/UNet++/UPerNet/DeepLabV3+ with multiple backbones.
- Training recipe: augmentations, threshold calibration, and leaderboard-oriented model selection.

## How The Project Works

At a high level, every pipeline follows the same shape:

1. Read raw challenge rasters and weak labels.
2. Reproject everything to a shared Sentinel-2 grid.
3. Decode + fuse weak labels into training targets, ignore masks, and weight maps.
4. Build feature tensors and normalization stats.
5. Train segmentation (or temporal multitask) models.
6. Run sliding-window inference on full tiles.
7. Export prediction rasters and convert them to submission-ready GeoJSON.

## Quick Start

### 1) Install

```bash
make install
```

### 2) Download dataset

```bash
make download_data_from_s3
```

Expected root after download:

```text
data/makeathon-challenge/
```

### 3) Run baseline pipeline

```bash
# preprocessing
./.venv/bin/python scripts/preprocess.py \
  --data-root data/makeathon-challenge \
  --output-dir output/preprocessing

# training
./.venv/bin/python scripts/train.py \
  model=resnet18_unet \
  epochs=40 \
  batch_size=4 \
  output_root=output/training_runs

# inference on validation split
./.venv/bin/python scripts/predict.py \
  checkpoint=output/training_runs/<run_name>/best.pt \
  split=val \
  output_dir=output/predictions/<run_name>
```

## Project Structure

```text
makeathon-challenge-2026-xylemx/
├── README.md
├── challenge.ipynb
├── osapiens-challenge-full-description.md
├── Makefile
├── requirements.txt
├── pyproject.toml
├── download_data.py
├── submission_utils.py
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── preprocessing_*.py
│   └── train_*.py
├── src/xylemx/
│   ├── config.py
│   ├── data/
│   ├── labels/
│   ├── preprocessing/
│   ├── models/
│   ├── training/
│   ├── temporal/
│   ├── single_2025/
│   └── visualization/
├── docs/
├── tests/
├── jobs/
└── output/  # generated experiment artifacts
```

What each main area does:

- `src/xylemx/`: core library (data loading, feature engineering, model building, training logic).
- `scripts/`: CLI entrypoints for reproducible runs.
- `docs/`: architecture, pipeline, and technical notes.
- `jobs/`: SLURM/local automation scripts.
- `tests/`: coverage for config parsing, label fusion, model registry, and utilities.
- `output/`: generated preprocessing caches, checkpoints, and predictions.

## Pipeline Tracks

- **Baseline snapshot**: standard segmentation from engineered multimodal snapshots.
- **Leaderboard**: stronger defaults and model/threshold selection.
- **Temporal**: segmentation + event-time prediction from monthly sequences.
- **Temporal HQ**: higher-quality temporal defaults for stronger generalization.
- **Single 2025**: simplified single-date summer-2025 variant.

## Key Files To Read First

1. [osapiens-challenge-full-description.md](./osapiens-challenge-full-description.md)
2. [challenge.ipynb](./challenge.ipynb)
3. [docs/pipeline-overview.md](./docs/pipeline-overview.md)
4. [docs/project-technical-guide.md](./docs/project-technical-guide.md)

## Documentation Index

- [Docs Home](./docs/README.md)
- [Baseline Pipeline](./docs/baseline-pipeline.md)
- [Pipeline Overview](./docs/pipeline-overview.md)
- [Project Technical Guide](./docs/project-technical-guide.md)
- [Weak Label Fusion](./docs/weak-label-fusion.md)
- [SLURM Jobs](./jobs/README.md)

## Notes

- Training uses weak supervision, so label fusion quality is a first-class part of model quality.
- Most scripts accept `key=value` overrides for fast experiment iteration.
- Generated outputs under `output/` are large and should be treated as run artifacts, not hand-edited source files.
