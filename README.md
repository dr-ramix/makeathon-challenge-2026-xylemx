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

## Preprocessing Details

All training tracks start from preprocessing and produce reusable cached arrays in `output/preprocessing*`.

### Baseline Snapshot Preprocessing (`scripts/preprocess.py`)

- Uses `src/xylemx/preprocessing/pipeline.py`.
- Reprojects modalities and weak labels to the Sentinel-2 reference grid.
- Builds multimodal features from staged snapshots.
- Supports `temporal_feature_mode=snapshot_pair` with `early`, `late`, and `late-early` deltas.
- Supports `temporal_feature_mode=snapshot_quad` with `early`, `middle1`, `middle2`, `late`, and deltas.
- Writes train/test feature caches and train supervision arrays.

Main artifacts:

- `features/{split}/{tile}.npy`
- `valid_masks/{split}/{tile}.npy`
- `targets/{tile}.npy`
- `ignore_masks/{tile}.npy`
- `weight_maps/{tile}.npy`
- `vote_counts/{tile}.npy`
- `normalization_stats.json`
- `train_tiles.json` and `val_tiles.json`

### Leaderboard Preprocessing (`scripts/preprocessing_leaderboard.py`)

Uses the same core engine with tuned defaults, including:

- `temporal_feature_mode=snapshot_quad`
- `use_s1_features=true`
- `use_aef_features=true`
- tighter clipping settings and stronger label thresholds

### Temporal Preprocessing (`scripts/preprocessing_temporal.py`)

- Builds monthly time-series tensors from `time_start` to `time_end`.
- Creates binary mask targets plus time-bin targets.
- Supports representations such as `early_middle_late_deltas`.
- Exports temporal specs and time-bin metadata for multitask training.

### Single-Year Variant (`scripts/preprocessing_single_2025.py`)

- Creates a simplified summer snapshot (default target year: `2025`).
- Useful as a controlled ablation against temporal/snapshot-rich setups.

## Weak-Label Fusion Details

Fusion is implemented in `src/xylemx/labels/consensus.py` and is central to training quality.

Label sources:

- `radd`
- `gladl` (merged across yearly files)
- `glads2`

Source-to-binary conversion:

- `radd_positive_mode=permissive`: any positive code becomes alert.
- `radd_positive_mode=conservative`: only high-confidence RADD classes.
- `gladl_threshold` and `glads2_threshold` control source sensitivity.

Fusion methods:

- `consensus_2of3`: positive if at least two sources vote positive.
- `union`: positive if any source votes positive.
- `unanimous`: positive if all available sources vote positive.
- `soft_vote`: positive if `vote_count / available_sources >= soft_vote_threshold`.

Training supervision outputs per pixel:

- `target`: binary target mask.
- `soft_target`: source-agreement ratio.
- `ignore_mask`: ignored pixels (outside extent and optional uncertain pixels).
- `weight_map`: vote-based weight (`vote_weight_0..3`).
- `vote_count` and `available_sources`: agreement diagnostics.

Default vote weights are:

- `vote_weight_0=1.0`
- `vote_weight_1=0.3`
- `vote_weight_2=0.8`
- `vote_weight_3=1.0`

## Model Details

### Segmentation Models (`src/xylemx/models/baseline.py`)

Native models:

- `small_unet`
- `coatnext_tiny_unet`

Composable timm models use:

- Backbone alias + decoder suffix pattern.
- Decoder suffix `_unet`
- Decoder suffix `_fpn`
- Decoder suffix `_unetpp`
- Decoder suffix `_upernet`
- Decoder suffix `_deeplabv3plus`
- CBAM variants `_unet_cbam`, `_fpn_cbam`, `_unetpp_cbam`, `_upernet_cbam`, `_deeplabv3plus_cbam`

Available backbone aliases include:

- `resnet18`, `resnet34`, `resnet50`, `resnet101`
- `efficientnet_b0`
- `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`
- `convnextv2_atto`, `convnextv2_femto`, `convnextv2_pico`, `convnextv2_nano`, `convnextv2_tiny`, `convnextv2_small`, `convnextv2_base`
- `coatnet0`, `coatnet1`, `coatnet2`, `coatnet3`
- `vgg11`, `vgg13`, `vgg16`, `vgg19`

Example names:

- `resnet50_fpn`
- `resnet34_unetpp`
- `convnext_tiny_upernet`
- `convnextv2_tiny_deeplabv3plus_cbam`

### Temporal Models (`src/xylemx/temporal/training.py`)

Supported temporal model names:

- `film_temporal_unet`
- `film_temporal_unet_plus`

Both are dual-head models that output:

- segmentation mask logits
- event-time logits (time-bin classification)

### Leaderboard Model Search

`scripts/train_leaderboard.py` supports candidate search and optional threshold calibration.
Default search candidates are:

- `resnet34_unetpp`
- `resnet50_fpn`
- `convnext_tiny_fpn`
- `convnextv2_tiny_unetpp`

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
