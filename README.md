# osapiens Challenge Makeathon 2026

## Detecting Deforestation from Space

![Deforestation event example](content/deforestation.png)

This repository contains the materials for the osapiens Makeathon 2026 challenge on deforestation detection from multimodal satellite data. The goal is to build a system that identifies deforestation events after 2020 using noisy, heterogeneous geospatial inputs and weak supervision signals.

## Start Here

If you are new to the repository, use these files in this order:

1. [osapiens-challenge-full-description.md](./osapiens-challenge-full-description.md) for the written challenge brief and context.
2. [challenge.ipynb](./challenge.ipynb) for the full walkthrough of the dataset structure, label encodings, visualizations, and submission example.
3. [download_data.py](./download_data.py) for the dataset download entrypoint used by the project.

## Repository Guide

- [challenge.ipynb](./challenge.ipynb): Main challenge notebook with data layout, modality descriptions, label definitions, examples, and submission guidance.
- [osapiens-challenge-full-description.md](./osapiens-challenge-full-description.md): Full challenge description.
- [download_data.py](./download_data.py): Downloads the challenge data from S3 into `./data`.
- [submission_utils.py](./submission_utils.py): Utility for converting prediction rasters into submission-ready GeoJSON.
- [Makefile](./Makefile): Convenience targets for environment setup and data download.

## Setup

Create the virtual environment and install the dependencies:

```bash
make install
```

Download the dataset:

```bash
make download_data_from_s3
```

This uses [download_data.py](./download_data.py) and stores the files under:

```text
data/makeathon-challenge/
```

## Dataset Layout

After downloading, the notebook expects the data in the following structure:

```text
data/makeathon-challenge/
├── sentinel-1/
│   ├── train/{tile_id}__s1_rtc/{tile_id}__s1_rtc_{year}_{month}_{ascending|descending}.tif
│   └── test/...
├── sentinel-2/
│   ├── train/{tile_id}__s2_l2a/{tile_id}__s2_l2a_{year}_{month}.tif
│   └── test/...
├── aef-embeddings/
│   ├── train/{tile_id}_{year}.tiff
│   └── test/...
├── labels/train/
│   ├── gladl/
│   ├── glads2/
│   └── radd/
└── metadata/
    ├── train_tiles.geojson
    └── test_tiles.geojson
```

## Explore the Notebook for the Full Challenge Walkthrough

## Docs

- [Baseline Pipeline](./docs/baseline-pipeline.md)
- [Weak Label Fusion](./docs/weak-label-fusion.md)
- [SLURM Jobs](./jobs/README.md)

## Baseline Pipeline

The repository now includes a lightweight, terminal-first baseline under `src/xylemx/` plus:

```text
scripts/preprocess.py
scripts/train.py
scripts/predict.py
```

Typical workflow:

```bash
./.venv/bin/pip install -e . --no-build-isolation

./.venv/bin/python scripts/preprocess.py \
  --data-root data/makeathon-challenge \
  --output-dir output/preprocessing

./.venv/bin/python scripts/train.py \
  model=small_unet \
  batch_size=8 \
  patch_size=128 \
  epochs=5 \
  lr=1e-3 \
  output_dir=output/train_runs/debug

./.venv/bin/python scripts/predict.py \
  checkpoint=output/train_runs/debug/best.pt \
  split=val \
  output_dir=output/predictions/debug
```

Supported model names now include:

- `small_unet`
- encoder-decoder variants use suffixes `*_unet`, `*_fpn`, `*_unetpp`, `*_upernet`, `*_deeplabv3plus`
- CBAM attention variants use suffixes `*_unet_cbam`, `*_fpn_cbam`, `*_unetpp_cbam`, `*_upernet_cbam`, and `*_deeplabv3plus_cbam`
- backbones currently include:
  `resnet18`, `resnet34`, `resnet50`, `resnet101`
  `efficientnet_b0`
  `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large`
  `convnextv2_atto`, `convnextv2_femto`, `convnextv2_pico`, `convnextv2_nano`, `convnextv2_tiny`, `convnextv2_small`, `convnextv2_base`
  `coatnet0`, `coatnet1`, `coatnet2`, `coatnet3`
  `vgg11`, `vgg13`, `vgg16`, `vgg19`

Examples:

- `resnet50_unet`
- `resnet50_fpn`
- `resnet50_unetpp`
- `resnet34_unet_cbam`
- `resnet34_fpn_cbam`
- `resnet34_fpn`
- `convnext_tiny_fpn`
- `convnext_tiny_upernet`
- `convnext_tiny_deeplabv3plus`
- `convnext_tiny_deeplabv3plus_cbam`
- `convnextv2_tiny_unetpp`
- `convnextv2_tiny_upernet`
- `convnextv2_tiny_deeplabv3plus`
- `coatnet0_fpn`
- `coatnet0_upernet`
- `vgg16_unetpp`

## Notes And Assumptions

- The first baseline uses Sentinel-2 only and builds one early-year composite and one late-year composite per tile. It prefers `2020` and `2025`, and falls back to the earliest/latest available Sentinel-2 year when a tile is missing one of those endpoints.
- Weak labels are decoded and reprojected onto the Sentinel-2 grid before consensus targets are built.
- The current multimodal preprocessing default uses a `snapshot_pair` feature mode: one early snapshot, one late snapshot, and one `late - early` delta for Sentinel-2, Sentinel-1, and AEF PCA features.
- One training tile in the provided data (`18NWM_9_4`) is only 2 pixels wide in Sentinel-2. The dataset loader skips tiles smaller than the requested patch size instead of forcing invalid patches into training.
- In this offline sandbox, editable install needed `--no-build-isolation` because `pip install -e .` attempted to resolve build dependencies from the network.
