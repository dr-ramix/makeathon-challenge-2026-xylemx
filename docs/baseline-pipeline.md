# XylemX Baseline Pipeline

This document explains the baseline pipeline that was implemented in this repository for the osapiens Makeathon 2026 deforestation challenge.

It is written to be practical for a hackathon team:

- what the pipeline does
- how the data flows through it
- how labels are decoded
- what math is used
- what assumptions were made
- how to run it from the terminal
- what outputs to expect

## 1. Goal

The goal is a simple, correct, hackathon-friendly segmentation pipeline for deforestation detection.

The implemented pipeline covers:

1. preprocessing challenge data
2. decoding weak labels correctly
3. splitting train into train and validation at tile level
4. training a baseline segmentation model
5. running inference
6. saving binary prediction rasters for later submission

The code lives under `src/xylemx/` and is meant to be used with an editable install.

## 2. What Was Implemented

The following modules and scripts were added or completed:

```text
src/xylemx/
├── data/
│   ├── io.py
│   ├── dataset.py
│   ├── splits.py
│   └── tiling.py
├── labels/
│   ├── decode.py
│   └── consensus.py
├── preprocessing/
│   ├── features.py
│   ├── normalize.py
│   └── pipeline.py
├── models/
│   ├── baseline.py
│   └── losses.py
└── training/
    ├── evaluate.py
    ├── metrics.py
    └── train.py

scripts/
├── preprocess.py
├── train.py
└── predict.py
```

Also added:

- `configs/baseline.yaml`
- this docs page

## 3. Repository Reality Check

The pipeline was built against the actual repository and actual downloaded data, not an invented schema.

Important real-world observations from this repo:

- some train tiles are missing `GLAD-S2` files, so consensus must use all available sources per tile instead of requiring all three
- some Sentinel-2 monthly rasters are not perfectly aligned to the rest of the tile, so the pipeline reprojects them onto a tile reference grid
- one train tile, `18NWM_9_4`, has Sentinel-2 scenes that are only 2 pixels wide, so patch training skips it
- some environments block PyTorch multiprocessing shared memory, so training automatically falls back from `num_workers > 0` to `num_workers=0` if needed

These are not theoretical edge cases. They happened in this repo and the implementation handles them.

## 4. Data Layout

Expected challenge layout:

```text
data/makeathon-challenge/
├── sentinel-1/
├── sentinel-2/
├── aef-embeddings/
├── labels/train/
│   ├── gladl/
│   ├── glads2/
│   └── radd/
└── metadata/
```

The first baseline uses Sentinel-2 only for model input, but the preprocessing code still inspects the full challenge structure so it can be extended later.

## 5. Baseline Design

### 5.1 Input Modality

The first baseline uses Sentinel-2 features only.

Why:

- Sentinel-2 already gives strong spectral information for vegetation loss
- it keeps the first version lightweight
- it avoids overcomplicating early experiments with reprojection of multiple modalities

### 5.2 Temporal Representation

Instead of feeding the full monthly time series directly into the network, the baseline builds two summary composites:

- an early-year composite
- a late-year composite

Preferred years:

- early year: `2020`
- late year: `2025`

Fallback rule:

- if `2020` is missing for a tile, use the earliest available Sentinel-2 year
- if `2025` is missing for a tile, use the latest available Sentinel-2 year

This keeps the baseline simple while still giving the model a change signal.

### 5.3 Feature Set

For each tile the feature stack has 30 channels:

1. 12-band early-year Sentinel-2 median composite
2. 12-band late-year Sentinel-2 median composite
3. `NDVI` for early year
4. `NDVI` for late year
5. `NBR` for early year
6. `NBR` for late year
7. early-year observation count
8. late-year observation count

So:

```text
12 + 12 + 4 + 2 = 30 channels
```

More explicitly:

```text
24 spectral channels
+ 4 index channels
+ 2 count channels
= 30 total channels
```

## 6. Label Decoding

Weak labels are decoded in `src/xylemx/labels/decode.py`.

Each decoder returns structured pixel-wise outputs:

- `is_positive`
- `confidence_score`
- `event_date`
- `raw_class`
- `is_uncertain`
- `valid_mask`

### 6.1 RADD

Raw definition:

- `0` means no alert
- leading digit `2` means low confidence
- leading digit `3` means high confidence
- remaining digits are days since `2014-12-31`

Let the raw pixel value be `r`.

Then:

```text
class(r) = floor(r / 10000)
days(r)  = r mod 10000
```

The implementation uses:

```text
positive      if class(r) in {2, 3} and days(r) > 0
confidence    = 0.6 for class 2, 1.0 for class 3
event_date    = 2014-12-31 + days(r)
raw_class     = class(r)
```

Example:

```text
r = 31234
class = 3
days = 1234
positive = True
confidence = 1.0
event_date = 2014-12-31 + 1234 days
```

### 6.2 GLAD-L

Files are year-specific:

- `gladl_{tile}_alert21.tif`
- `gladl_{tile}_alertDate21.tif`
- ...

Rules:

- `alertYY = 0` means no loss
- `alertYY = 2` means probable loss
- `alertYY = 3` means confirmed loss
- `alertDateYY` is day-of-year in year `20YY`

The implementation uses:

```text
positive      if alert in {2, 3} and alertDate > 0
confidence    = 0.6 for class 2, 1.0 for class 3
event_date    = Jan 1 of that year + (day_of_year - 1)
raw_class     = alert
```

Example:

```text
year = 2024
alert = 3
alertDate = 150

positive = True
confidence = 1.0
event_date = 2024-05-29
```

because day 150 of leap year 2024 is `2024-05-29`.

All GLAD-L year rasters are decoded separately and then combined into one source-level label map by taking:

- pixel-wise positive OR
- pixel-wise maximum confidence
- earliest non-null event date

### 6.3 GLAD-S2

Rules:

- `0` = no loss
- `1` = recent observation only
- `2` = low confidence
- `3` = medium confidence
- `4` = high confidence
- `alertDate` is days since `2019-01-01`

The implementation uses:

```text
positive      if alert >= 2 and alertDate > 0
confidence    = 0.2, 0.45, 0.7, 1.0 for classes 1, 2, 3, 4
event_date    = 2019-01-01 + alertDate
uncertain     if alert == 1
raw_class     = alert
```

Example:

```text
alert = 4
alertDate = 365

positive = True
confidence = 1.0
event_date = 2020-01-01
```

## 7. Raster Alignment

Weak labels and some auxiliary rasters are not delivered on the same grid as Sentinel-2.

The preprocessing pipeline reprojects weak labels onto the reference Sentinel-2 grid using nearest-neighbor resampling.

For a source raster `L_src` and destination grid `G_s2`:

```text
L_dst = ReprojectNearest(L_src -> G_s2)
```

This is important because segmentation targets must align pixel-to-pixel with model inputs.

The tile reference grid is chosen from the largest Sentinel-2 raster available for that tile, not just the first sorted file. This is necessary because some tiles contain clipped monthly scenes.

## 8. Consensus Target Construction

Consensus is implemented in `src/xylemx/labels/consensus.py`.

Available modes:

- `agreement`
- `soft`

Default baseline:

- `agreement`
- `min_positive_sources = 2`

### 8.1 Per-source Quantities

For each pixel `p` and source `s`:

- `z_s(p) in {0,1}` indicates whether that source says deforestation
- `c_s(p) in [0,1]` is that source confidence
- `a_s(p) in {0,1}` indicates whether that source is available

Then:

```text
votes(p)      = sum_s z_s(p)
avail(p)      = sum_s a_s(p)
soft(p)       = sum_s c_s(p) / max(avail(p), 1)
```

### 8.2 Agreement Mode

Agreement mode creates a hard binary target:

```text
y(p) = 1 if votes(p) >= N
     = 0 otherwise
```

where `N = min_positive_sources`.

The ignore mask is:

```text
ignore(p) = 1 if
    (0 < votes(p) < N)
    or uncertain(p)
    or avail(p) = 0
```

Interpretation:

- if too few sources agree, the pixel is not trusted enough for hard supervision
- if a source explicitly marks the pixel as uncertain, we can ignore it
- if no source is available, we ignore it

Example with `N = 2`:

```text
RADD    = positive
GLAD-L  = positive
GLAD-S2 = missing

votes = 2
target = 1
ignore = 0
```

Another example:

```text
RADD    = positive
GLAD-L  = negative
GLAD-S2 = missing

votes = 1
target = 0
ignore = 1
```

This prevents weak single-source positives from being treated as confident negatives.

### 8.3 Soft Mode

Soft mode uses the source confidences directly:

```text
y_soft(p) = soft(p)
```

This was implemented for flexibility, but the baseline defaults to hard agreement mode.

## 9. Feature Math

### 9.1 Median Composite

For a band `b` and all monthly observations in a selected year:

```text
X_b(p) = median over valid monthly observations at pixel p
```

The code uses `nanmedian`, so invalid pixels are excluded.

### 9.2 NDVI

Using:

- `NIR = B08`
- `RED = B04`

```text
NDVI = (NIR - RED) / (NIR + RED + eps)
```

where `eps = 1e-6`.

Example:

```text
NIR = 0.62
RED = 0.21
NDVI = (0.62 - 0.21) / (0.62 + 0.21) ≈ 0.494
```

### 9.3 NBR

Using:

- `NIR = B08`
- `SWIR2 = B12`

```text
NBR = (NIR - SWIR2) / (NIR + SWIR2 + eps)
```

Example:

```text
NIR = 0.58
SWIR2 = 0.31
NBR = (0.58 - 0.31) / (0.58 + 0.31) ≈ 0.303
```

NBR is useful because burned or disturbed vegetation often shows stronger change in the SWIR bands.

## 10. Normalization

Normalization stats are computed on the train split only.

For each feature channel `k`:

```text
mu_k    = mean of all valid train pixels in channel k
sigma_k = std  of all valid train pixels in channel k
```

Then for each pixel value:

```text
x'_k = (x_k - mu_k) / max(sigma_k, 1e-6)
```

This avoids train/validation leakage and keeps the input scale stable.

## 11. Train/Validation Split

Splitting is tile-level, not pixel-level.

Why that matters:

- random pixel splits would leak spatial context
- neighboring pixels from the same tile are highly correlated
- tile-level splitting is a much more honest generalization test

The split implementation:

- is deterministic given a seed
- supports coarse stratification using positive-pixel fraction
- writes:
  - `output/preprocessing/train_tiles.json`
  - `output/preprocessing/val_tiles.json`

In the verified local preprocessing run, the split was:

```text
train tiles: 13
val tiles:   3
```

## 12. Patch Dataset

The training dataset is patch-based.

For each tile:

1. load cached feature tensor
2. load target mask
3. load ignore mask
4. generate patch windows of size `patch_size`
5. skip useless patches with too little valid data

Patch metadata includes:

- tile id
- top-left pixel
- valid fraction
- positive fraction

### 12.1 Oversampling

If `oversample_positives=true`, patches with positive fraction above a small threshold are duplicated in the sampling list.

This is a simple hackathon-friendly way to reduce class imbalance without changing the loss definition.

## 13. Augmentations

The dataset applies safe spatial augmentations during training:

- random horizontal flip
- random vertical flip
- random rotation by `0`, `90`, `180`, or `270` degrees
- optional Gaussian noise

These are geospatially safe because they do not distort the raster scale or band semantics.

## 14. Model

The baseline model is in `src/xylemx/models/baseline.py`.

### 14.1 Default Model

`small_unet`

It is a lightweight UNet-style segmentation model with:

- convolution blocks
- max-pooling encoder
- transposed-convolution decoder
- skip connections
- single-channel output logits

### 14.2 Output

The model predicts one logit per pixel:

```text
logit(p) in R
```

The probability is:

```text
prob(p) = sigmoid(logit(p))
```

and the binary prediction is:

```text
pred(p) = 1 if prob(p) >= threshold else 0
```

## 15. Losses

Losses are implemented in `src/xylemx/models/losses.py`.

Supported:

- `bce`
- `dice`
- `bce_dice`

Default:

- `bce_dice`

### 15.1 BCE With Logits

For one pixel:

```text
L_BCE = - y log(sigmoid(z)) - (1-y) log(1 - sigmoid(z))
```

where:

- `z` is the logit
- `y` is the target

Ignored pixels are removed from the average.

### 15.2 Dice Loss

Let `p_i` be predicted probabilities and `y_i` be targets over valid pixels.

Dice score:

```text
Dice = (2 sum_i p_i y_i + smooth) / (sum_i p_i + sum_i y_i + smooth)
```

Dice loss:

```text
L_Dice = 1 - Dice
```

### 15.3 Combined Loss

The combined loss is:

```text
L = L_BCE + L_Dice
```

This is a common baseline for binary segmentation because:

- BCE gives stable pixel-wise gradients
- Dice helps with class imbalance and overlap quality

## 16. Evaluation Metrics

Metrics are implemented in `src/xylemx/training/metrics.py`.

All metrics support ignore masks.

Important implementation detail:

- metrics are now computed from one global confusion matrix accumulated over all non-ignored pixels in the whole split
- they are not computed by averaging per-batch metric values

This is more accurate for the task because validation patches can contain very different numbers of usable pixels. Global-count aggregation weights each pixel correctly.

Let:

- `TP` = true positives
- `FP` = false positives
- `FN` = false negatives
- `TN` = true negatives

Then:

### 16.1 Precision

```text
Precision = TP / (TP + FP)
```

### 16.2 Recall

```text
Recall = TP / (TP + FN)
```

### 16.3 IoU

```text
IoU = TP / (TP + FP + FN)
```

### 16.4 Dice / F1

```text
Dice = 2 TP / (2 TP + FP + FN)
```

For binary segmentation in this pipeline:

```text
F1 = Dice
```

### 16.5 Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Example:

```text
TP = 80
FP = 20
FN = 40
TN = 860

Precision = 80 / 100 = 0.80
Recall    = 80 / 120 ≈ 0.667
IoU       = 80 / 140 ≈ 0.571
Dice      = 160 / 220 ≈ 0.727
F1        = 160 / 220 ≈ 0.727
Accuracy  = 940 / 1000 = 0.94
```

## 17. Inference and Stitching

Inference is patch-based.

For each tile:

1. load cached features
2. normalize them with saved train stats
3. generate patch windows
4. run model on each patch
5. average overlapping patch probabilities
6. threshold to a binary raster
7. save a single-band GeoTIFF

If a pixel is covered by multiple patches:

```text
P_final(p) = sum_j P_j(p) / count(p)
```

Then:

```text
Y_final(p) = 1 if P_final(p) >= threshold else 0
```

This avoids visible seams between patches.

## 18. Preprocessing Outputs

The preprocessing script writes to `output/preprocessing/`.

Main outputs:

- `train_tiles.json`
- `val_tiles.json`
- `normalization_stats.json`
- `feature_spec.json`
- `tile_metadata.json`
- `summary.json`
- `features/train/{tile}.npy`
- `features/test/{tile}.npy`
- `valid_masks/{split}/{tile}.npy`
- `targets/{tile}.npy`
- `ignore_masks/{tile}.npy`

## 19. Training Outputs

The training script writes to `output/train_runs/<run_name>/`.

Main outputs:

- `config.json`
- `metrics.json`
- `best.pt`
- `last.pt`

The checkpoint contains:

- model weights
- resolved config
- epoch number
- input channel count
- best validation metric at save time

## 20. Prediction Outputs

The prediction script writes to `output/predictions/<run_name>/`.

Main outputs:

- one binary GeoTIFF per predicted tile
- `summary.json`

These rasters are compatible with the existing submission utility:

- `submission_utils.py`

The intended next step is:

```python
from submission_utils import raster_to_geojson

geojson = raster_to_geojson("output/predictions/debug/47QQV_2_4.tif")
```

## 21. Commands

### 21.1 Install

Standard editable install:

```bash
pip install -e .
```

In the offline sandbox where this baseline was verified, this needed:

```bash
./.venv/bin/pip install -e . --no-build-isolation
```

because build isolation attempted to fetch `setuptools` from the network.

### 21.2 Preprocess

```bash
./.venv/bin/python scripts/preprocess.py \
  --data-root data/makeathon-challenge \
  --output-dir output/preprocessing \
  --val-ratio 0.2 \
  --split-seed 42 \
  --consensus-mode agreement \
  --min-positive-sources 2
```

### 21.3 Train

```bash
./.venv/bin/python scripts/train.py \
  model=small_unet \
  batch_size=8 \
  patch_size=128 \
  epochs=5 \
  lr=1e-3 \
  output_dir=output/train_runs/debug
```

The script prints the resolved config at startup and saves the resolved config again into `config.json`.

### 21.4 Predict

```bash
./.venv/bin/python scripts/predict.py \
  checkpoint=output/train_runs/debug/best.pt \
  split=val \
  output_dir=output/predictions/debug
```

To predict on the challenge test split:

```bash
./.venv/bin/python scripts/predict.py \
  checkpoint=output/train_runs/debug/best.pt \
  split=test \
  output_dir=output/predictions/test_from_debug
```

## 21.5 SLURM Jobs

SLURM templates are provided in `jobs/`:

- `jobs/preprocess.slurm`
- `jobs/train_baseline.slurm`
- `jobs/train_all_models.slurm`
- `jobs/submit_all_models.sh`
- `jobs/run_all_models_local.sh`
- `jobs/predict_val.slurm`
- `jobs/predict_all_models.slurm`
- `jobs/submit_all_and_predict.sh`

Examples:

```bash
sbatch jobs/preprocess.slurm
sbatch jobs/train_baseline.slurm
sbatch jobs/train_all_models.slurm
bash jobs/submit_all_models.sh
bash jobs/run_all_models_local.sh
bash jobs/submit_all_and_predict.sh
sbatch jobs/predict_val.slurm
sbatch jobs/predict_all_models.slurm
```

Override runtime parameters at submit time:

```bash
sbatch --export=ALL,RUN_NAME=baseline_gpu01,EPOCHS=20,BATCH_SIZE=8 jobs/train_baseline.slurm
```

Run every registered model sequentially for 20 epochs each:

```bash
sbatch jobs/train_all_models.slurm
```

Submit one dependent training job per model:

```bash
bash jobs/submit_all_models.sh
```

Submit the full training dependency chain and automatically queue all-model prediction after the last training job:

```bash
bash jobs/submit_all_and_predict.sh
```

Run the whole multi-model training sequence locally on the current machine without SLURM:

```bash
bash jobs/run_all_models_local.sh
```

Generate predictions for all trained model checkpoints:

```bash
sbatch jobs/predict_all_models.slurm
```

## 22. Example Run Results

The pipeline was verified locally in this repository.

### 22.1 Verified Preprocessing Run

Command:

```bash
./.venv/bin/python scripts/preprocess.py \
  --data-root data/makeathon-challenge \
  --output-dir output/preprocessing
```

Result:

- completed successfully
- cached train and test features
- wrote train and validation splits
- wrote normalization stats

### 22.2 Verified Training Run

Command:

```bash
./.venv/bin/python scripts/train.py \
  model=small_unet \
  batch_size=8 \
  patch_size=128 \
  epochs=5 \
  lr=1e-3 \
  output_dir=output/train_runs/debug
```

Observed best validation Dice in the verified run:

```text
0.3897
```

Stored in:

- `output/train_runs/debug/metrics.json`
- `output/train_runs/debug/best.pt`

### 22.3 Verified Prediction Run

Command:

```bash
./.venv/bin/python scripts/predict.py \
  checkpoint=output/train_runs/debug/best.pt \
  split=val \
  output_dir=output/predictions/debug
```

This produced validation prediction rasters for:

- `47QQV_2_4`
- `48PWV_7_8`
- `48PXC_7_7`

## 23. Important Assumptions

These assumptions are intentional and should be understood by the team.

### 23.1 This Is a Baseline, Not the Final Model

The current baseline emphasizes:

- correctness
- readability
- fast iteration

It does not yet use:

- Sentinel-1 model inputs
- AEF embedding inputs
- temporal transformers or sequence models
- sophisticated uncertainty calibration
- advanced spatial postprocessing

### 23.2 Consensus Is Conservative

The default hard target requires agreement from at least 2 positive sources.

That reduces noisy supervision, but it may also miss some real positives supported by only one source.

### 23.3 Endpoint Composites Trade Detail For Simplicity

Using early and late yearly composites is much simpler than using full monthly time series, but it discards within-year timing information.

That is a good trade for a first baseline, not necessarily the final competition strategy.

## 24. Good Next Steps

Once this baseline is stable, the next upgrades that make sense are:

1. add Sentinel-1 features on the same reference grid
2. add AEF embeddings after reprojection to the Sentinel-2 grid
3. try soft-label training using `consensus_mode=soft`
4. tune threshold on validation instead of fixing `0.5`
5. add a stronger model once data handling is fully trusted

## 25. File Guide

If you want to inspect the implementation directly:

- data scanning and raster helpers: `src/xylemx/data/io.py`
- patch dataset: `src/xylemx/data/dataset.py`
- tile splitting: `src/xylemx/data/splits.py`
- label decoding: `src/xylemx/labels/decode.py`
- consensus logic: `src/xylemx/labels/consensus.py`
- feature engineering: `src/xylemx/preprocessing/features.py`
- preprocessing orchestration: `src/xylemx/preprocessing/pipeline.py`
- UNet baseline: `src/xylemx/models/baseline.py`
- losses: `src/xylemx/models/losses.py`
- metrics: `src/xylemx/training/metrics.py`
- training loop: `src/xylemx/training/train.py`
- prediction script: `scripts/predict.py`

## 26. Bottom Line

The implemented pipeline is intentionally modest but correct:

- it decodes the weak labels accurately
- it builds consensus targets carefully
- it avoids tile leakage by splitting at tile level
- it trains a real segmentation baseline
- it writes binary GeoTIFF prediction rasters for submission conversion later

That is a strong place to start for a hackathon team because it gives you a trustworthy baseline you can extend instead of spending time debugging the fundamentals.
