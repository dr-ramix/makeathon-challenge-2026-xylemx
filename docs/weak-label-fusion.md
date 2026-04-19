# Weak-Label Fusion And Target Geometry

This document explains how the training target is built from the three weak-label sources in this repository:

- `RADD`
- `GLAD-L`
- `GLAD-S2`

It also explains:

- the math of the fusion rule
- the CRS and reprojection steps
- how raster sizes and channel counts change
- why the input tensor and output mask line up pixel-for-pixel

The implementation described here matches the current code in:

- [src/xylemx/labels/consensus.py](../src/xylemx/labels/consensus.py)
- [src/xylemx/preprocessing/pipeline.py](../src/xylemx/preprocessing/pipeline.py)
- [src/xylemx/data/io.py](../src/xylemx/data/io.py)
- [src/xylemx/preprocessing/features.py](../src/xylemx/preprocessing/features.py)

## 1. Terminology

Strictly speaking, the repository does not create perfect ground-truth labels. It creates a fused training target from weak supervision.

So when we say "true label" below, what we really mean is:

- the final fused target mask used for training

For one tile, the fused supervision contains:

- `target`: binary training target, shape `[H, W]`
- `weight_map`: per-pixel training weight, shape `[H, W]`
- `ignore_mask`: pixels to exclude from loss, shape `[H, W]`
- `vote_count`: how many weak sources voted positive, shape `[H, W]`
- `valid_extent`: whether at least one source covers that pixel, shape `[H, W]`

## 1.1 Mathematical Notation

For one tile, let the Sentinel-2 master grid be the discrete pixel domain

```text
Omega = {0, ..., H-1} x {0, ..., W-1}
```

where each pixel is indexed by:

```text
x = (i, j) in Omega
```

We write:

- `X : Omega -> R^C` for the input feature tensor
- `y : Omega -> {0,1}` for the fused hard target
- `w : Omega -> R_{>=0}` for the pixel weight map
- `m : Omega -> {0,1}` for the ignore mask

Equivalently, in array form:

```text
X in R^(C x H x W)
y in {0,1}^(H x W)
w in R^(H x W)
m in {0,1}^(H x W)
```

For the three weak sources, define:

```text
S = {R, GL, GS2}
```

where:

- `R` is RADD
- `GL` is GLAD-L
- `GS2` is GLAD-S2

For each source `s in S`, after reprojection onto the Sentinel-2 grid, we have:

```text
L_s : Omega -> {0,1}
V_s : Omega -> {0,1}
```

where:

- `L_s(x)` is the binary positive label from source `s`
- `V_s(x)` is the valid-coverage indicator from source `s`

## 2. Master Grid And CRS Alignment

### 2.1 Sentinel-2 Is The Master Grid

Every tile uses one Sentinel-2 raster as the master spatial reference.

That master raster provides:

- `CRS`
- `transform`
- `width`
- `height`

In code, this is loaded from:

- [src/xylemx/data/io.py](../src/xylemx/data/io.py)

using:

```python
dst_profile = get_raster_profile(record.reference_s2_path)
```

This `dst_profile` becomes the common grid for:

- weak labels
- Sentinel-1
- AEF embeddings
- final feature tensor
- final training target

More explicitly, the Sentinel-2 master grid defines the tuple:

```text
G_s2 = (CRS_s2, T_s2, H, W)
```

where:

- `CRS_s2` is the Sentinel-2 coordinate reference system
- `T_s2` is the affine transform from pixel coordinates to map coordinates
- `H` is the raster height
- `W` is the raster width

Every preprocessed artifact for the tile is expressed on this same grid `G_s2`.

### 2.2 Reprojection Rule

Every label raster is read and reprojected to the Sentinel-2 grid with nearest-neighbor interpolation:

```python
array, valid_mask = read_reprojected_raster(
    path,
    dst_profile,
    resampling=Resampling.nearest,
    out_dtype=np.float32,
)
```

Nearest neighbor is important for labels because the label values are categorical and should not be blended.

More formally, if a source label raster lives on its own native grid `Omega_s`, then preprocessing applies a reprojection operator:

```text
Pi_s2<-s : Omega_s -> Omega
```

to produce a raster on the Sentinel-2 grid. In practice this is implemented by raster resampling with:

- source CRS = label CRS
- destination CRS = Sentinel-2 CRS
- source transform = label transform
- destination transform = Sentinel-2 transform
- interpolation = nearest neighbor

So the label-processing order is:

```text
raw raster on Omega_s
   -> reproject to Omega
   -> binarize to L_s
   -> fuse across sources
```

At the geometry level, the reprojection step can be thought of as:

```text
(CRS_s, T_s, H_s, W_s) -> (CRS_s2, T_s2, H, W)
```

The resulting raster has:

- the Sentinel-2 CRS
- the Sentinel-2 affine transform
- the Sentinel-2 width
- the Sentinel-2 height

So even if:

- RADD starts on grid `G_R`
- GLAD-L starts on grid `G_GL`
- GLAD-S2 starts on grid `G_GS2`

after reprojection they all live on the same tile grid:

```text
Pi_s2<-R(G_R)     = G_s2
Pi_s2<-GL(G_GL)   = G_s2
Pi_s2<-GS2(G_GS2) = G_s2
```

### 2.2.1 Pixel-Center Interpretation

Let `x = (i, j)` be a pixel index on the Sentinel-2 grid. Its map-space location is given by:

```text
u(x) = T_s2(i, j)
```

When a source raster is reprojected, the value assigned to output pixel `x` is the value of the source raster evaluated around the same map-space location `u(x)`.

For labels, nearest-neighbor interpolation means:

```text
L_s(x) = source label value at the nearest source pixel to u(x)
```

This is why categorical label codes are preserved rather than blurred.

### 2.2.2 Why Nearest Neighbor Matters For Fusion

Suppose a label pixel is binary:

```text
0 = no alert
1 = alert
```

If bilinear interpolation were used, neighboring pixels could create fractional values like:

```text
0.25, 0.5, 0.75
```

which would no longer be faithful categorical labels. Nearest-neighbor reprojection avoids that issue:

```text
{0,1} -> {0,1}
```

at the reprojection stage, before thresholding and fusion.

### 2.3 Why Input And Output Match Pixel-For-Pixel

After preprocessing, the following all share the same spatial shape:

- feature tensor: `[C, H, W]`
- fused target: `[H, W]`
- ignore mask: `[H, W]`
- weight map: `[H, W]`

So if pixel `(y, x)` in the input features refers to one location on the Sentinel-2 grid, then pixel `(y, x)` in the target refers to that same location.

That is the reason the model can be trained as dense segmentation:

- input: one tensor on the S2 grid
- output: one pixel-wise mask on the same S2 grid

Equivalently, preprocessing guarantees a common domain:

```text
dom(X) = dom(y) = dom(w) = dom(m) = Omega
```

This common-domain property is the most important geometric invariant in the whole pipeline.

## 3. Raw Weak Labels Before Fusion

## 3.1 RADD

Raw file:

- one single-band raster per train tile

After reprojection:

- shape: `[H, W]`

Binary conversion rule in the default permissive mode:

```text
R(x) = 1 if raw_radd(x) > 0, else 0
```

In code:

```python
radd_positive_mask(raw, mode="permissive")
```

There is also a conservative mode:

```text
R(x) = 1 if floor(raw_radd(x) / 10000) >= 3
```

### 3.1.1 RADD Code Structure

The raw RADD value can be decomposed as:

```text
raw_radd(x) = 10000 * c(x) + d(x)
```

where:

- `c(x)` is the leading code family
- `d(x)` is the remaining day-count payload

Equivalently:

```text
c(x) = floor(raw_radd(x) / 10000)
d(x) = raw_radd(x) mod 10000
```

In the challenge data, the important interpretation is:

- `c(x) = 0`: no alert
- `c(x) = 2`: lower-confidence alert family
- `c(x) = 3`: higher-confidence alert family

This is why the repository exposes two RADD binarization modes.

### 3.1.2 Permissive Mode

Permissive mode treats any non-zero RADD code as a positive weak signal:

```text
R_perm(x) = 1[raw_radd(x) > 0]
```

This means both lower-confidence and higher-confidence alerts are accepted:

```text
R_perm(x) = 1 if c(x) in {2, 3}
```

Practical interpretation:

- higher recall
- more positive weak labels
- more tolerance for noisy or uncertain alerts
- useful when you do not want to miss candidate deforestation pixels

### 3.1.3 Conservative Mode

Conservative mode only accepts the higher-confidence RADD family:

```text
R_cons(x) = 1[c(x) >= 3]
```

In the current implementation this is:

```text
R_cons(x) = 1[floor(raw_radd(x) / 10000) >= 3]
```

Practical interpretation:

- lower recall
- higher precision
- fewer positive weak labels
- useful when you want RADD to contribute only strong evidence

### 3.1.4 Side-By-Side Example

Suppose a raw RADD pixel has value:

```text
raw_radd(x) = 21234
```

Then:

```text
c(x) = floor(21234 / 10000) = 2
d(x) = 21234 mod 10000 = 1234
```

So:

```text
R_perm(x) = 1
R_cons(x) = 0
```

Now suppose:

```text
raw_radd(x) = 31234
```

Then:

```text
c(x) = 3
d(x) = 1234
```

So:

```text
R_perm(x) = 1
R_cons(x) = 1
```

And if:

```text
raw_radd(x) = 0
```

then:

```text
R_perm(x) = 0
R_cons(x) = 0
```

### 3.1.5 How The RADD Mode Changes Fusion

The RADD mode changes only the binary source mask `R(x)`.

Everything after that stays the same:

- reprojection still maps RADD to the Sentinel-2 grid
- vote counting still uses `votes(x) = R(x) + G_L(x) + G_S2(x)`
- the final fused target still depends on the selected fusion rule

So changing RADD mode modifies the fused target indirectly by changing the vote count.

For example, suppose:

```text
G_L(x)  = 1
G_S2(x) = 0
raw_radd(x) = 21234
```

Then:

```text
R_perm(x) = 1
R_cons(x) = 0
```

Under `consensus_2of3`:

```text
votes_perm(x) = 1 + 1 + 0 = 2
votes_cons(x) = 0 + 1 + 0 = 1
```

So the final hard target changes:

```text
y_perm(x) = 1[2 >= 2] = 1
y_cons(x) = 1[1 >= 2] = 0
```

That example is the core difference:

- permissive mode allows weaker RADD evidence to help push a pixel into consensus
- conservative mode requires stronger RADD evidence before it can affect consensus

### 3.1.6 When To Use Which Mode

Use `permissive` when:

- you want broader positive coverage
- you are comfortable letting the consensus rule and weight map absorb some noise
- you prefer recall over precision for the RADD source

Use `conservative` when:

- you want RADD to act as a stronger but rarer signal
- you suspect lower-confidence RADD alerts are too noisy for your experiment
- you prefer precision over recall for the RADD source

In command-line terms:

```bash
python scripts/preprocess.py \
  --output-dir output/preprocessing/snapshot_pair \
  temporal_feature_mode=snapshot_pair \
  radd_positive_mode=permissive
```

or:

```bash
python scripts/preprocess.py \
  --output-dir output/preprocessing/snapshot_pair \
  temporal_feature_mode=snapshot_pair \
  radd_positive_mode=conservative
```

## 3.2 GLAD-L

Raw files:

- one single-band alert raster per year
- years `2021, 2022, 2023, 2024, 2025`

Each yearly alert raster is first reprojected to the S2 grid:

- each reprojected yearly alert has shape `[H, W]`

Then each year is binarized:

```text
G_L^y(x) = 1 if alert_y(x) >= threshold, else 0
```

Default threshold:

```text
threshold = 2
```

Then the yearly masks are merged with logical OR:

```text
G_L(x) = G_L^2021(x) OR G_L^2022(x) OR G_L^2023(x) OR G_L^2024(x) OR G_L^2025(x)
```

So GLAD-L becomes one binary raster:

- shape: `[H, W]`

## 3.3 GLAD-S2

Raw file:

- one single-band alert raster per train tile

After reprojection:

- shape: `[H, W]`

Binary conversion rule:

```text
G_S2(x) = 1 if alert(x) >= threshold, else 0
```

Default threshold:

```text
threshold = 1
```

So GLAD-S2 becomes one binary raster:

- shape: `[H, W]`

## 4. Per-Source Valid Extents

Each source also has a valid-coverage mask after reprojection:

- `V_R(x)` for RADD
- `V_GL(x)` for GLAD-L
- `V_GS2(x)` for GLAD-S2

These tell us whether the source actually covers pixel `x`.

This matters because not every source necessarily covers every pixel for every tile.

The pipeline keeps those valid extents and uses them during fusion.

It is useful to think of `L_s` and `V_s` as separate objects:

```text
L_s(x) answers: "does this source vote positive here?"
V_s(x) answers: "is this source even defined here?"
```

That distinction is why the fusion logic can avoid treating unlabeled space as a clean negative.

## 5. Fusion Math

Let the three binary source masks at a pixel `x` be:

```text
R(x)    in {0, 1}
G_L(x)  in {0, 1}
G_S2(x) in {0, 1}
```

Equivalently, using generic source notation:

```text
L_s(x) in {0,1} for each s in S
```

The per-pixel source vector is:

```text
ell(x) = [R(x), G_L(x), G_S2(x)]^T in {0,1}^3
```

### 5.1 Vote Count

The positive vote count is:

```text
votes(x) = R(x) + G_L(x) + G_S2(x)
```

or, equivalently,

```text
votes(x) = sum_{s in S} L_s(x)
```

In stacked-tensor form, if:

```text
L in {0,1}^{3 x H x W}
```

then:

```text
votes = sum over axis 0 of L
```

so `votes` is a raster in:

```text
votes in {0,1,2,3}^{H x W}
```

So:

- `votes(x) = 0` means nobody says deforestation
- `votes(x) = 1` means only one source says deforestation
- `votes(x) = 2` means two sources agree
- `votes(x) = 3` means all three agree

### 5.2 Available Source Count

The number of sources that actually cover pixel `x` is:

```text
avail(x) = V_R(x) + V_GL(x) + V_GS2(x)
```

or, in generic notation,

```text
avail(x) = sum_{s in S} V_s(x)
```

In stacked form, if:

```text
V in {0,1}^{3 x H x W}
```

then:

```text
avail = sum over axis 0 of V
```

so:

```text
avail in {0,1,2,3}^{H x W}
```

The valid label extent is:

```text
valid_extent(x) = 1 if avail(x) > 0 else 0
```

or:

```text
valid_extent(x) = 1[avail(x) > 0]
```

### 5.3 Soft Target

The soft vote fraction is:

```text
soft(x) = votes(x) / max(avail(x), 1)
```

This quantity lies in:

```text
soft(x) in [0, 1]
```

and can be interpreted as the empirical fraction of available sources that vote positive at pixel `x`.

So:

- `soft(x) = 0` means no available source voted positive
- `soft(x) = 1` means every available source voted positive
- intermediate values mean partial agreement among available sources

This is saved, but the default training target is the hard consensus target below.

### 5.4 Default Hard Target: Consensus 2-of-3

The default fusion mode is:

```text
target(x) = 1 if votes(x) >= 2 else 0
```

Using indicator notation:

```text
target(x) = 1[votes(x) >= 2]
```

This means a pixel is labeled positive only when at least two weak sources agree.

Because `R(x)` itself depends on the chosen RADD mode, the full default target can be written as:

```text
y(x ; mode) = 1[R_mode(x) + G_L(x) + G_S2(x) >= 2]
```

where:

```text
mode in {permissive, conservative}
```

and:

```text
R_permissive(x)   = 1[raw_radd(x) > 0]
R_conservative(x) = 1[floor(raw_radd(x) / 10000) >= 3]
```

At the tile level, the final fused target raster is therefore:

```text
y = F(L, V)
```

where `F` is the fusion operator determined by:

- the binarization rules for each source
- the valid-extent masks
- the chosen fusion method
- the ignore-mask settings
- the per-vote weight mapping

For the default setup, `F` acts pixelwise:

```text
F(L, V)(x) = 1[sum_{s in S} L_s(x) >= 2]
```

with additional outputs:

```text
w(x) = alpha_{votes(x)}
m(x) = 1[avail(x)=0] OR beta(x)
```

where:

- `alpha_k` is the configured vote-weight lookup
- `beta(x) = 1[votes(x)=1]` only when `ignore_uncertain_single_source_positives=true`

Other implemented options are:

- `union`: `target(x) = 1 if votes(x) >= 1`
- `unanimous`: `target(x) = 1 if votes(x) == 3`
- `soft_vote`: `target(x) = 1 if soft(x) >= soft_vote_threshold`

The default hard target over the full tile can therefore be viewed as a function:

```text
y : Omega -> {0,1},  y(x) = 1[votes(x) >= 2]
```

## 6. Loss Weighting

The pipeline also creates a per-pixel weight map from the vote count.

Default weights:

```text
w(votes=3) = 1.0
w(votes=2) = 0.8
w(votes=1) = 0.3
w(votes=0) = 1.0
```

So the weight function is:

```text
weight(x) =
    1.0 if votes(x) = 3
    0.8 if votes(x) = 2
    0.3 if votes(x) = 1
    1.0 if votes(x) = 0
```

Equivalently, as a lookup table:

```text
weight(x) = alpha_{votes(x)}
```

with:

```text
alpha_0 = 1.0
alpha_1 = 0.3
alpha_2 = 0.8
alpha_3 = 1.0
```

So the full weak-supervision output of the fusion stage can be written as the tuple:

```text
(y, soft, w, m, votes, valid_extent, avail)
```

with each component defined on `Omega`.

Then ignored pixels are zeroed out:

```text
if ignore(x) = 1, then weight(x) = 0
```

This lets the model trust strong consensus more than weak single-source positives.

## 7. Ignore Mask

The ignore mask is built in two parts.

### 7.1 Ignore Outside Label Extent

If no source covers a pixel, it can be ignored:

```text
ignore(x) = 1 if valid_extent(x) = 0
```

This is the default behavior because:

- we do not want to train on pixels where no weak label source provides any supervision

### 7.2 Optional Ignore For Single-Source Positives

If configured, pixels with exactly one positive vote are also ignored:

```text
ignore(x) = 1 if votes(x) = 1
```

Default:

```text
ignore_uncertain_single_source_positives = false
```

So by default they are not ignored; they are simply given lower weight.

Over the full tile, the effective supervised set is:

```text
Omega_sup = {x in Omega : m(x) = 0}
```

Only pixels in `Omega_sup` contribute to the loss.

This means the loss is effectively evaluated over:

```text
Omega_sup subseteq Omega
```

rather than all pixels in the tile.

## 8. Worked Example

Suppose after reprojection onto the Sentinel-2 grid we have one pixel `x` with:

```text
R(x)    = 1
G_L(x)  = 0
G_S2(x) = 1
```

and all three sources cover the pixel:

```text
V_R(x) = V_GL(x) = V_GS2(x) = 1
```

Then:

```text
votes(x) = 1 + 0 + 1 = 2
avail(x) = 1 + 1 + 1 = 3
soft(x)  = 2 / 3
target(x) = 1   because votes(x) >= 2
weight(x) = 0.8 because votes(x) = 2
ignore(x) = 0   because the pixel is inside the valid label extent
```

Another example:

```text
R(x)    = 1
G_L(x)  = 0
G_S2(x) = 0
```

Then:

```text
votes(x) = 1
target(x) = 0      under consensus_2of3
weight(x) = 0.3
ignore(x) = 0      by default
```

If `ignore_uncertain_single_source_positives=true`, then:

```text
ignore(x) = 1
weight(x) = 0
```

Now consider the same example under two RADD modes:

```text
raw_radd(x) = 21234
G_L(x) = 1
G_S2(x) = 0
```

Then:

```text
R_perm(x) = 1  -> votes_perm(x) = 2 -> y_perm(x) = 1
R_cons(x) = 0  -> votes_cons(x) = 1 -> y_cons(x) = 0
```

So a single configuration switch can change the final target at that pixel even though GLAD-L and GLAD-S2 stayed the same.

A compact way to summarize the pipeline at one pixel is:

```text
ell(x) -> votes(x), avail(x) -> y(x), w(x), m(x)
```

## 9. Shape Changes Through The Label Pipeline

For one tile, the label pipeline changes shape like this.

### 9.1 Before Reprojection

Each source begins as one or more single-band rasters on its own native grid:

- RADD: `1 x H_radd x W_radd`
- GLAD-S2: `1 x H_gs2 x W_gs2`
- GLAD-L: `5 x H_gl x W_gl` across the five yearly files, one band each

The CRS and shape may differ from the Sentinel-2 tile grid.

### 9.2 After Reprojection To Sentinel-2 Grid

Each source is mapped to the Sentinel-2 tile geometry:

- RADD binary: `[H, W]`
- GLAD-S2 binary: `[H, W]`
- GLAD-L yearly alerts: `5 x H x W`, then OR-reduced to one `[H, W]`

Temporary stacked source tensor:

```text
[3, H, W]
```

More formally:

```text
L in {0,1}^(|S| x H x W) = {0,1}^(3 x H x W)
V in {0,1}^(|S| x H x W) = {0,1}^(3 x H x W)
```

where:

```text
L[0, :, :] = R
L[1, :, :] = G_L
L[2, :, :] = G_S2
```

and similarly:

```text
V[0, :, :] = V_R
V[1, :, :] = V_GL
V[2, :, :] = V_GS2
```

The fusion stage is therefore a map:

```text
({0,1}^{3 x H x W}, {0,1}^{3 x H x W}) -> (
    {0,1}^{H x W},
    [0,1]^{H x W},
    R^{H x W},
    {0,1}^{H x W},
    {0,1,2,3}^{H x W},
    {0,1}^{H x W},
    {0,1,2,3}^{H x W}
)
```

corresponding to:

```text
(L, V) -> (y, soft, w, m, votes, valid_extent, avail)
```

because the fused logic stacks:

- one binary RADD mask
- one binary GLAD-L mask
- one binary GLAD-S2 mask

### 9.3 Final Label Outputs

The final supervision stored for training is:

- `target`: `[H, W]`
- `weight_map`: `[H, W]`
- `ignore_mask`: `[H, W]`
- `vote_count`: `[H, W]`

So the label side ends with one spatial mask per product, not a multi-channel semantic label tensor.

The important geometric fact is that all of these are dense rasters on the same domain `Omega`, so there is no later resizing step inside training.

## 10. Relationship To The Model Input Tensor

The current preprocessing default uses snapshot-pair multimodal features.

For one tile, the model input is:

```text
X in R^[63, H, W]
```

or equivalently:

```text
X : Omega -> R^63
```

with:

- Sentinel-2 early: `12`
- Sentinel-2 late: `12`
- Sentinel-2 delta: `12`
- Sentinel-1 early: `1`
- Sentinel-1 late: `1`
- Sentinel-1 delta: `1`
- AEF early PCA: `8`
- AEF late PCA: `8`
- AEF delta PCA: `8`

Total:

```text
36 + 3 + 24 = 63 channels
```

The fused training target is:

```text
y in {0,1}^[H, W]
```

The weight map is:

```text
w in R^[H, W]
```

The ignore mask is:

```text
m in {0,1}^[H, W]
```

So the input/output geometry is:

```text
input  = [63, H, W]
target = [H, W]
```

At the per-pixel level:

```text
X(:, i, j) in R^63
y(i, j)    in {0,1}
w(i, j)    in R_{>=0}
m(i, j)    in {0,1}
```

The channel count changes, but the pixel grid does not.

In other words, for every pixel `x in Omega`:

```text
X(x) contains the input features at x
y(x) is the fused supervision at x
w(x) is the confidence weight at x
m(x) says whether x is ignored
```

So fusion defines the supervision field on exactly the same domain used by the model input.

## 11. Model Output Shape

During training on patches of size `P x P`, the model outputs:

```text
[B, 1, P, P]
```

If `f_theta` is the segmentation model with parameters `theta`, then for one patch:

```text
f_theta : R^(C x P x P) -> R^(1 x P x P)
```

and after the sigmoid:

```text
p_theta : Omega_patch -> [0,1]
```

with:

```text
p_theta(x) = sigma(f_theta(X_patch)(x))
```

For example, with:

- `batch_size = 8`
- `patch_size = 128`

the model output is:

```text
[8, 1, 128, 128]
```

For a full tile during inference, the stitched output is:

```text
[1, H, W]
```

and after thresholding it becomes a single-band binary mask:

```text
[H, W]
```

Because preprocessing aligned everything to the Sentinel-2 grid, the final prediction has the same pixel layout as the fused target.

## 12. Concrete Pixel-Matching Summary

For each tile:

1. pick Sentinel-2 reference raster
2. read and reproject label rasters to that exact grid
3. convert each weak source to a binary `[H, W]` mask
4. compute vote-based fused target on `[H, W]`
5. build input features on the same `[H, W]` grid
6. train a model that maps `[C, H, W] -> [1, H, W]`

That is why the number of output pixels always matches the number of target pixels.

Another useful way to say it is:

```text
|Omega_input| = |Omega_target| = H * W
```

Only the feature dimension changes:

```text
input channels  = C
output channels = 1
```

The segmentation model changes feature dimension, not geometry:

```text
R^(C x H x W) -> R^(1 x H x W)
```

and the label fusion stage is what provides the target on that same `H x W` lattice.

## 13. Practical Example

Suppose one tile ends up on the Sentinel-2 grid with:

```text
H = 1002
W = 1002
```

Then the cached artifacts are shaped like:

```text
features     = [63, 1002, 1002]
target       = [1002, 1002]
weight_map   = [1002, 1002]
ignore_mask  = [1002, 1002]
vote_count   = [1002, 1002]
```

A full-tile prediction from the model will correspond to:

```text
probability  = [1002, 1002]
binary mask  = [1002, 1002]
```

So there is a one-to-one pixel correspondence between:

- the fused training target
- the model prediction
- the exported output raster

## 14. Important Caveat

These are fused weak labels, not human-verified definitive labels.

That means:

- some positives may be false positives
- some negatives may be false negatives
- consensus and pixel weighting are used to reduce this noise, not eliminate it

So the repository treats the fused target as:

- training supervision

rather than:

- guaranteed ground truth
