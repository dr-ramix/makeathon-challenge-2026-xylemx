"""Generate high-resolution, professional visualizations for model and data diagnostics.

This script creates rich per-tile visual artifacts from:
- raw satellite observations (Sentinel-2 / Sentinel-1),
- preprocessed feature tensors,
- fused labels (target / ignore / vote_count / weight_map),
- model predictions from the best checkpoint.

It is designed for qualitative analysis, reporting, and model debugging.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from xylemx.config import ExperimentConfig
from xylemx.data.io import (
    S1_PATTERN,
    S2_PATTERN,
    get_feature_path,
    get_ignore_mask_path,
    get_preview_path,
    get_target_path,
    get_valid_mask_path,
    get_vote_count_path,
    get_weight_map_path,
    load_json,
    scan_tiles,
)
from xylemx.models.baseline import build_model
from xylemx.training.inference import load_normalized_tile, predict_probability_map

plt.style.use("seaborn-v0_8-whitegrid")


@dataclass(slots=True)
class ModelBundle:
    checkpoint_path: Path
    checkpoint_payload: dict[str, Any]
    config: ExperimentConfig
    model: torch.nn.Module
    device: torch.device
    threshold: float
    tta: bool
    preprocessing_dir_from_checkpoint: Path
    data_root_from_checkpoint: Path


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_percentile(values: np.ndarray, q: float, fallback: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    return float(np.percentile(finite, q))


def _scale_to_unit(values: np.ndarray, *, valid: np.ndarray | None = None, q_low: float = 2.0, q_high: float = 98.0) -> np.ndarray:
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = valid & np.isfinite(values)
    if not valid.any():
        return np.zeros(values.shape, dtype=np.float32)
    lo = float(np.percentile(values[valid], q_low))
    hi = float(np.percentile(values[valid], q_high))
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi):
        hi = 1.0
    if abs(hi - lo) < 1e-6:
        return np.zeros(values.shape, dtype=np.float32)
    scaled = (values - lo) / (hi - lo)
    return np.clip(np.nan_to_num(scaled, nan=0.0), 0.0, 1.0).astype(np.float32)


def _to_uint8_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray, *, valid: np.ndarray | None = None) -> np.ndarray:
    r = _scale_to_unit(red, valid=valid)
    g = _scale_to_unit(green, valid=valid)
    b = _scale_to_unit(blue, valid=valid)
    rgb = np.stack([r, g, b], axis=-1)
    return np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def _feature_map(feature_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(feature_names)}


def _get_feature_layer(features: np.ndarray, fmap: dict[str, int], name: str) -> np.ndarray | None:
    idx = fmap.get(name)
    if idx is None or idx >= features.shape[0]:
        return None
    layer = features[idx].astype(np.float32)
    return layer


def _pick_stage(feature_names: list[str], prefix: str, preferred: list[str] | None = None) -> str | None:
    preferred = preferred or ["late", "middle2", "middle", "middle1", "early"]
    stage_candidates: list[str] = []
    ignored = {"delta", "valid", "missing", "any"}
    for name in feature_names:
        token = f"{prefix}_"
        if not name.startswith(token):
            continue
        tail = name[len(token) :]
        if not tail:
            continue
        stage = tail.split("_", 1)[0]
        if stage in ignored:
            continue
        stage_candidates.append(stage)
    unique = list(dict.fromkeys(stage_candidates))
    if not unique:
        return None
    for stage in preferred:
        if stage in unique:
            return stage
    return unique[-1]


def _build_s2_composites(features: np.ndarray, fmap: dict[str, int], feature_names: list[str]) -> dict[str, np.ndarray]:
    stage = _pick_stage(feature_names, "s2")
    if stage is None:
        return {}

    red = _get_feature_layer(features, fmap, f"s2_{stage}_B04")
    green = _get_feature_layer(features, fmap, f"s2_{stage}_B03")
    blue = _get_feature_layer(features, fmap, f"s2_{stage}_B02")
    nir = _get_feature_layer(features, fmap, f"s2_{stage}_B08")
    swir2 = _get_feature_layer(features, fmap, f"s2_{stage}_B12")
    if red is None or green is None or blue is None:
        return {}

    valid = np.isfinite(red) & np.isfinite(green) & np.isfinite(blue)
    outputs: dict[str, np.ndarray] = {
        "s2_true_color": _to_uint8_rgb(red, green, blue, valid=valid),
    }
    if nir is not None:
        outputs["s2_false_color_nir"] = _to_uint8_rgb(nir, red, green, valid=np.isfinite(nir) & valid)
        ndvi = np.full(red.shape, np.nan, dtype=np.float32)
        denom = nir + red
        good = np.isfinite(nir) & np.isfinite(red) & (np.abs(denom) > 1e-6)
        ndvi[good] = (nir[good] - red[good]) / denom[good]
        outputs["s2_ndvi"] = np.clip(np.nan_to_num(ndvi, nan=0.0), -1.0, 1.0)
    if nir is not None and swir2 is not None:
        outputs["s2_swir_nir_red"] = _to_uint8_rgb(swir2, nir, red, valid=np.isfinite(swir2) & np.isfinite(nir) & np.isfinite(red))
    return outputs


def _build_s1_views(features: np.ndarray, fmap: dict[str, int], feature_names: list[str]) -> dict[str, np.ndarray]:
    stage = _pick_stage(feature_names, "s1")
    outputs: dict[str, np.ndarray] = {}
    if stage is not None:
        vv = _get_feature_layer(features, fmap, f"s1_{stage}")
        if vv is not None:
            outputs["s1_radar_vv"] = vv
    delta = _get_feature_layer(features, fmap, "s1_delta")
    if delta is not None:
        outputs["s1_radar_delta"] = delta
    return outputs


def _build_aef_views(features: np.ndarray, fmap: dict[str, int], feature_names: list[str]) -> dict[str, np.ndarray]:
    stage = _pick_stage(feature_names, "aef")
    if stage is None:
        return {}
    pc1 = _get_feature_layer(features, fmap, f"aef_{stage}_pc01")
    pc2 = _get_feature_layer(features, fmap, f"aef_{stage}_pc02")
    pc3 = _get_feature_layer(features, fmap, f"aef_{stage}_pc03")
    outputs: dict[str, np.ndarray] = {}
    if pc1 is not None:
        outputs["aef_pc01"] = pc1
    if pc2 is not None:
        outputs["aef_pc02"] = pc2
    if pc3 is not None:
        outputs["aef_pc03"] = pc3
    if pc1 is not None and pc2 is not None and pc3 is not None:
        valid = np.isfinite(pc1) & np.isfinite(pc2) & np.isfinite(pc3)
        outputs["aef_rgb"] = _to_uint8_rgb(pc1, pc2, pc3, valid=valid)
    return outputs


def _blend_overlay(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    base = base_rgb.astype(np.float32).copy()
    color_vector = np.asarray(color, dtype=np.float32)
    positive = mask.astype(bool)
    base[positive] = (1.0 - alpha) * base[positive] + alpha * color_vector
    return np.clip(base, 0.0, 255.0).astype(np.uint8)


def _confusion_map(target: np.ndarray, pred: np.ndarray, ignore_mask: np.ndarray | None) -> np.ndarray:
    truth = target.astype(bool)
    guess = pred.astype(bool)
    ignored = np.zeros_like(truth, dtype=bool) if ignore_mask is None else ignore_mask.astype(bool)

    canvas = np.zeros((*truth.shape, 3), dtype=np.uint8)
    canvas[:, :] = np.array([24, 24, 24], dtype=np.uint8)
    canvas[ignored] = np.array([90, 90, 90], dtype=np.uint8)
    canvas[truth & guess & ~ignored] = np.array([34, 197, 94], dtype=np.uint8)   # TP
    canvas[~truth & guess & ~ignored] = np.array([239, 68, 68], dtype=np.uint8)   # FP
    canvas[truth & ~guess & ~ignored] = np.array([245, 158, 11], dtype=np.uint8)  # FN
    return canvas


def _metrics(target: np.ndarray, pred: np.ndarray, ignore_mask: np.ndarray | None) -> dict[str, float]:
    truth = target.astype(bool)
    guess = pred.astype(bool)
    ignored = np.zeros_like(truth, dtype=bool) if ignore_mask is None else ignore_mask.astype(bool)
    valid = ~ignored

    tp = float(np.logical_and(truth, guess)[valid].sum())
    fp = float(np.logical_and(~truth, guess)[valid].sum())
    fn = float(np.logical_and(truth, ~guess)[valid].sum())
    denom_iou = tp + fp + fn
    denom_dice = 2.0 * tp + fp + fn
    iou = tp / denom_iou if denom_iou > 0 else 0.0
    dice = (2.0 * tp) / denom_dice if denom_dice > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "iou": iou, "dice": dice}


def _save_single_layer(
    path: Path,
    *,
    layer: np.ndarray,
    title: str,
    dpi: int,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    show_colorbar: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.5, 7.5))
    image = axis.imshow(layer, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=12, fontweight="bold")
    axis.axis("off")
    if show_colorbar and cmap is not None:
        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.ax.tick_params(labelsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_overview_panel(
    path: Path,
    *,
    tile_id: str,
    split: str,
    dpi: int,
    layers: list[dict[str, Any]],
    panel_title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = 4
    rows = int(math.ceil(len(layers) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.7 * rows))
    axes = np.atleast_1d(axes).reshape(rows, columns)
    for axis in axes.flat:
        axis.axis("off")

    for axis, layer in zip(axes.flat, layers, strict=False):
        image = axis.imshow(layer["array"], cmap=layer.get("cmap"), vmin=layer.get("vmin"), vmax=layer.get("vmax"))
        axis.set_title(layer["title"], fontsize=11, fontweight="bold")
        axis.axis("off")
        if layer.get("show_colorbar", False):
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.ax.tick_params(labelsize=7)

    figure.suptitle(f"{tile_id} | {split} | {panel_title}", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _choose_evenly_spaced_indices(total: int, limit: int) -> np.ndarray:
    if total <= 0:
        return np.asarray([], dtype=np.int64)
    if total <= limit:
        return np.arange(total, dtype=np.int64)
    return np.unique(np.round(np.linspace(0, total - 1, num=limit)).astype(np.int64))


def _read_s2_rgb(path: Path) -> np.ndarray | None:
    with rasterio.open(path) as src:
        if src.count < 4:
            return None
        # B04, B03, B02 -> true color
        data = src.read([4, 3, 2], masked=True).astype(np.float32)
        filled = np.asarray(data.filled(np.nan), dtype=np.float32)
    if filled.ndim != 3:
        return None
    return filled


def _read_s1_vv(path: Path) -> np.ndarray | None:
    with rasterio.open(path) as src:
        if src.count < 1:
            return None
        data = src.read(1, masked=True).astype(np.float32)
        return np.asarray(data.filled(np.nan), dtype=np.float32)


def _sample_finite(values: np.ndarray, *, max_count: int, seed: int = 42) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size <= max_count:
        return finite
    rng = np.random.default_rng(seed)
    indices = rng.choice(finite.size, size=max_count, replace=False)
    return finite[indices]


def _save_s2_timeline(path: Path, s2_paths: list[Path], *, max_steps: int, dpi: int) -> int:
    if not s2_paths:
        return 0
    chosen_indices = _choose_evenly_spaced_indices(len(s2_paths), max_steps)
    chosen_paths = [s2_paths[int(idx)] for idx in chosen_indices]
    stacks: list[np.ndarray] = []
    labels: list[str] = []
    for item in chosen_paths:
        rgb_stack = _read_s2_rgb(item)
        if rgb_stack is None:
            continue
        stacks.append(rgb_stack)
        match = S2_PATTERN.match(item.name)
        if match:
            labels.append(f"{match.group('year')}-{int(match.group('month')):02d}")
        else:
            labels.append(item.stem)
    if not stacks:
        return 0

    channel_samples: list[list[np.ndarray]] = [[], [], []]
    for channel in range(3):
        for frame in stacks:
            channel_samples[channel].append(_sample_finite(frame[channel], max_count=200000, seed=42 + channel))

    channel_ranges: list[tuple[float, float]] = []
    for channel in range(3):
        samples = [sample for sample in channel_samples[channel] if sample.size > 0]
        if not samples:
            channel_ranges.append((0.0, 1.0))
            continue
        merged_samples = np.concatenate(samples, axis=0)
        lo = float(np.percentile(merged_samples, 2.0))
        hi = float(np.percentile(merged_samples, 98.0))
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = 1.0
        if abs(hi - lo) < 1e-6:
            hi = lo + 1.0
        channel_ranges.append((lo, hi))

    rgb_uint8: list[np.ndarray] = []
    for frame in stacks:
        scaled = np.zeros((frame.shape[1], frame.shape[2], 3), dtype=np.float32)
        for channel in range(3):
            lo, hi = channel_ranges[channel]
            denom = max(hi - lo, 1e-6)
            scaled[..., channel] = np.clip((frame[channel] - lo) / denom, 0.0, 1.0)
        rgb_uint8.append(np.round(np.nan_to_num(scaled, nan=0.0) * 255.0).astype(np.uint8))

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = len(rgb_uint8)
    figure, axes = plt.subplots(1, columns, figsize=(4.2 * columns, 4.3))
    axes = np.atleast_1d(axes)
    for axis, image, label in zip(axes, rgb_uint8, labels, strict=True):
        axis.imshow(image)
        axis.set_title(label, fontsize=10, fontweight="bold")
        axis.axis("off")
    figure.suptitle("Sentinel-2 Temporal RGB Timeline", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.01, 1.0, 0.95))
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return int(len(rgb_uint8))


def _save_s1_timeline(path: Path, s1_paths: list[Path], *, max_steps: int, dpi: int) -> int:
    if not s1_paths:
        return 0

    grouped: dict[tuple[int, int], list[Path]] = {}
    for item in s1_paths:
        match = S1_PATTERN.match(item.name)
        if match is None:
            continue
        key = (int(match.group("year")), int(match.group("month")))
        grouped.setdefault(key, []).append(item)
    if not grouped:
        return 0

    ordered_keys = sorted(grouped.keys())
    chosen_indices = _choose_evenly_spaced_indices(len(ordered_keys), max_steps)
    chosen_keys = [ordered_keys[int(idx)] for idx in chosen_indices]

    frames: list[np.ndarray] = []
    labels: list[str] = []
    for year, month in chosen_keys:
        arrays: list[np.ndarray] = []
        for source in grouped[(year, month)]:
            vv = _read_s1_vv(source)
            if vv is not None:
                arrays.append(vv)
        if not arrays:
            continue

        min_height = min(array.shape[0] for array in arrays)
        min_width = min(array.shape[1] for array in arrays)
        aligned = [array[:min_height, :min_width] for array in arrays]
        stacked = np.stack(aligned, axis=0).astype(np.float32)
        finite = np.isfinite(stacked)
        counts = finite.sum(axis=0).astype(np.int16)
        weighted_sum = np.where(finite, stacked, 0.0).sum(axis=0, dtype=np.float32)
        averaged = np.full((min_height, min_width), np.nan, dtype=np.float32)
        np.divide(weighted_sum, np.maximum(counts, 1), out=averaged, where=counts > 0)
        frames.append(averaged)
        labels.append(f"{year:04d}-{month:02d}")
    if not frames:
        return 0

    sampled_frames = [_sample_finite(frame, max_count=200000, seed=777) for frame in frames]
    sampled_frames = [sample for sample in sampled_frames if sample.size > 0]
    if sampled_frames:
        merged_samples = np.concatenate(sampled_frames, axis=0)
        lo = float(np.percentile(merged_samples, 2.0))
        hi = float(np.percentile(merged_samples, 98.0))
    else:
        lo, hi = 0.0, 1.0
    denom = max(float(hi - lo), 1e-6)

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = len(frames)
    figure, axes = plt.subplots(1, columns, figsize=(4.2 * columns, 4.3))
    axes = np.atleast_1d(axes)
    for axis, image, label in zip(axes, frames, labels, strict=True):
        scaled = np.clip((image - lo) / denom, 0.0, 1.0)
        axis.imshow(np.nan_to_num(scaled, nan=0.0), cmap="magma", vmin=0.0, vmax=1.0)
        axis.set_title(label, fontsize=10, fontweight="bold")
        axis.axis("off")
    figure.suptitle("Sentinel-1 Radar Timeline (VV)", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.01, 1.0, 0.95))
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return int(len(frames))


def _save_threshold_curve(
    path: Path,
    *,
    probability: np.ndarray,
    target: np.ndarray,
    ignore_mask: np.ndarray,
    dpi: int,
) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 19)
    dice_values: list[float] = []
    iou_values: list[float] = []
    best_threshold = 0.5
    best_dice = -1.0
    for threshold in thresholds:
        pred = (probability >= threshold).astype(np.uint8)
        stats = _metrics(target, pred, ignore_mask)
        dice_values.append(stats["dice"])
        iou_values.append(stats["iou"])
        if stats["dice"] > best_dice:
            best_dice = stats["dice"]
            best_threshold = float(threshold)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.plot(thresholds, dice_values, linewidth=2.5, marker="o", markersize=4, label="Dice")
    axis.plot(thresholds, iou_values, linewidth=2.0, marker="s", markersize=3.5, label="IoU")
    axis.axvline(best_threshold, linestyle="--", color="black", linewidth=1.5, label=f"Best Dice @{best_threshold:.2f}")
    axis.set_title("Threshold Sweep", fontsize=13, fontweight="bold")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return best_threshold, best_dice


def _save_label_fusion_diagnostics(
    path: Path,
    *,
    target: np.ndarray,
    ignore_mask: np.ndarray,
    vote_count: np.ndarray | None,
    weight_map: np.ndarray | None,
    dpi: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax = axes.ravel()

    ax[0].imshow(target, cmap="gray", vmin=0.0, vmax=1.0)
    ax[0].set_title("Fused Target")
    ax[0].axis("off")

    ax[1].imshow(ignore_mask.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)
    ax[1].set_title("Ignore Mask")
    ax[1].axis("off")

    if vote_count is not None:
        vote_image = ax[2].imshow(vote_count, cmap="viridis", vmin=0, vmax=3)
        ax[2].set_title("Vote Count (0-3)")
        ax[2].axis("off")
        figure.colorbar(vote_image, ax=ax[2], fraction=0.046, pad=0.04)
    else:
        ax[2].text(0.5, 0.5, "vote_count not available", ha="center", va="center")
        ax[2].axis("off")

    if weight_map is not None:
        weights = weight_map[np.isfinite(weight_map)]
        if weights.size > 0:
            ax[3].hist(weights, bins=30, color="#2563eb", alpha=0.9)
        ax[3].set_title("Weight Map Distribution")
        ax[3].set_xlabel("weight")
        ax[3].set_ylabel("pixels")
    else:
        ax[3].text(0.5, 0.5, "weight_map not available", ha="center", va="center")
        ax[3].axis("off")

    figure.suptitle("Label Fusion Diagnostics", fontsize=15, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.01, 1.0, 0.95))
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_preprocessing_histograms(
    path: Path,
    *,
    features: np.ndarray,
    fmap: dict[str, int],
    normalization_stats: dict[str, Any] | None,
    dpi: int,
) -> None:
    channel_candidates = [
        "s2_late_B04",
        "s2_late_B08",
        "s1_late",
        "aef_late_pc01",
    ]
    selected: list[tuple[str, np.ndarray, np.ndarray | None]] = []

    means = np.asarray(normalization_stats.get("mean", []), dtype=np.float32) if normalization_stats else np.asarray([], dtype=np.float32)
    stds = np.asarray(normalization_stats.get("std", []), dtype=np.float32) if normalization_stats else np.asarray([], dtype=np.float32)

    for name in channel_candidates:
        idx = fmap.get(name)
        if idx is None or idx >= features.shape[0]:
            continue
        raw = features[idx].astype(np.float32)
        normalized: np.ndarray | None = None
        if idx < len(means) and idx < len(stds):
            std = float(stds[idx]) if np.isfinite(stds[idx]) else 1.0
            std = max(std, 1e-6)
            normalized = (raw - float(means[idx])) / std
        selected.append((name, raw, normalized))

    if not selected:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(selected)
    figure, axes = plt.subplots(rows, 1, figsize=(10, 3.5 * rows))
    axes = np.atleast_1d(axes)

    for axis, (name, raw, normalized) in zip(axes, selected, strict=True):
        raw_values = raw[np.isfinite(raw)]
        if raw_values.size > 0:
            axis.hist(raw_values, bins=80, alpha=0.6, color="#0f766e", label=f"{name} raw", density=True)
        if normalized is not None:
            normalized_values = normalized[np.isfinite(normalized)]
            if normalized_values.size > 0:
                axis.hist(normalized_values, bins=80, alpha=0.45, color="#dc2626", label=f"{name} normalized", density=True)
        axis.set_title(name)
        axis.legend(loc="upper right")
        axis.grid(True, alpha=0.2)

    figure.suptitle("Preprocessing Diagnostics: Raw vs Normalized Channels", fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _resolve_best_checkpoint(
    *,
    checkpoint: str | None,
    run_dir: str | None,
    output_root: str | None,
) -> Path | None:
    if checkpoint:
        candidate = Path(checkpoint)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Checkpoint does not exist: {candidate}")

    if run_dir:
        candidate = Path(run_dir) / "checkpoints" / "best.pt"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"best.pt not found under run_dir: {candidate}")

    if output_root:
        root = Path(output_root)
        leaderboard = root / "leaderboard.json"
        if leaderboard.exists():
            payload = load_json(leaderboard)
            if isinstance(payload, list):
                for row in payload:
                    candidate = Path(str(row.get("run_dir", ""))) / "checkpoints" / "best.pt"
                    if candidate.exists():
                        return candidate
        discovered = sorted(root.glob("*/checkpoints/best.pt"), reverse=True)
        if discovered:
            return discovered[0]

    return None


def _load_model_bundle(
    checkpoint_path: Path,
    *,
    threshold_override: float | None,
    tta_override: bool | None,
    device_name: str,
) -> ModelBundle:
    payload = torch.load(checkpoint_path, map_location="cpu")
    config = ExperimentConfig(**payload["config"])
    device = _resolve_device(device_name)
    model = build_model(
        config.model,
        in_channels=int(payload["in_channels"]),
        dropout=config.dropout,
        stochastic_depth=config.stochastic_depth,
        pretrained=config.encoder_pretrained,
    ).to(device)
    state_dict = payload["ema_state_dict"] or payload["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    threshold = float(config.inference_threshold if threshold_override is None else threshold_override)
    tta = bool(config.tta if tta_override is None else tta_override)
    return ModelBundle(
        checkpoint_path=checkpoint_path,
        checkpoint_payload=payload,
        config=config,
        model=model,
        device=device,
        threshold=threshold,
        tta=tta,
        preprocessing_dir_from_checkpoint=Path(str(payload["preprocessing_dir"])),
        data_root_from_checkpoint=Path(config.data_root),
    )


def _resolve_visualization_inputs(
    *,
    preprocessing_dir_arg: str | None,
    data_root_arg: str | None,
    model_bundle: ModelBundle | None,
) -> tuple[Path, Path]:
    if preprocessing_dir_arg:
        preprocessing_dir = Path(preprocessing_dir_arg)
    elif model_bundle is not None:
        preprocessing_dir = model_bundle.preprocessing_dir_from_checkpoint
    else:
        raise ValueError("preprocessing_dir is required when no checkpoint/run_dir/output_root best model can be resolved")

    if data_root_arg:
        data_root = Path(data_root_arg)
    elif model_bundle is not None:
        data_root = model_bundle.data_root_from_checkpoint
    else:
        data_root = Path("data/makeathon-challenge")

    return preprocessing_dir, data_root


def _select_tile_ids(
    *,
    split: str,
    preprocessing_dir: Path,
    records: dict[str, Any],
    tile_ids_override: list[str],
    num_tiles: int,
    sample_mode: str,
    seed: int,
) -> list[str]:
    if tile_ids_override:
        selected = [tile_id for tile_id in tile_ids_override if tile_id in records]
        missing = [tile_id for tile_id in tile_ids_override if tile_id not in records]
        if missing:
            print(f"[warn] Ignoring missing tile ids for split={split}: {missing}", flush=True)
        return selected

    if split == "val":
        path = preprocessing_dir / "val_tiles.json"
        if path.exists():
            candidates = [tile_id for tile_id in load_json(path) if tile_id in records]
        else:
            candidates = sorted(records.keys())
    elif split == "train":
        path = preprocessing_dir / "train_tiles.json"
        if path.exists():
            candidates = [tile_id for tile_id in load_json(path) if tile_id in records]
        else:
            candidates = sorted(records.keys())
    else:
        candidates = sorted(records.keys())

    if num_tiles <= 0 or len(candidates) <= num_tiles:
        return candidates

    if sample_mode == "first":
        return candidates[:num_tiles]

    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(np.asarray(candidates), size=num_tiles, replace=False).tolist())
    return chosen


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-dir", type=str, default="", help="Path to preprocessing artifacts")
    parser.add_argument("--data-root", type=str, default="", help="Path to raw challenge data root")
    parser.add_argument("--output-dir", type=Path, default=Path("output/professional_visualizations"))
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--num-tiles", type=int, default=8)
    parser.add_argument("--tile-ids", type=str, default="", help="Comma-separated tile ids")
    parser.add_argument("--sample-mode", type=str, choices=["random", "first"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=360)
    parser.add_argument("--max-time-steps", type=int, default=8)

    parser.add_argument("--checkpoint", type=str, default="", help="Explicit checkpoint path")
    parser.add_argument("--run-dir", type=str, default="", help="Run directory containing checkpoints/best.pt")
    parser.add_argument(
        "--output-root",
        type=str,
        default="output/training_runs",
        help="Root with leaderboard.json for automatic best model selection",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override prediction threshold")
    parser.add_argument("--tta", action="store_true", help="Force test-time augmentation on")
    parser.add_argument("--no-tta", action="store_true", help="Force test-time augmentation off")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    args = parser.parse_args()

    if args.tta and args.no_tta:
        raise ValueError("Cannot pass both --tta and --no-tta")
    tta_override = True if args.tta else False if args.no_tta else None

    checkpoint_path = _resolve_best_checkpoint(
        checkpoint=args.checkpoint or None,
        run_dir=args.run_dir or None,
        output_root=args.output_root or None,
    )
    model_bundle: ModelBundle | None = None
    if checkpoint_path is not None:
        model_bundle = _load_model_bundle(
            checkpoint_path,
            threshold_override=args.threshold,
            tta_override=tta_override,
            device_name=args.device,
        )
        print(f"[info] Using checkpoint: {checkpoint_path}", flush=True)
        print(f"[info] Model: {model_bundle.config.model}", flush=True)
        print(f"[info] Device: {model_bundle.device}", flush=True)
        print(f"[info] Threshold: {model_bundle.threshold:.3f} | TTA: {model_bundle.tta}", flush=True)
    else:
        print("[warn] No checkpoint resolved. Prediction-dependent visuals will be skipped.", flush=True)

    preprocessing_dir, data_root = _resolve_visualization_inputs(
        preprocessing_dir_arg=args.preprocessing_dir or None,
        data_root_arg=args.data_root or None,
        model_bundle=model_bundle,
    )
    if not preprocessing_dir.exists():
        raise FileNotFoundError(f"preprocessing_dir does not exist: {preprocessing_dir}")
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_for_scan = "test" if args.split == "test" else "train"
    cache_split = "test" if args.split == "test" else "train"
    records = scan_tiles(data_root, split_for_scan)
    tile_ids = _select_tile_ids(
        split=args.split,
        preprocessing_dir=preprocessing_dir,
        records=records,
        tile_ids_override=_parse_csv(args.tile_ids),
        num_tiles=args.num_tiles,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    if not tile_ids:
        raise RuntimeError("No tiles selected. Check split/tile ids/preprocessing artifacts.")
    print(f"[info] Selected {len(tile_ids)} tiles for visualization", flush=True)

    feature_spec_path = preprocessing_dir / "feature_spec.json"
    if not feature_spec_path.exists():
        raise FileNotFoundError(f"Missing feature_spec.json in preprocessing dir: {feature_spec_path}")
    feature_spec = load_json(feature_spec_path)
    feature_names = [str(name) for name in feature_spec.get("feature_names", [])]
    if not feature_names:
        raise RuntimeError(f"feature_spec.json does not contain feature_names: {feature_spec_path}")
    fmap = _feature_map(feature_names)

    normalization_stats: dict[str, Any] | None = None
    normalization_path = preprocessing_dir / "normalization_stats.json"
    if normalization_path.exists():
        normalization_stats = load_json(normalization_path)

    dataset_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for index, tile_id in enumerate(tile_ids, start=1):
            print(f"[info] [{index}/{len(tile_ids)}] Processing tile {tile_id}", flush=True)
            tile_dir = args.output_dir / tile_id
            layers_dir = tile_dir / "layers"
            layers_dir.mkdir(parents=True, exist_ok=True)

            feature_path = get_feature_path(preprocessing_dir, cache_split, tile_id)
            valid_path = get_valid_mask_path(preprocessing_dir, cache_split, tile_id)
            if not feature_path.exists() or not valid_path.exists():
                print(f"[warn] Missing features/valid_mask for tile={tile_id}, skipping", flush=True)
                continue

            features = np.load(feature_path).astype(np.float32)
            valid_mask = np.load(valid_path).astype(bool)
            preview_path = get_preview_path(preprocessing_dir, cache_split, tile_id)
            preview = np.load(preview_path) if preview_path.exists() else None
            preview_rgb = None
            if preview is not None and preview.ndim == 3:
                preview_rgb = np.moveaxis(preview, 0, -1) if preview.shape[0] in {1, 3} else preview
                preview_rgb = np.clip(preview_rgb, 0, 255).astype(np.uint8)

            target: np.ndarray | None = None
            ignore_mask: np.ndarray | None = None
            vote_count: np.ndarray | None = None
            weight_map: np.ndarray | None = None
            if split_for_scan == "train":
                target_path = get_target_path(preprocessing_dir, tile_id)
                ignore_path = get_ignore_mask_path(preprocessing_dir, tile_id)
                vote_path = get_vote_count_path(preprocessing_dir, tile_id)
                weight_path = get_weight_map_path(preprocessing_dir, tile_id)
                if target_path.exists():
                    target = np.load(target_path).astype(np.float32)
                if ignore_path.exists():
                    ignore_mask = np.load(ignore_path).astype(bool)
                if vote_path.exists():
                    vote_count = np.load(vote_path)
                if weight_path.exists():
                    weight_map = np.load(weight_path).astype(np.float32)

            probability: np.ndarray | None = None
            pred: np.ndarray | None = None
            if model_bundle is not None:
                normalized_features, normalized_valid = load_normalized_tile(
                    preprocessing_dir,
                    split=cache_split,
                    tile_id=tile_id,
                    config=model_bundle.config,
                )
                probability = predict_probability_map(
                    model_bundle.model,
                    normalized_features,
                    normalized_valid,
                    device=model_bundle.device,
                    patch_size=model_bundle.config.patch_size,
                    stride=model_bundle.config.eval_stride,
                    batch_size=model_bundle.config.batch_size,
                    mixed_precision=model_bundle.config.mixed_precision,
                    tta=model_bundle.tta,
                    tta_modes=model_bundle.config.tta_modes,
                )
                pred = ((probability >= model_bundle.threshold) & normalized_valid).astype(np.uint8)

            s2_layers = _build_s2_composites(features, fmap, feature_names)
            s1_layers = _build_s1_views(features, fmap, feature_names)
            aef_layers = _build_aef_views(features, fmap, feature_names)

            if "s2_true_color" in s2_layers:
                _save_single_layer(
                    layers_dir / "s2_true_color.png",
                    layer=s2_layers["s2_true_color"],
                    title="Sentinel-2 True Color (Late)",
                    dpi=args.dpi,
                    show_colorbar=False,
                )
            if "s2_false_color_nir" in s2_layers:
                _save_single_layer(
                    layers_dir / "s2_false_color_nir.png",
                    layer=s2_layers["s2_false_color_nir"],
                    title="Sentinel-2 False Color (NIR-R-G)",
                    dpi=args.dpi,
                    show_colorbar=False,
                )
            if "s2_swir_nir_red" in s2_layers:
                _save_single_layer(
                    layers_dir / "s2_swir_nir_red.png",
                    layer=s2_layers["s2_swir_nir_red"],
                    title="Sentinel-2 SWIR-NIR-Red",
                    dpi=args.dpi,
                    show_colorbar=False,
                )
            if "s2_ndvi" in s2_layers:
                _save_single_layer(
                    layers_dir / "s2_ndvi.png",
                    layer=s2_layers["s2_ndvi"],
                    title="NDVI (Late)",
                    dpi=args.dpi,
                    cmap="RdYlGn",
                    vmin=-1.0,
                    vmax=1.0,
                    show_colorbar=True,
                )
            if "s1_radar_vv" in s1_layers:
                _save_single_layer(
                    layers_dir / "s1_radar_vv.png",
                    layer=s1_layers["s1_radar_vv"],
                    title="Sentinel-1 Radar (VV)",
                    dpi=args.dpi,
                    cmap="magma",
                    vmin=_safe_percentile(s1_layers["s1_radar_vv"], 2.0, 0.0),
                    vmax=_safe_percentile(s1_layers["s1_radar_vv"], 98.0, 1.0),
                    show_colorbar=True,
                )
            if "s1_radar_delta" in s1_layers:
                scale = max(
                    abs(_safe_percentile(s1_layers["s1_radar_delta"], 2.0, -1.0)),
                    abs(_safe_percentile(s1_layers["s1_radar_delta"], 98.0, 1.0)),
                    1e-6,
                )
                _save_single_layer(
                    layers_dir / "s1_radar_delta.png",
                    layer=s1_layers["s1_radar_delta"],
                    title="Sentinel-1 Radar Delta (Late-Early)",
                    dpi=args.dpi,
                    cmap="coolwarm",
                    vmin=-scale,
                    vmax=scale,
                    show_colorbar=True,
                )
            if "aef_rgb" in aef_layers:
                _save_single_layer(
                    layers_dir / "aef_rgb.png",
                    layer=aef_layers["aef_rgb"],
                    title="AEF PCA Composite",
                    dpi=args.dpi,
                    show_colorbar=False,
                )
            if preview_rgb is not None:
                _save_single_layer(
                    layers_dir / "input_preview.png",
                    layer=preview_rgb,
                    title="Input Preview",
                    dpi=args.dpi,
                    show_colorbar=False,
                )

            panel_layers: list[dict[str, Any]] = []
            if "s2_true_color" in s2_layers:
                panel_layers.append({"title": "S2 True Color", "array": s2_layers["s2_true_color"], "show_colorbar": False})
            if "s2_false_color_nir" in s2_layers:
                panel_layers.append({"title": "S2 False Color", "array": s2_layers["s2_false_color_nir"], "show_colorbar": False})
            if "s2_swir_nir_red" in s2_layers:
                panel_layers.append({"title": "S2 SWIR/NIR/Red", "array": s2_layers["s2_swir_nir_red"], "show_colorbar": False})
            if "s2_ndvi" in s2_layers:
                panel_layers.append(
                    {"title": "NDVI", "array": s2_layers["s2_ndvi"], "cmap": "RdYlGn", "vmin": -1.0, "vmax": 1.0, "show_colorbar": True}
                )
            if "s1_radar_vv" in s1_layers:
                panel_layers.append(
                    {
                        "title": "S1 Radar VV",
                        "array": s1_layers["s1_radar_vv"],
                        "cmap": "magma",
                        "vmin": _safe_percentile(s1_layers["s1_radar_vv"], 2.0, 0.0),
                        "vmax": _safe_percentile(s1_layers["s1_radar_vv"], 98.0, 1.0),
                        "show_colorbar": True,
                    }
                )
            if "s1_radar_delta" in s1_layers:
                radar_delta = s1_layers["s1_radar_delta"]
                delta_scale = max(abs(_safe_percentile(radar_delta, 2.0, -1.0)), abs(_safe_percentile(radar_delta, 98.0, 1.0)), 1e-6)
                panel_layers.append(
                    {
                        "title": "S1 Delta",
                        "array": radar_delta,
                        "cmap": "coolwarm",
                        "vmin": -delta_scale,
                        "vmax": delta_scale,
                        "show_colorbar": True,
                    }
                )
            if "aef_rgb" in aef_layers:
                panel_layers.append({"title": "AEF PCA RGB", "array": aef_layers["aef_rgb"], "show_colorbar": False})
            if valid_mask is not None:
                panel_layers.append({"title": "Valid Mask", "array": valid_mask.astype(np.float32), "cmap": "gray", "vmin": 0.0, "vmax": 1.0})
            if target is not None:
                panel_layers.append({"title": "Fused Label (True)", "array": target, "cmap": "gray", "vmin": 0.0, "vmax": 1.0})
            if ignore_mask is not None:
                panel_layers.append({"title": "Ignore Mask", "array": ignore_mask.astype(np.float32), "cmap": "gray", "vmin": 0.0, "vmax": 1.0})
            if vote_count is not None:
                panel_layers.append({"title": "Vote Count", "array": vote_count, "cmap": "viridis", "vmin": 0, "vmax": 3, "show_colorbar": True})
            if weight_map is not None:
                panel_layers.append(
                    {
                        "title": "Weight Map",
                        "array": weight_map,
                        "cmap": "cividis",
                        "vmin": _safe_percentile(weight_map, 2.0, 0.0),
                        "vmax": _safe_percentile(weight_map, 98.0, 1.0),
                        "show_colorbar": True,
                    }
                )

            metrics_payload: dict[str, float] = {}
            if probability is not None:
                panel_layers.append({"title": "Prediction Probability", "array": probability, "cmap": "magma", "vmin": 0.0, "vmax": 1.0, "show_colorbar": True})
                uncertainty = 1.0 - (2.0 * np.abs(probability - 0.5))
                panel_layers.append({"title": "Uncertainty", "array": uncertainty, "cmap": "inferno", "vmin": 0.0, "vmax": 1.0, "show_colorbar": True})
            if pred is not None:
                panel_layers.append({"title": "Prediction Mask", "array": pred.astype(np.float32), "cmap": "gray", "vmin": 0.0, "vmax": 1.0})

            overlay_base = s2_layers.get("s2_true_color")
            if overlay_base is None:
                overlay_base = preview_rgb
            if overlay_base is not None and pred is not None:
                panel_layers.append({"title": "Prediction Overlay", "array": _blend_overlay(overlay_base, pred, (236, 72, 153)), "show_colorbar": False})
            if overlay_base is not None and target is not None:
                panel_layers.append({"title": "True Overlay", "array": _blend_overlay(overlay_base, target, (59, 130, 246)), "show_colorbar": False})
            if target is not None and pred is not None:
                panel_layers.append({"title": "TP / FP / FN", "array": _confusion_map(target, pred, ignore_mask), "show_colorbar": False})
                metrics_payload = _metrics(target, pred, ignore_mask)

            panel_title = f"High-Resolution Qualitative Dashboard"
            if metrics_payload:
                panel_title += f" | Dice {metrics_payload['dice']:.3f} IoU {metrics_payload['iou']:.3f}"
            _save_overview_panel(
                tile_dir / "overview_panel.png",
                tile_id=tile_id,
                split=args.split,
                dpi=args.dpi,
                layers=panel_layers,
                panel_title=panel_title,
            )

            if probability is not None and target is not None and ignore_mask is not None:
                best_threshold, best_dice = _save_threshold_curve(
                    tile_dir / "threshold_sweep.png",
                    probability=probability,
                    target=target,
                    ignore_mask=ignore_mask,
                    dpi=args.dpi,
                )
                metrics_payload["best_threshold_from_sweep"] = float(best_threshold)
                metrics_payload["best_dice_from_sweep"] = float(best_dice)

            if target is not None and ignore_mask is not None:
                _save_label_fusion_diagnostics(
                    tile_dir / "label_fusion_diagnostics.png",
                    target=target,
                    ignore_mask=ignore_mask,
                    vote_count=vote_count,
                    weight_map=weight_map,
                    dpi=args.dpi,
                )

            _save_preprocessing_histograms(
                tile_dir / "preprocessing_histograms.png",
                features=features,
                fmap=fmap,
                normalization_stats=normalization_stats,
                dpi=args.dpi,
            )

            record = records[tile_id]
            s2_steps = _save_s2_timeline(
                tile_dir / "temporal_s2_timeline.png",
                record.sentinel2_files,
                max_steps=max(args.max_time_steps, 1),
                dpi=args.dpi,
            )
            s1_steps = _save_s1_timeline(
                tile_dir / "temporal_s1_timeline.png",
                record.sentinel1_files,
                max_steps=max(args.max_time_steps, 1),
                dpi=args.dpi,
            )

            tile_summary: dict[str, Any] = {
                "tile_id": tile_id,
                "split": args.split,
                "num_feature_channels": int(features.shape[0]),
                "height": int(features.shape[1]),
                "width": int(features.shape[2]),
                "valid_fraction": float(valid_mask.mean()),
                "num_s2_steps_visualized": int(s2_steps),
                "num_s1_steps_visualized": int(s1_steps),
                "used_checkpoint": str(model_bundle.checkpoint_path) if model_bundle else None,
                "threshold": float(model_bundle.threshold) if model_bundle else None,
            }
            if target is not None and ignore_mask is not None:
                valid_pixels = (~ignore_mask).sum()
                tile_summary["target_positive_fraction"] = float(target[~ignore_mask].mean()) if valid_pixels > 0 else 0.0
            if pred is not None and ignore_mask is not None:
                valid_pixels = (~ignore_mask).sum()
                tile_summary["pred_positive_fraction"] = float(pred[~ignore_mask].mean()) if valid_pixels > 0 else 0.0
            tile_summary.update(metrics_payload)

            with (tile_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(tile_summary, handle, indent=2, sort_keys=True)

            dataset_rows.append(tile_summary)

    if dataset_rows:
        _write_csv(args.output_dir / "dataset_summary.csv", dataset_rows)

        has_model_metrics = [row for row in dataset_rows if "dice" in row]
        if has_model_metrics:
            has_model_metrics = sorted(has_model_metrics, key=lambda item: float(item["dice"]), reverse=True)
            figure, axis = plt.subplots(figsize=(max(8.0, 0.7 * len(has_model_metrics)), 5.2))
            axis.bar([row["tile_id"] for row in has_model_metrics], [row["dice"] for row in has_model_metrics], color="#2563eb", alpha=0.9, label="Dice")
            axis.plot([row["tile_id"] for row in has_model_metrics], [row.get("iou", 0.0) for row in has_model_metrics], color="#dc2626", marker="o", linewidth=2.0, label="IoU")
            axis.set_ylim(0.0, 1.0)
            axis.set_ylabel("Score")
            axis.set_xlabel("Tile")
            axis.set_title("Per-Tile Quality Summary")
            axis.tick_params(axis="x", rotation=60)
            axis.legend()
            axis.grid(True, alpha=0.2)
            figure.tight_layout()
            figure.savefig(args.output_dir / "dataset_quality_summary.png", dpi=args.dpi, bbox_inches="tight")
            plt.close(figure)

    payload = {
        "output_dir": str(args.output_dir),
        "preprocessing_dir": str(preprocessing_dir),
        "data_root": str(data_root),
        "split": args.split,
        "selected_tiles": tile_ids,
        "num_tiles_processed": len(dataset_rows),
        "checkpoint": str(model_bundle.checkpoint_path) if model_bundle else None,
        "threshold": float(model_bundle.threshold) if model_bundle else None,
        "tta": bool(model_bundle.tta) if model_bundle else None,
        "dpi": int(args.dpi),
    }
    with (args.output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    print(f"[done] Visualizations written to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
