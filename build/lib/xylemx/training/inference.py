"""Sliding-window inference helpers for full-tile prediction."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from xylemx.config import ExperimentConfig
from xylemx.data.io import get_feature_path, get_valid_mask_path, load_json
from xylemx.data.tiling import generate_patch_records
from xylemx.preprocessing.normalize import normalize_array


def _amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return nullcontext()


def _apply_tta(inputs: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "hflip":
        return torch.flip(inputs, dims=(-1,))
    if mode == "vflip":
        return torch.flip(inputs, dims=(-2,))
    if mode == "rot90":
        return torch.rot90(inputs, k=1, dims=(-2, -1))
    raise ValueError(f"Unsupported TTA mode: {mode}")


def _invert_tta(outputs: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "hflip":
        return torch.flip(outputs, dims=(-1,))
    if mode == "vflip":
        return torch.flip(outputs, dims=(-2,))
    if mode == "rot90":
        return torch.rot90(outputs, k=3, dims=(-2, -1))
    raise ValueError(f"Unsupported TTA mode: {mode}")


def load_normalized_tile(
    preprocessing_dir: str | Path,
    *,
    split: str,
    tile_id: str,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a cached feature tensor and apply train-fitted normalization."""

    preprocessing_dir = Path(preprocessing_dir)
    normalization_stats = load_json(preprocessing_dir / "normalization_stats.json")
    features = np.load(get_feature_path(preprocessing_dir, split, tile_id)).astype(np.float32)
    valid_mask = np.load(get_valid_mask_path(preprocessing_dir, split, tile_id)).astype(bool)
    features = normalize_array(features, normalization_stats, normalization=config.normalization)
    if config.normalized_feature_clip > 0:
        features = np.clip(features, -config.normalized_feature_clip, config.normalized_feature_clip)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, valid_mask


def predict_probability_map(
    model: torch.nn.Module,
    features: np.ndarray,
    valid_mask: np.ndarray,
    *,
    device: torch.device,
    patch_size: int,
    stride: int,
    batch_size: int,
    mixed_precision: bool,
    tta: bool = False,
    tta_modes: list[str] | None = None,
) -> np.ndarray:
    """Run stitched full-tile inference and return a probability map."""

    patches = generate_patch_records(
        "tile",
        shape=valid_mask.shape,
        patch_size=patch_size,
        stride=stride,
        valid_mask=valid_mask,
        min_valid_fraction=0.0,
    )
    if not patches:
        return np.zeros(valid_mask.shape, dtype=np.float32)

    prob_sum = np.zeros(valid_mask.shape, dtype=np.float32)
    count_sum = np.zeros(valid_mask.shape, dtype=np.float32)
    mode_list = ["identity"] + (tta_modes or []) if tta else ["identity"]

    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch_records = patches[start : start + batch_size]
            batch_inputs_list: list[np.ndarray] = []
            for patch in batch_records:
                patch_array = features[:, patch.y : patch.y + patch.height, patch.x : patch.x + patch.width]
                if patch.height != patch_size or patch.width != patch_size:
                    padded = np.zeros((features.shape[0], patch_size, patch_size), dtype=np.float32)
                    padded[:, : patch.height, : patch.width] = patch_array
                    patch_array = padded
                batch_inputs_list.append(patch_array)
            batch_inputs = np.stack(batch_inputs_list, axis=0)
            inputs = torch.from_numpy(batch_inputs).float().to(device)

            mode_probabilities: list[torch.Tensor] = []
            for mode in mode_list:
                transformed = inputs if mode == "identity" else _apply_tta(inputs, mode)
                with _amp_context(device, enabled=mixed_precision and device.type == "cuda"):
                    logits = model(transformed)
                probabilities = torch.sigmoid(logits)
                if mode != "identity":
                    probabilities = _invert_tta(probabilities, mode)
                mode_probabilities.append(probabilities)

            probabilities = torch.mean(torch.stack(mode_probabilities, dim=0), dim=0).cpu().numpy()[:, 0]
            for patch, patch_prob in zip(batch_records, probabilities, strict=True):
                valid_prob = patch_prob[: patch.height, : patch.width]
                y_slice = slice(patch.y, patch.y + patch.height)
                x_slice = slice(patch.x, patch.x + patch.width)
                prob_sum[y_slice, x_slice] += valid_prob
                count_sum[y_slice, x_slice] += 1.0

    probability_map = np.divide(prob_sum, np.maximum(count_sum, 1.0))
    probability_map[~valid_mask] = 0.0
    return probability_map.astype(np.float32)
