"""Sliding-window inference for the temporal model."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from xylemx.data.io import load_json
from xylemx.data.tiling import generate_patch_records
from xylemx.temporal.dataset import ConditionNormalizer, TemporalNormalizer
from xylemx.temporal.io import get_temporal_cond_path, get_temporal_input_path, get_temporal_valid_mask_path


def load_temporal_tile(
    preprocessing_dir: str | Path,
    *,
    split: str,
    tile_id: str,
    clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one cached temporal tensor + condition vector with train-fitted normalization."""

    preprocessing_dir = Path(preprocessing_dir)

    feature_stats = load_json(preprocessing_dir / "normalization_stats.json")
    feature_normalizer = TemporalNormalizer(
        mean=np.asarray(feature_stats["mean"], dtype=np.float32),
        std=np.asarray(feature_stats["std"], dtype=np.float32),
        clip=clip,
    )

    condition_stats_path = preprocessing_dir / "condition_stats.json"
    if condition_stats_path.exists():
        condition_stats = load_json(condition_stats_path)
        cond_normalizer = ConditionNormalizer(
            mean=np.asarray(condition_stats.get("mean", []), dtype=np.float32),
            std=np.asarray(condition_stats.get("std", []), dtype=np.float32),
        )
    else:
        cond_normalizer = ConditionNormalizer(mean=np.zeros((0,), dtype=np.float32), std=np.ones((0,), dtype=np.float32))

    inputs = np.load(get_temporal_input_path(preprocessing_dir, split, tile_id)).astype(np.float32)
    valid_mask = np.load(get_temporal_valid_mask_path(preprocessing_dir, split, tile_id)).astype(bool)

    cond_path = get_temporal_cond_path(preprocessing_dir, split, tile_id)
    if cond_path.exists():
        cond = np.load(cond_path).astype(np.float32).reshape(-1)
    else:
        cond = cond_normalizer.mean.copy()

    normalized_inputs = feature_normalizer(inputs)
    if normalized_inputs.ndim == 4:
        normalized_inputs = normalized_inputs.reshape(-1, normalized_inputs.shape[-2], normalized_inputs.shape[-1])

    normalized_cond = cond_normalizer(cond)
    return normalized_inputs, normalized_cond, valid_mask


def predict_temporal_tile(
    model: torch.nn.Module,
    inputs: np.ndarray,
    cond: np.ndarray,
    valid_mask: np.ndarray,
    *,
    device: torch.device,
    patch_size: int,
    stride: int,
    batch_size: int,
    mixed_precision: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict mask probabilities and time-bin labels for a full tile."""

    patches = generate_patch_records(
        "tile",
        shape=valid_mask.shape,
        patch_size=patch_size,
        stride=stride,
        valid_mask=valid_mask,
        min_valid_fraction=0.0,
    )
    if not patches:
        return np.zeros(valid_mask.shape, dtype=np.float32), np.full(valid_mask.shape, -1, dtype=np.int16)

    prob_sum = np.zeros(valid_mask.shape, dtype=np.float32)
    time_logit_sum: np.ndarray | None = None
    count_sum = np.zeros(valid_mask.shape, dtype=np.float32)

    cond_tensor: torch.Tensor | None = None
    if cond.size > 0:
        cond_tensor = torch.from_numpy(cond.astype(np.float32)).to(device)

    autocast_enabled = mixed_precision and device.type == "cuda"
    with torch.no_grad():
        for start in range(0, len(patches), batch_size):
            batch_records = patches[start : start + batch_size]
            batch_inputs = []
            for patch in batch_records:
                patch_array = inputs[:, patch.y : patch.y + patch.height, patch.x : patch.x + patch.width]
                padded = np.zeros((inputs.shape[0], patch_size, patch_size), dtype=np.float32)
                padded[:, : patch.height, : patch.width] = patch_array
                batch_inputs.append(padded)

            batch = torch.from_numpy(np.stack(batch_inputs, axis=0)).float().to(device)
            if cond_tensor is not None:
                batch_cond = cond_tensor[None, :].repeat(batch.shape[0], 1)
            else:
                batch_cond = None

            autocast = torch.amp.autocast("cuda", enabled=True) if autocast_enabled else nullcontext()
            with autocast:
                outputs = model(batch, batch_cond)

            mask_prob = torch.sigmoid(outputs["mask_logits"]).cpu().numpy()[:, 0]
            time_logits = outputs["time_logits"].cpu().numpy()
            if time_logit_sum is None:
                time_logit_sum = np.zeros((time_logits.shape[1], *valid_mask.shape), dtype=np.float32)

            for patch, patch_prob, patch_time_logits in zip(batch_records, mask_prob, time_logits, strict=True):
                y_slice = slice(patch.y, patch.y + patch.height)
                x_slice = slice(patch.x, patch.x + patch.width)
                prob_sum[y_slice, x_slice] += patch_prob[: patch.height, : patch.width]
                time_logit_sum[:, y_slice, x_slice] += patch_time_logits[:, : patch.height, : patch.width]
                count_sum[y_slice, x_slice] += 1.0

    probability = np.divide(prob_sum, np.maximum(count_sum, 1.0))
    probability[~valid_mask] = 0.0
    averaged_time_logits = np.divide(time_logit_sum, np.maximum(count_sum[None, ...], 1.0))
    time_prediction = np.argmax(averaged_time_logits, axis=0).astype(np.int16)
    time_prediction[~valid_mask] = -1
    return probability.astype(np.float32), time_prediction
