"""Patch datasets for the temporal pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from xylemx.data.io import load_json
from xylemx.data.tiling import PatchRecord, generate_patch_records
from xylemx.temporal.config import TemporalTrainConfig
from xylemx.temporal.io import (
    get_temporal_ignore_mask_path,
    get_temporal_input_path,
    get_temporal_mask_target_path,
    get_temporal_time_target_path,
    get_temporal_valid_mask_path,
    get_temporal_weight_map_path,
)


@dataclass(slots=True)
class TemporalNormalizer:
    """Per-channel z-score normalization for temporal tensors."""

    mean: np.ndarray
    std: np.ndarray
    clip: float

    def __call__(self, array: np.ndarray) -> np.ndarray:
        flattened = array.reshape(-1, array.shape[-2], array.shape[-1]).astype(np.float32)
        normalized = (flattened - self.mean[:, None, None]) / self.std[:, None, None]
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        if self.clip > 0:
            normalized = np.clip(normalized, -self.clip, self.clip)
        return normalized.reshape(array.shape).astype(np.float32)


def _load_normalizer(preprocessing_dir: Path, clip: float) -> TemporalNormalizer:
    stats = load_json(preprocessing_dir / "normalization_stats.json")
    return TemporalNormalizer(
        mean=np.asarray(stats["mean"], dtype=np.float32),
        std=np.asarray(stats["std"], dtype=np.float32),
        clip=clip,
    )


def _spatial_augment(
    inputs: np.ndarray,
    mask_target: np.ndarray,
    time_target: np.ndarray,
    weight_map: np.ndarray,
    ignore_mask: np.ndarray,
    config: TemporalTrainConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if config.horizontal_flip and rng.random() < config.horizontal_flip_p:
        inputs = np.flip(inputs, axis=-1).copy()
        mask_target = np.flip(mask_target, axis=1).copy()
        time_target = np.flip(time_target, axis=1).copy()
        weight_map = np.flip(weight_map, axis=1).copy()
        ignore_mask = np.flip(ignore_mask, axis=1).copy()

    if config.vertical_flip and rng.random() < config.vertical_flip_p:
        inputs = np.flip(inputs, axis=-2).copy()
        mask_target = np.flip(mask_target, axis=0).copy()
        time_target = np.flip(time_target, axis=0).copy()
        weight_map = np.flip(weight_map, axis=0).copy()
        ignore_mask = np.flip(ignore_mask, axis=0).copy()

    if config.rotate90 and rng.random() < config.rotate90_p:
        rotations = int(rng.integers(1, 4))
        inputs = np.rot90(inputs, k=rotations, axes=(-2, -1)).copy()
        mask_target = np.rot90(mask_target, k=rotations).copy()
        time_target = np.rot90(time_target, k=rotations).copy()
        weight_map = np.rot90(weight_map, k=rotations).copy()
        ignore_mask = np.rot90(ignore_mask, k=rotations).copy()

    if config.transpose and rng.random() < config.transpose_p:
        if inputs.ndim == 3:
            inputs = np.transpose(inputs, (0, 2, 1)).copy()
        else:
            inputs = np.transpose(inputs, (0, 1, 3, 2)).copy()
        mask_target = np.transpose(mask_target, (1, 0)).copy()
        time_target = np.transpose(time_target, (1, 0)).copy()
        weight_map = np.transpose(weight_map, (1, 0)).copy()
        ignore_mask = np.transpose(ignore_mask, (1, 0)).copy()

    if config.gaussian_noise and rng.random() < config.gaussian_noise_p:
        inputs = inputs + rng.normal(0.0, config.gaussian_noise_std, size=inputs.shape).astype(np.float32)

    return inputs, mask_target, time_target, weight_map, ignore_mask


class TemporalPatchDataset(Dataset):
    """Patch dataset for temporal segmentation and time-bin training."""

    def __init__(
        self,
        *,
        tile_ids: list[str],
        preprocessing_dir: str | Path,
        split: str,
        patch_size: int,
        stride: int,
        config: TemporalTrainConfig,
        training: bool,
    ) -> None:
        self.tile_ids = tile_ids
        self.preprocessing_dir = Path(preprocessing_dir)
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.config = config
        self.training = training
        self.rng = np.random.default_rng(config.seed if not training else config.seed + 17)
        self.normalizer = _load_normalizer(self.preprocessing_dir, config.normalized_feature_clip)

        self._input_cache: dict[str, np.ndarray] = {}
        self._valid_cache: dict[str, np.ndarray] = {}
        self._mask_cache: dict[str, np.ndarray] = {}
        self._time_cache: dict[str, np.ndarray] = {}
        self._ignore_cache: dict[str, np.ndarray] = {}
        self._weight_cache: dict[str, np.ndarray] = {}

        patch_records: list[PatchRecord] = []
        for tile_id in tile_ids:
            valid_mask = np.load(get_temporal_valid_mask_path(self.preprocessing_dir, split, tile_id)).astype(bool)
            if valid_mask.shape[0] < patch_size or valid_mask.shape[1] < patch_size:
                continue
            target = np.load(get_temporal_mask_target_path(self.preprocessing_dir, tile_id)).astype(np.float32)
            ignore_mask = np.load(get_temporal_ignore_mask_path(self.preprocessing_dir, tile_id)).astype(bool)
            patch_records.extend(
                generate_patch_records(
                    tile_id,
                    shape=valid_mask.shape,
                    patch_size=patch_size,
                    stride=stride,
                    valid_mask=valid_mask,
                    target=target,
                    ignore_mask=ignore_mask,
                    min_valid_fraction=config.min_valid_patch_fraction,
                )
            )

        if training and config.positive_patch_sampling:
            positive = [item for item in patch_records if item.positive_fraction >= config.positive_patch_min_fraction]
            negative = [item for item in patch_records if item.positive_fraction < config.positive_patch_min_fraction]
            if positive and negative:
                target_positive = int(
                    math.ceil(len(negative) * config.positive_patch_ratio / max(1.0 - config.positive_patch_ratio, 1e-6))
                )
                multiplier = max(1, int(math.ceil(target_positive / max(len(positive), 1))))
                patch_records = negative + positive * multiplier
        self.patch_records = patch_records

    def __len__(self) -> int:
        return len(self.patch_records)

    def _load_tile(self, tile_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if tile_id not in self._input_cache:
            self._input_cache[tile_id] = np.load(get_temporal_input_path(self.preprocessing_dir, self.split, tile_id)).astype(np.float32)
            self._valid_cache[tile_id] = np.load(get_temporal_valid_mask_path(self.preprocessing_dir, self.split, tile_id)).astype(bool)
            self._mask_cache[tile_id] = np.load(get_temporal_mask_target_path(self.preprocessing_dir, tile_id)).astype(np.float32)
            self._time_cache[tile_id] = np.load(get_temporal_time_target_path(self.preprocessing_dir, tile_id)).astype(np.int64)
            self._ignore_cache[tile_id] = np.load(get_temporal_ignore_mask_path(self.preprocessing_dir, tile_id)).astype(bool)
            self._weight_cache[tile_id] = np.load(get_temporal_weight_map_path(self.preprocessing_dir, tile_id)).astype(np.float32)
        return (
            self._input_cache[tile_id],
            self._valid_cache[tile_id],
            self._mask_cache[tile_id],
            self._time_cache[tile_id],
            self._ignore_cache[tile_id],
            self._weight_cache[tile_id],
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        patch = self.patch_records[index]
        inputs, valid_mask, mask_target, time_target, ignore_mask, weight_map = self._load_tile(patch.tile_id)
        y_slice = slice(patch.y, patch.y + self.patch_size)
        x_slice = slice(patch.x, patch.x + self.patch_size)

        if inputs.ndim == 4:
            input_patch = inputs[:, :, y_slice, x_slice]
        else:
            input_patch = inputs[:, y_slice, x_slice]
        valid_patch = valid_mask[y_slice, x_slice]
        mask_patch = mask_target[y_slice, x_slice]
        time_patch = time_target[y_slice, x_slice]
        ignore_patch = ignore_mask[y_slice, x_slice] | ~valid_patch
        weight_patch = weight_map[y_slice, x_slice].copy()
        weight_patch[ignore_patch] = 0.0
        time_patch[ignore_patch | (mask_patch < 0.5)] = self.config.time_ignore_index

        if self.training:
            input_patch, mask_patch, time_patch, weight_patch, ignore_patch = _spatial_augment(
                input_patch.copy(),
                mask_patch.copy(),
                time_patch.copy(),
                weight_patch.copy(),
                ignore_patch.copy(),
                self.config,
                self.rng,
            )

        input_patch = self.normalizer(input_patch)

        return {
            "inputs": torch.from_numpy(input_patch).float(),
            "mask_target": torch.from_numpy(mask_patch[None, ...]).float(),
            "time_target": torch.from_numpy(time_patch).long(),
            "ignore_mask": torch.from_numpy(ignore_patch[None, ...]).bool(),
            "weight_map": torch.from_numpy(weight_patch[None, ...]).float(),
            "valid_mask": torch.from_numpy(valid_patch[None, ...]).bool(),
            "tile_id": patch.tile_id,
            "y": patch.y,
            "x": patch.x,
        }
