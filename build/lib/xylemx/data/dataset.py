"""Patch datasets backed by cached preprocessing artifacts."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from xylemx.config import ExperimentConfig
from xylemx.data.io import (
    get_feature_path,
    get_ignore_mask_path,
    get_target_path,
    get_valid_mask_path,
    get_weight_map_path,
    load_json,
)
from xylemx.data.tiling import PatchRecord, generate_patch_records
from xylemx.preprocessing.normalize import normalize_array
from xylemx.training.augmentations import SampleAugmentor

LOGGER = logging.getLogger(__name__)


class SegmentationPatchDataset(Dataset):
    """Patch dataset for training and validation."""

    def __init__(
        self,
        *,
        tile_ids: list[str],
        preprocessing_dir: str | Path,
        split: str,
        patch_size: int,
        stride: int,
        config: ExperimentConfig,
        training: bool,
    ) -> None:
        self.tile_ids = tile_ids
        self.preprocessing_dir = Path(preprocessing_dir)
        self.split = split
        self.patch_size = patch_size
        self.stride = stride
        self.config = config
        self.training = training
        self.normalization_stats = load_json(self.preprocessing_dir / "normalization_stats.json")
        self.augmentor = SampleAugmentor(config=config, rng=np.random.default_rng(config.seed)) if training else None

        self._feature_cache: dict[str, np.ndarray] = {}
        self._valid_mask_cache: dict[str, np.ndarray] = {}
        self._target_cache: dict[str, np.ndarray] = {}
        self._ignore_cache: dict[str, np.ndarray] = {}
        self._weight_cache: dict[str, np.ndarray] = {}

        patch_records: list[PatchRecord] = []
        for tile_id in tile_ids:
            valid_mask = np.load(get_valid_mask_path(self.preprocessing_dir, split, tile_id)).astype(bool)
            if valid_mask.shape[0] < patch_size or valid_mask.shape[1] < patch_size:
                LOGGER.warning("Skipping tile %s because it is smaller than patch_size=%d", tile_id, patch_size)
                continue

            target = np.load(get_target_path(self.preprocessing_dir, tile_id)).astype(np.float32)
            ignore_mask = np.load(get_ignore_mask_path(self.preprocessing_dir, tile_id)).astype(bool)
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

        if not patch_records:
            raise ValueError("No patch records were generated for the requested dataset")

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

    def _load_tile(self, tile_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if tile_id not in self._feature_cache:
            self._feature_cache[tile_id] = np.load(get_feature_path(self.preprocessing_dir, self.split, tile_id)).astype(np.float32)
            self._valid_mask_cache[tile_id] = np.load(get_valid_mask_path(self.preprocessing_dir, self.split, tile_id)).astype(bool)
            self._target_cache[tile_id] = np.load(get_target_path(self.preprocessing_dir, tile_id)).astype(np.float32)
            self._ignore_cache[tile_id] = np.load(get_ignore_mask_path(self.preprocessing_dir, tile_id)).astype(bool)
            self._weight_cache[tile_id] = np.load(get_weight_map_path(self.preprocessing_dir, tile_id)).astype(np.float32)
        return (
            self._feature_cache[tile_id],
            self._valid_mask_cache[tile_id],
            self._target_cache[tile_id],
            self._ignore_cache[tile_id],
            self._weight_cache[tile_id],
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        patch = self.patch_records[index]
        features, valid_mask, target, ignore_mask, weight_map = self._load_tile(patch.tile_id)
        y_slice = slice(patch.y, patch.y + self.patch_size)
        x_slice = slice(patch.x, patch.x + self.patch_size)

        feature_patch = features[:, y_slice, x_slice]
        valid_patch = valid_mask[y_slice, x_slice]
        target_patch = target[y_slice, x_slice]
        ignore_patch = ignore_mask[y_slice, x_slice] | ~valid_patch
        weight_patch = weight_map[y_slice, x_slice].copy()
        weight_patch[ignore_patch] = 0.0

        if self.training and self.augmentor is not None:
            feature_patch, target_patch, weight_patch, ignore_patch = self.augmentor(
                feature_patch.copy(),
                target_patch.copy(),
                weight_patch.copy(),
                ignore_patch.copy(),
            )

        feature_patch = normalize_array(
            feature_patch,
            self.normalization_stats,
            normalization=self.config.normalization,
        )
        if self.config.normalized_feature_clip > 0:
            feature_patch = np.clip(
                feature_patch,
                -self.config.normalized_feature_clip,
                self.config.normalized_feature_clip,
            )
        feature_patch = np.nan_to_num(feature_patch, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "inputs": torch.from_numpy(feature_patch).float(),
            "target": torch.from_numpy(target_patch[None, ...]).float(),
            "ignore_mask": torch.from_numpy(ignore_patch[None, ...]).bool(),
            "weight_map": torch.from_numpy(weight_patch[None, ...]).float(),
            "valid_mask": torch.from_numpy(valid_patch[None, ...]).bool(),
            "tile_id": patch.tile_id,
            "y": patch.y,
            "x": patch.x,
        }
