"""Augmentations for pixel-wise segmentation patches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from xylemx.config import ExperimentConfig

S2_FEATURE_CHANNELS = 36


@dataclass(slots=True)
class SampleAugmentor:
    """Sample-level spatial and channel augmentations."""

    config: ExperimentConfig
    rng: np.random.Generator

    def __call__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        weight_map: np.ndarray,
        ignore_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.config.horizontal_flip and self.rng.random() < self.config.horizontal_flip_p:
            features = np.flip(features, axis=2).copy()
            target = np.flip(target, axis=1).copy()
            weight_map = np.flip(weight_map, axis=1).copy()
            ignore_mask = np.flip(ignore_mask, axis=1).copy()

        if self.config.vertical_flip and self.rng.random() < self.config.vertical_flip_p:
            features = np.flip(features, axis=1).copy()
            target = np.flip(target, axis=0).copy()
            weight_map = np.flip(weight_map, axis=0).copy()
            ignore_mask = np.flip(ignore_mask, axis=0).copy()

        if self.config.rotate90 and self.rng.random() < self.config.rotate90_p:
            rotations = int(self.rng.integers(1, 4))
            features = np.rot90(features, k=rotations, axes=(1, 2)).copy()
            target = np.rot90(target, k=rotations).copy()
            weight_map = np.rot90(weight_map, k=rotations).copy()
            ignore_mask = np.rot90(ignore_mask, k=rotations).copy()

        if self.config.transpose and self.rng.random() < self.config.transpose_p:
            features = np.transpose(features, (0, 2, 1)).copy()
            target = np.transpose(target, (1, 0)).copy()
            weight_map = np.transpose(weight_map, (1, 0)).copy()
            ignore_mask = np.transpose(ignore_mask, (1, 0)).copy()

        if self.config.gaussian_noise and self.rng.random() < self.config.gaussian_noise_p:
            noise = self.rng.normal(0.0, self.config.gaussian_noise_std, size=features.shape).astype(np.float32)
            features = features + noise

        if self.config.s2_brightness_jitter and self.rng.random() < self.config.s2_brightness_jitter_p:
            delta = float(self.rng.uniform(-self.config.s2_brightness_scale, self.config.s2_brightness_scale))
            features[:S2_FEATURE_CHANNELS] *= 1.0 + delta

        if self.config.s2_contrast_jitter and self.rng.random() < self.config.s2_contrast_jitter_p:
            factor = 1.0 + float(self.rng.uniform(-self.config.s2_contrast_scale, self.config.s2_contrast_scale))
            s2_block = features[:S2_FEATURE_CHANNELS]
            center = np.nanmean(s2_block, axis=(1, 2), keepdims=True)
            features[:S2_FEATURE_CHANNELS] = (s2_block - center) * factor + center

        return features, target, weight_map, ignore_mask


def apply_mixup(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight_maps: torch.Tensor,
    ignore_masks: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply mixup to a batch of segmentation patches."""

    if inputs.shape[0] < 2:
        return inputs, targets, weight_maps, ignore_masks

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(inputs.shape[0], device=inputs.device)

    valid_a = (~ignore_masks).float()
    valid_b = (~ignore_masks[perm]).float()
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[perm]
    mixed_weights = lam * weight_maps * valid_a + (1.0 - lam) * weight_maps[perm] * valid_b
    numerator = lam * targets * weight_maps * valid_a + (1.0 - lam) * targets[perm] * weight_maps[perm] * valid_b
    mixed_targets = torch.where(mixed_weights > 0, numerator / torch.clamp(mixed_weights, min=1e-6), torch.zeros_like(targets))
    mixed_ignore = mixed_weights <= 0
    return mixed_inputs, mixed_targets, mixed_weights, mixed_ignore


def apply_cutmix(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight_maps: torch.Tensor,
    ignore_masks: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply CutMix to a batch of segmentation patches."""

    if inputs.shape[0] < 2:
        return inputs, targets, weight_maps, ignore_masks

    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(inputs.shape[0], device=inputs.device)

    _, _, height, width = inputs.shape
    cut_ratio = np.sqrt(max(0.0, 1.0 - lam))
    cut_h = max(1, int(height * cut_ratio))
    cut_w = max(1, int(width * cut_ratio))
    center_y = int(np.random.randint(0, height))
    center_x = int(np.random.randint(0, width))

    y1 = max(0, center_y - cut_h // 2)
    y2 = min(height, center_y + cut_h // 2)
    x1 = max(0, center_x - cut_w // 2)
    x2 = min(width, center_x + cut_w // 2)

    mixed_inputs = inputs.clone()
    mixed_targets = targets.clone()
    mixed_weights = weight_maps.clone()
    mixed_ignore = ignore_masks.clone()

    mixed_inputs[:, :, y1:y2, x1:x2] = inputs[perm, :, y1:y2, x1:x2]
    mixed_targets[:, :, y1:y2, x1:x2] = targets[perm, :, y1:y2, x1:x2]
    mixed_weights[:, :, y1:y2, x1:x2] = weight_maps[perm, :, y1:y2, x1:x2]
    mixed_ignore[:, :, y1:y2, x1:x2] = ignore_masks[perm, :, y1:y2, x1:x2]
    return mixed_inputs, mixed_targets, mixed_weights, mixed_ignore
