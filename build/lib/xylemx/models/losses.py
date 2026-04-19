"""Segmentation losses with ignore masks and pixel weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def _prepare_targets(targets: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    if label_smoothing <= 0:
        return targets
    return targets * (1.0 - label_smoothing) + 0.5 * label_smoothing


def _valid_weight_map(
    targets: torch.Tensor,
    ignore_mask: torch.Tensor | None,
    weight_map: torch.Tensor | None,
) -> torch.Tensor:
    valid = torch.ones_like(targets, dtype=torch.bool) if ignore_mask is None else ~ignore_mask.bool()
    weights = torch.ones_like(targets, dtype=torch.float32) if weight_map is None else weight_map.float()
    return weights * valid.float()


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_mask: torch.Tensor | None = None,
    weight_map: torch.Tensor | None = None,
    pos_weight: float = 1.0,
    label_smoothing: float = 0.0,
    hard_negative_mining: bool = False,
    hard_negative_ratio: float = 3.0,
) -> torch.Tensor:
    """Weighted BCE loss over non-ignored pixels."""

    prepared_targets = _prepare_targets(targets, label_smoothing)
    weights = _valid_weight_map(targets, ignore_mask, weight_map)
    losses = F.binary_cross_entropy_with_logits(
        logits,
        prepared_targets,
        reduction="none",
        pos_weight=torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device),
    )

    if hard_negative_mining:
        positives = targets >= 0.5
        negatives = ~positives
        positive_count = int((weights * positives.float() > 0).sum().item())
        negative_keep = max(positive_count * int(hard_negative_ratio), 1)
        negative_losses = losses[negatives & (weights > 0)]
        if negative_losses.numel() > negative_keep:
            threshold = torch.topk(negative_losses, k=negative_keep).values.min()
            keep_mask = positives | (losses >= threshold)
            weights = weights * keep_mask.float()

    weighted = losses * weights
    denom = torch.clamp(weights.sum(), min=1e-6)
    return weighted.sum() / denom


def weighted_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_mask: torch.Tensor | None = None,
    weight_map: torch.Tensor | None = None,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft dice loss with pixel weights."""

    weights = _valid_weight_map(targets, ignore_mask, weight_map)
    probabilities = torch.sigmoid(logits)
    numerator = 2.0 * torch.sum(probabilities * targets * weights)
    denominator = torch.sum(probabilities * weights) + torch.sum(targets * weights)
    score = (numerator + smooth) / (denominator + smooth)
    return 1.0 - score


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_mask: torch.Tensor | None = None,
    weight_map: torch.Tensor | None = None,
    gamma: float = 2.0,
    pos_weight: float = 1.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Binary focal loss with optional pixel weights."""

    prepared_targets = _prepare_targets(targets, label_smoothing)
    weights = _valid_weight_map(targets, ignore_mask, weight_map)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        prepared_targets,
        reduction="none",
        pos_weight=torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device),
    )
    probabilities = torch.sigmoid(logits)
    pt = torch.where(targets >= 0.5, probabilities, 1.0 - probabilities)
    focal = torch.pow(1.0 - pt, gamma) * bce
    denom = torch.clamp(weights.sum(), min=1e-6)
    return torch.sum(focal * weights) / denom


@dataclass(slots=True)
class LossConfig:
    name: str
    bce_weight: float = 0.7
    dice_weight: float = 0.3
    pos_weight: float = 1.0
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    hard_negative_mining: bool = False
    hard_negative_ratio: float = 3.0


class SegmentationLoss(nn.Module):
    """Composable loss wrapper used by the trainer."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        if self.config.name not in {"bce", "dice", "bce_dice", "focal"}:
            raise ValueError(f"Unsupported loss: {self.config.name}")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        ignore_mask: torch.Tensor | None = None,
        weight_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.config.name == "bce":
            return weighted_bce_with_logits(
                logits,
                targets,
                ignore_mask=ignore_mask,
                weight_map=weight_map,
                pos_weight=self.config.pos_weight,
                label_smoothing=self.config.label_smoothing,
                hard_negative_mining=self.config.hard_negative_mining,
                hard_negative_ratio=self.config.hard_negative_ratio,
            )
        if self.config.name == "dice":
            return weighted_dice_loss(logits, targets, ignore_mask=ignore_mask, weight_map=weight_map)
        if self.config.name == "focal":
            return focal_loss_with_logits(
                logits,
                targets,
                ignore_mask=ignore_mask,
                weight_map=weight_map,
                gamma=self.config.focal_gamma,
                pos_weight=self.config.pos_weight,
                label_smoothing=self.config.label_smoothing,
            )

        bce = weighted_bce_with_logits(
            logits,
            targets,
            ignore_mask=ignore_mask,
            weight_map=weight_map,
            pos_weight=self.config.pos_weight,
            label_smoothing=self.config.label_smoothing,
            hard_negative_mining=self.config.hard_negative_mining,
            hard_negative_ratio=self.config.hard_negative_ratio,
        )
        dice = weighted_dice_loss(logits, targets, ignore_mask=ignore_mask, weight_map=weight_map)
        return self.config.bce_weight * bce + self.config.dice_weight * dice


def build_loss(
    name: str,
    *,
    bce_weight: float = 0.7,
    dice_weight: float = 0.3,
    pos_weight: float = 1.0,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    hard_negative_mining: bool = False,
    hard_negative_ratio: float = 3.0,
) -> SegmentationLoss:
    """Factory for supported segmentation losses."""

    return SegmentationLoss(
        LossConfig(
            name=name.lower(),
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            hard_negative_mining=hard_negative_mining,
            hard_negative_ratio=hard_negative_ratio,
        )
    )
