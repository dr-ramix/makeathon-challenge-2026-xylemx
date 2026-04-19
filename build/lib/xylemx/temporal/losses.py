"""Temporal multi-task losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from xylemx.models.losses import build_loss


class TemporalMultiTaskLoss(nn.Module):
    """Mask loss plus time-bin cross-entropy over positive pixels only."""

    def __init__(
        self,
        *,
        mask_loss_name: str,
        bce_weight: float,
        dice_weight: float,
        pos_weight: float,
        lambda_time: float,
        time_loss_weight: float,
        time_ignore_index: int,
    ) -> None:
        super().__init__()
        self.mask_loss = build_loss(
            mask_loss_name,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            pos_weight=pos_weight,
        )
        self.lambda_time = lambda_time
        self.time_loss_weight = time_loss_weight
        self.time_ignore_index = time_ignore_index

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        mask_target: torch.Tensor,
        time_target: torch.Tensor,
        *,
        ignore_mask: torch.Tensor | None = None,
        weight_map: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask_logits = outputs["mask_logits"]
        time_logits = outputs["time_logits"]
        mask_loss = self.mask_loss(mask_logits, mask_target, ignore_mask=ignore_mask, weight_map=weight_map)

        effective_target = time_target.clone()
        if ignore_mask is not None:
            effective_target = effective_target.masked_fill(ignore_mask[:, 0], self.time_ignore_index)
        valid_time = effective_target != self.time_ignore_index
        if valid_time.any():
            time_loss = F.cross_entropy(
                time_logits,
                effective_target,
                ignore_index=self.time_ignore_index,
                reduction="none",
            )
            if weight_map is not None:
                weights = weight_map[:, 0].float()
                time_loss = (time_loss * weights * valid_time.float()).sum() / torch.clamp(
                    (weights * valid_time.float()).sum(),
                    min=1e-6,
                )
            else:
                time_loss = time_loss[valid_time].mean()
        else:
            time_loss = mask_logits.new_zeros(())

        total = mask_loss + self.lambda_time * self.time_loss_weight * time_loss
        return {
            "loss": total,
            "mask_loss": mask_loss,
            "time_loss": time_loss,
        }
