"""Evaluation metrics for binary segmentation."""

from __future__ import annotations

from typing import TypedDict

import torch


class ConfusionCounts(TypedDict):
    """Confusion-matrix counts accumulated over valid pixels."""

    tp: int
    fp: int
    fn: int
    tn: int
    valid_pixels: int


def _flatten_valid(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    if ignore_mask is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        valid = ~ignore_mask.bool()
    return predictions[valid], targets[valid] >= 0.5


def empty_confusion_counts() -> ConfusionCounts:
    """Return a zeroed confusion-count structure."""

    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "valid_pixels": 0}


def compute_confusion_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> ConfusionCounts:
    """Compute confusion counts for one batch of binary segmentation predictions."""

    predictions, target_mask = _flatten_valid(logits, targets, ignore_mask=ignore_mask, threshold=threshold)
    if predictions.numel() == 0:
        return empty_confusion_counts()

    predictions = predictions.to(torch.bool)
    target_mask = target_mask.to(torch.bool)

    tp = int(torch.sum(predictions & target_mask).item())
    fp = int(torch.sum(predictions & ~target_mask).item())
    fn = int(torch.sum(~predictions & target_mask).item())
    tn = int(torch.sum(~predictions & ~target_mask).item())
    valid_pixels = int(predictions.numel())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "valid_pixels": valid_pixels}


def add_confusion_counts(total: ConfusionCounts, batch: ConfusionCounts) -> ConfusionCounts:
    """Add one set of confusion counts into an accumulator."""

    return {
        "tp": total["tp"] + batch["tp"],
        "fp": total["fp"] + batch["fp"],
        "fn": total["fn"] + batch["fn"],
        "tn": total["tn"] + batch["tn"],
        "valid_pixels": total["valid_pixels"] + batch["valid_pixels"],
    }


def metrics_from_confusion_counts(counts: ConfusionCounts) -> dict[str, float]:
    """Convert global confusion counts into task-aligned segmentation metrics."""

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    valid_pixels = counts["valid_pixels"]

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    iou = float(tp / max(tp + fp + fn, 1))
    dice = float((2 * tp) / max(2 * tp + fp + fn, 1))
    accuracy = float((tp + tn) / max(valid_pixels, 1))

    return {
        "iou": iou,
        "dice": dice,
        "f1": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def compute_binary_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_mask: torch.Tensor | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute binary-segmentation metrics from logits using global-count formulas."""

    counts = compute_confusion_counts(logits, targets, ignore_mask=ignore_mask, threshold=threshold)
    return metrics_from_confusion_counts(counts)
