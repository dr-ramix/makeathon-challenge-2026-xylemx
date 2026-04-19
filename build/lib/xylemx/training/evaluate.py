"""Validation helpers."""

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from xylemx.training.metrics import add_confusion_counts, compute_confusion_counts, empty_confusion_counts, metrics_from_confusion_counts


def _amp_context(device: torch.device, enabled: bool):
    """Create an autocast context only when it is supported and useful."""

    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return nullcontext()


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn,
    *,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    """Run model evaluation on one dataloader."""

    model.eval()
    losses: list[float] = []
    confusion = empty_confusion_counts()

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            targets = batch["target"].to(device)
            ignore_mask = batch["ignore_mask"].to(device)

            amp_context = _amp_context(device, enabled=use_amp and device.type == "cuda")
            with amp_context:
                logits = model(inputs)
                loss = loss_fn(logits, targets, ignore_mask=ignore_mask)

            losses.append(float(loss.item()))
            confusion = add_confusion_counts(
                confusion,
                compute_confusion_counts(logits, targets, ignore_mask=ignore_mask),
            )

    aggregate = metrics_from_confusion_counts(confusion)
    aggregate["loss"] = float(sum(losses) / max(len(losses), 1))
    aggregate["valid_pixels"] = float(confusion["valid_pixels"])
    return aggregate
