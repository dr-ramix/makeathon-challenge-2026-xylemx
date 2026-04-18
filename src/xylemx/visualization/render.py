"""Visualization helpers for qualitative inspection of predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _as_hwc_preview(preview: np.ndarray | None) -> np.ndarray | None:
    if preview is None:
        return None
    if preview.ndim == 3 and preview.shape[0] in {1, 3}:
        preview = np.moveaxis(preview, 0, -1)
    if preview.ndim != 3:
        return None
    if preview.dtype != np.uint8:
        preview = np.clip(preview, 0, 255).astype(np.uint8)
    return preview


def _mask_to_bw(mask: np.ndarray) -> np.ndarray:
    return (np.clip(mask.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)


def _colorize_probability(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability.astype(np.float32), 0.0, 1.0)
    rgba = plt.get_cmap("magma")(clipped)
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def _blend_overlay(base: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    base_rgb = _as_hwc_preview(base)
    if base_rgb is None:
        base_rgb = np.stack([_mask_to_bw(mask)] * 3, axis=-1)
    blended = base_rgb.astype(np.float32).copy()
    color_array = np.asarray(color, dtype=np.float32)
    positive = mask.astype(bool)
    blended[positive] = (1.0 - alpha) * blended[positive] + alpha * color_array
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _make_confusion_map(true_mask: np.ndarray, pred_mask: np.ndarray, ignore_mask: np.ndarray | None = None) -> np.ndarray:
    true_binary = true_mask.astype(bool)
    pred_binary = pred_mask.astype(bool)
    ignore_binary = np.zeros_like(true_binary, dtype=bool) if ignore_mask is None else ignore_mask.astype(bool)

    canvas = np.zeros((*true_binary.shape, 3), dtype=np.uint8)
    canvas[..., :] = np.array([20, 20, 20], dtype=np.uint8)
    canvas[ignore_binary] = np.array([70, 70, 70], dtype=np.uint8)
    canvas[true_binary & pred_binary & ~ignore_binary] = np.array([34, 197, 94], dtype=np.uint8)   # TP
    canvas[~true_binary & pred_binary & ~ignore_binary] = np.array([239, 68, 68], dtype=np.uint8)   # FP
    canvas[true_binary & ~pred_binary & ~ignore_binary] = np.array([245, 158, 11], dtype=np.uint8)  # FN
    return canvas


def _compute_metrics(true_mask: np.ndarray, pred_mask: np.ndarray, ignore_mask: np.ndarray | None = None) -> dict[str, float]:
    true_binary = true_mask.astype(bool)
    pred_binary = pred_mask.astype(bool)
    ignore_binary = np.zeros_like(true_binary, dtype=bool) if ignore_mask is None else ignore_mask.astype(bool)
    valid = ~ignore_binary

    tp = float(np.logical_and(true_binary, pred_binary)[valid].sum())
    fp = float(np.logical_and(~true_binary, pred_binary)[valid].sum())
    fn = float(np.logical_and(true_binary, ~pred_binary)[valid].sum())
    denom_iou = tp + fp + fn
    denom_dice = 2.0 * tp + fp + fn
    iou = tp / denom_iou if denom_iou > 0 else 0.0
    dice = (2.0 * tp) / denom_dice if denom_dice > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "iou": iou, "dice": dice}


def save_bw_mask(path: str | Path, mask: np.ndarray) -> None:
    """Save a binary mask as a black-and-white PNG."""

    path = Path(path)
    _ensure_dir(path)
    plt.imsave(path, mask.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)


def save_probability_map(path: str | Path, probability: np.ndarray) -> None:
    """Save a probability map as a higher-contrast color PNG."""

    path = Path(path)
    _ensure_dir(path)
    plt.imsave(path, _colorize_probability(probability))


def save_preview_image(path: str | Path, preview: np.ndarray | None) -> None:
    """Save the input preview as a standalone PNG."""

    preview_rgb = _as_hwc_preview(preview)
    if preview_rgb is None:
        return
    path = Path(path)
    _ensure_dir(path)
    plt.imsave(path, preview_rgb)


def save_overlay_image(
    path: str | Path,
    *,
    preview: np.ndarray | None,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    """Save a colored segmentation overlay on top of the preview."""

    path = Path(path)
    _ensure_dir(path)
    plt.imsave(path, _blend_overlay(preview, mask, color))


def save_confusion_map(
    path: str | Path,
    *,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    ignore_mask: np.ndarray | None = None,
) -> None:
    """Save a TP/FP/FN confusion visualization."""

    path = Path(path)
    _ensure_dir(path)
    plt.imsave(path, _make_confusion_map(true_mask, pred_mask, ignore_mask))


def save_panel(
    path: str | Path,
    *,
    preview: np.ndarray | None,
    true_mask: np.ndarray,
    probability: np.ndarray | None,
    pred_mask: np.ndarray,
    ignore_mask: np.ndarray | None = None,
    dpi: int = 220,
) -> None:
    """Save a professional multi-view qualitative panel."""

    path = Path(path)
    _ensure_dir(path)

    preview_rgb = _as_hwc_preview(preview)
    confusion_map = _make_confusion_map(true_mask, pred_mask, ignore_mask)
    metrics = _compute_metrics(true_mask, pred_mask, ignore_mask)

    images: list[tuple[str, np.ndarray, dict[str, float | str]]] = []
    if preview_rgb is not None:
        images.append(("Original-Look Preview", preview_rgb, {}))
        images.append(("True Overlay", _blend_overlay(preview_rgb, true_mask, (59, 130, 246)), {}))
        images.append(("Prediction Overlay", _blend_overlay(preview_rgb, pred_mask, (236, 72, 153)), {}))
    images.append(("True Mask", true_mask, {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}))
    images.append(("Pred Mask", pred_mask, {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}))
    if probability is not None:
        images.append(("Pred Probability", _colorize_probability(probability), {}))
    images.append(("TP / FP / FN", confusion_map, {}))

    columns = 3
    rows = int(np.ceil(len(images) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.0 * columns, 4.8 * rows), dpi=dpi)
    axes = np.atleast_1d(axes).reshape(rows, columns)

    for axis in axes.flat:
        axis.axis("off")

    for axis, (title, image, kwargs) in zip(axes.flat, images, strict=False):
        axis.imshow(image, **kwargs)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.axis("off")

    stats_text = (
        f"Dice {metrics['dice']:.3f} | IoU {metrics['iou']:.3f} | "
        f"TP {int(metrics['tp'])} | FP {int(metrics['fp'])} | FN {int(metrics['fn'])}"
    )
    figure.suptitle(stats_text, fontsize=14, fontweight="bold")
    figure.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
