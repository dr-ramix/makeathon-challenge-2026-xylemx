"""Visualization helpers for black-and-white mask inspection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_bw_mask(path: str | Path, mask: np.ndarray) -> None:
    """Save a binary mask as a black-and-white PNG."""

    path = Path(path)
    _ensure_dir(path)
    plt.imsave(path, mask.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)


def save_probability_map(path: str | Path, probability: np.ndarray) -> None:
    """Save a probability map as a grayscale PNG."""

    path = Path(path)
    _ensure_dir(path)
    plt.imsave(path, probability.astype(np.float32), cmap="gray", vmin=0.0, vmax=1.0)


def save_panel(
    path: str | Path,
    *,
    preview: np.ndarray | None,
    true_mask: np.ndarray,
    probability: np.ndarray | None,
    pred_mask: np.ndarray,
) -> None:
    """Save a side-by-side panel for one sample."""

    path = Path(path)
    _ensure_dir(path)

    images: list[tuple[str, np.ndarray, dict[str, float | str]]] = []
    if preview is not None:
        images.append(("Input Preview", np.moveaxis(preview, 0, -1), {}))
    images.append(("True Mask", true_mask, {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}))
    if probability is not None:
        images.append(("Pred Prob", probability, {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}))
    images.append(("Pred Mask", pred_mask, {"cmap": "gray", "vmin": 0.0, "vmax": 1.0}))

    figure, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    axes = np.atleast_1d(axes)
    for axis, (title, image, kwargs) in zip(axes, images, strict=True):
        axis.imshow(image, **kwargs)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
