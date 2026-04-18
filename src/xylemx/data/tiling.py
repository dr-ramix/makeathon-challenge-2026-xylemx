"""Patch generation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PatchRecord:
    """Location and quality metadata for one tile patch."""

    tile_id: str
    y: int
    x: int
    height: int
    width: int
    valid_fraction: float
    positive_fraction: float


def _window_positions(length: int, patch_size: int, stride: int) -> list[int]:
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive integers")
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    if positions[-1] != length - patch_size:
        positions.append(length - patch_size)
    return positions


def generate_patch_records(
    tile_id: str,
    shape: tuple[int, int],
    *,
    patch_size: int,
    stride: int,
    valid_mask: np.ndarray | None = None,
    target: np.ndarray | None = None,
    ignore_mask: np.ndarray | None = None,
    min_valid_fraction: float = 0.05,
) -> list[PatchRecord]:
    """Generate patch metadata for a tile."""

    height, width = shape
    rows = _window_positions(height, patch_size, stride)
    cols = _window_positions(width, patch_size, stride)
    valid_mask = np.ones(shape, dtype=bool) if valid_mask is None else valid_mask.astype(bool)
    ignore_mask = np.zeros(shape, dtype=bool) if ignore_mask is None else ignore_mask.astype(bool)

    records: list[PatchRecord] = []
    for y in rows:
        for x in cols:
            valid_patch = valid_mask[y : y + patch_size, x : x + patch_size]
            valid_fraction = float(valid_patch.mean())
            if valid_fraction < min_valid_fraction:
                continue

            positive_fraction = 0.0
            if target is not None:
                label_patch = target[y : y + patch_size, x : x + patch_size]
                ignore_patch = ignore_mask[y : y + patch_size, x : x + patch_size]
                usable = valid_patch & ~ignore_patch
                if usable.any():
                    positive_fraction = float(label_patch[usable].mean())

            records.append(
                PatchRecord(
                    tile_id=tile_id,
                    y=y,
                    x=x,
                    height=min(patch_size, height - y),
                    width=min(patch_size, width - x),
                    valid_fraction=valid_fraction,
                    positive_fraction=positive_fraction,
                )
            )
    return records
