"""Train/validation splitting helpers."""

from __future__ import annotations

import random
from typing import Mapping

import numpy as np


def split_train_val_tiles(
    tile_ids: list[str],
    *,
    val_ratio: float,
    seed: int,
    positive_fractions: Mapping[str, float] | None = None,
    stratify: bool = True,
) -> tuple[list[str], list[str]]:
    """Split tiles deterministically, optionally using coarse stratification."""

    if not tile_ids:
        raise ValueError("No tile ids available for splitting")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    target_val = max(1, int(round(len(tile_ids) * val_ratio)))
    rng = random.Random(seed)

    if not stratify or positive_fractions is None or len(tile_ids) < 6:
        shuffled = list(tile_ids)
        rng.shuffle(shuffled)
        val_tiles = sorted(shuffled[:target_val])
        train_tiles = sorted(shuffled[target_val:])
        return train_tiles, val_tiles

    fractions = np.asarray([float(positive_fractions.get(tile_id, 0.0)) for tile_id in tile_ids], dtype=np.float32)
    if np.allclose(fractions, fractions[0]):
        shuffled = list(tile_ids)
        rng.shuffle(shuffled)
        val_tiles = sorted(shuffled[:target_val])
        train_tiles = sorted(shuffled[target_val:])
        return train_tiles, val_tiles

    quantiles = np.quantile(fractions, [0.0, 0.33, 0.66, 1.0])
    bins = np.digitize(fractions, quantiles[1:-1], right=True)

    grouped: dict[int, list[str]] = {}
    for tile_id, bin_id in zip(tile_ids, bins, strict=True):
        grouped.setdefault(int(bin_id), []).append(tile_id)

    val_tiles: list[str] = []
    remaining: list[str] = []
    for group_tiles in grouped.values():
        group_tiles = sorted(group_tiles)
        rng.shuffle(group_tiles)
        group_val = min(len(group_tiles), max(1, int(round(len(group_tiles) * val_ratio))))
        val_tiles.extend(group_tiles[:group_val])
        remaining.extend(group_tiles[group_val:])

    val_tiles = sorted(set(val_tiles))
    if len(val_tiles) > target_val:
        rng.shuffle(val_tiles)
        spill = sorted(val_tiles[target_val:])
        remaining.extend(spill)
        val_tiles = sorted(val_tiles[:target_val])
    elif len(val_tiles) < target_val:
        remaining = sorted(set(remaining) - set(val_tiles))
        rng.shuffle(remaining)
        val_tiles = sorted(val_tiles + remaining[: target_val - len(val_tiles)])

    train_tiles = sorted(set(tile_ids) - set(val_tiles))
    return train_tiles, sorted(val_tiles)
