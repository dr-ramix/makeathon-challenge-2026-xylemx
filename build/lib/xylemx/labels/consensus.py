"""Weak-label fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(slots=True)
class LabelFusionResult:
    """Final fused supervision for one tile."""

    target: np.ndarray
    soft_target: np.ndarray
    weight_map: np.ndarray
    ignore_mask: np.ndarray
    vote_count: np.ndarray
    valid_extent: np.ndarray
    available_sources: np.ndarray


def radd_positive_mask(raw: np.ndarray, mode: str = "permissive") -> np.ndarray:
    """Convert RADD labels into a binary positive mask."""

    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint16, copy=False)
    if mode == "permissive":
        return raw > 0
    if mode == "conservative":
        return (raw // 10000) >= 3
    raise ValueError(f"Unsupported RADD positive mode: {mode}")


def gladl_positive_mask(alerts_by_year: list[np.ndarray], threshold: int = 2) -> np.ndarray:
    """Merge yearly GLAD-L rasters into one binary mask."""

    if not alerts_by_year:
        raise ValueError("GLAD-L requires at least one yearly alert raster")
    positive = np.zeros_like(alerts_by_year[0], dtype=bool)
    for alert in alerts_by_year:
        clean = np.nan_to_num(alert, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int16, copy=False)
        positive |= clean >= threshold
    return positive


def glads2_positive_mask(alert: np.ndarray, threshold: int = 1) -> np.ndarray:
    """Convert GLAD-S2 alerts into a binary mask."""

    clean = np.nan_to_num(alert, nan=0.0, posinf=0.0, neginf=0.0).astype(np.int16, copy=False)
    return clean >= threshold


def fuse_binary_masks(
    source_masks: Mapping[str, np.ndarray],
    source_valid_masks: Mapping[str, np.ndarray],
    *,
    method: str = "consensus_2of3",
    soft_vote_threshold: float = 0.5,
    ignore_uncertain_single_source_positives: bool = False,
    ignore_outside_label_extent: bool = True,
    vote_weight_0: float = 1.0,
    vote_weight_1: float = 0.3,
    vote_weight_2: float = 0.8,
    vote_weight_3: float = 1.0,
) -> LabelFusionResult:
    """Fuse per-source binary weak labels into a final target and weight map."""

    if not source_masks:
        raise ValueError("source_masks must not be empty")
    if source_masks.keys() != source_valid_masks.keys():
        raise ValueError("source_masks and source_valid_masks must share the same keys")

    first_shape = next(iter(source_masks.values())).shape
    stack = []
    valid_stack = []
    for key in source_masks:
        mask = source_masks[key].astype(bool)
        valid = source_valid_masks[key].astype(bool)
        if mask.shape != first_shape or valid.shape != first_shape:
            raise ValueError("All source masks and valid extents must share the same shape")
        stack.append(mask)
        valid_stack.append(valid)

    source_stack = np.stack(stack, axis=0)
    valid_stack_array = np.stack(valid_stack, axis=0)
    vote_count = source_stack.sum(axis=0).astype(np.uint8)
    available_sources = valid_stack_array.sum(axis=0).astype(np.uint8)
    valid_extent = available_sources > 0

    denom = np.maximum(available_sources.astype(np.float32), 1.0)
    soft_target = vote_count.astype(np.float32) / denom

    method = method.lower()
    if method == "consensus_2of3":
        target = vote_count >= 2
    elif method == "union":
        target = vote_count >= 1
    elif method == "unanimous":
        target = vote_count == len(source_masks)
    elif method == "soft_vote":
        target = soft_target >= soft_vote_threshold
    else:
        raise ValueError(f"Unsupported label fusion method: {method}")

    weight_map = np.full(first_shape, vote_weight_0, dtype=np.float32)
    weight_map[vote_count == 1] = vote_weight_1
    weight_map[vote_count == 2] = vote_weight_2
    weight_map[vote_count >= 3] = vote_weight_3

    ignore_mask = np.zeros(first_shape, dtype=bool)
    if ignore_outside_label_extent:
        ignore_mask |= ~valid_extent
    if ignore_uncertain_single_source_positives:
        ignore_mask |= vote_count == 1

    weight_map[ignore_mask] = 0.0
    return LabelFusionResult(
        target=target.astype(np.float32),
        soft_target=soft_target.astype(np.float32),
        weight_map=weight_map,
        ignore_mask=ignore_mask,
        vote_count=vote_count,
        valid_extent=valid_extent,
        available_sources=available_sources,
    )
