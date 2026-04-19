"""Path helpers for temporal preprocessing artifacts."""

from __future__ import annotations

from pathlib import Path


def get_temporal_input_path(preprocessing_dir: str | Path, split: str, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "inputs" / split / f"{tile_id}.npy"


def get_temporal_valid_mask_path(preprocessing_dir: str | Path, split: str, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "valid_masks" / split / f"{tile_id}.npy"


def get_temporal_cond_path(preprocessing_dir: str | Path, split: str, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "conditions" / split / f"{tile_id}.npy"


def get_temporal_mask_target_path(preprocessing_dir: str | Path, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "targets" / "mask" / f"{tile_id}.npy"


def get_temporal_time_target_path(preprocessing_dir: str | Path, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "targets" / "time" / f"{tile_id}.npy"


def get_temporal_ignore_mask_path(preprocessing_dir: str | Path, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "ignore_masks" / f"{tile_id}.npy"


def get_temporal_weight_map_path(preprocessing_dir: str | Path, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "weight_maps" / f"{tile_id}.npy"
