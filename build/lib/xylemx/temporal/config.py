"""Configuration helpers for the temporal pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass(slots=True)
class TemporalPreprocessingConfig:
    """Flat preprocessing config for temporal artifact generation."""

    data_root: str = "data/makeathon-challenge"
    seed: int = 42
    val_ratio: float = 0.2
    split_seed: int = 42
    stratify_split: bool = True

    time_start: str = "2020-01"
    time_end: str = "2025-12"
    time_step_month_stride: int = 1
    time_bin_mode: str = "year"
    time_merge_strategy: str = "highest_confidence"
    representation: str = "sequence"
    flatten_time: bool = False
    summary_window_count: int = 3

    include_sentinel2: bool = True
    include_sentinel1: bool = True
    include_aef: bool = True
    add_s2_indices: bool = True
    add_validity_channels: bool = True
    aef_pca_dim: int = 8
    pca_num_samples_per_raster: int = 2000

    patch_size: int = 128
    patch_stride: int = 128
    min_valid_patch_fraction: float = 0.05

    min_label_confidence: float = 0.45
    label_fusion: str = "consensus_2of3"
    radd_positive_mode: str = "permissive"
    gladl_threshold: int = 2
    glads2_threshold: int = 2
    soft_vote_threshold: float = 0.5
    ignore_uncertain_single_source_positives: bool = False
    ignore_outside_label_extent: bool = True
    vote_weight_3: float = 1.0
    vote_weight_2: float = 0.8
    vote_weight_1: float = 0.3
    vote_weight_0: float = 1.0
    time_ignore_index: int = -1
    min_normalization_std: float = 1e-2


@dataclass(slots=True)
class TemporalTrainConfig:
    """Train-time config for the temporal model."""

    preprocessing_dir: str = "output/preprocessing_temporal"
    data_root: str = "data/makeathon-challenge"
    output_root: str = "output/temporal_runs"
    run_name: str = ""
    short_tag: str = ""
    seed: int = 42
    num_workers: int = 4
    device: str = "auto"

    model: str = "temporal_unet"
    patch_size: int = 128
    train_stride: int = 128
    eval_stride: int = 128
    batch_size: int = 4
    epochs: int = 30
    lr: float = 3e-4
    min_lr: float = 1e-6
    warmup_epochs: int = 2
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True

    stem_channels: int = 32
    base_channels: int = 32
    dropout: float = 0.1
    temporal_kernel_size: int = 3

    mask_loss: str = "bce_dice"
    bce_weight: float = 0.7
    dice_weight: float = 0.3
    pos_weight: float = 1.0
    lambda_time: float = 0.5
    time_loss_weight: float = 1.0
    time_ignore_index: int = -1

    positive_patch_sampling: bool = True
    positive_patch_ratio: float = 0.5
    positive_patch_min_fraction: float = 0.001
    min_valid_patch_fraction: float = 0.05

    horizontal_flip: bool = True
    horizontal_flip_p: float = 0.5
    vertical_flip: bool = True
    vertical_flip_p: float = 0.5
    rotate90: bool = True
    rotate90_p: float = 0.75
    transpose: bool = False
    transpose_p: float = 0.2
    gaussian_noise: bool = False
    gaussian_noise_p: float = 0.0
    gaussian_noise_std: float = 0.01
    normalized_feature_clip: float = 10.0

    inference_threshold: float = 0.5
    export_splits: list[str] = field(default_factory=lambda: ["val"])
    save_probability_rasters: bool = False
    save_time_rasters: bool = True
    create_submission_geojson: bool = False
    polygon_time_strategy: str = "majority"


def _unwrap_optional(field_type: Any) -> Any:
    origin = get_origin(field_type)
    if origin is None:
        return field_type
    args = [arg for arg in get_args(field_type) if arg is not type(None)]
    return args[0] if len(args) == 1 else field_type


def _coerce_value(field_type: Any, value: Any) -> Any:
    field_type = _unwrap_optional(field_type)
    origin = get_origin(field_type)

    if field_type is bool:
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value: {value}")
    if field_type is int:
        return int(value)
    if field_type is float:
        return float(value)
    if field_type is str:
        return str(value)
    if field_type is Path:
        return Path(value)
    if origin is list:
        item_type = get_args(field_type)[0]
        if isinstance(value, list):
            return [_coerce_value(item_type, item) for item in value]
        parsed = yaml.safe_load(value) if isinstance(value, str) else value
        if isinstance(parsed, list):
            return [_coerce_value(item_type, item) for item in parsed]
        if isinstance(parsed, str):
            return [_coerce_value(item_type, item.strip()) for item in parsed.split(",") if item.strip()]
        raise ValueError(f"Cannot parse list value: {value}")
    return value


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return payload


def parse_temporal_cli_overrides(
    argv: list[str],
    *,
    config_cls: type[TemporalPreprocessingConfig] | type[TemporalTrainConfig],
) -> TemporalPreprocessingConfig | TemporalTrainConfig:
    """Parse ``key=value`` CLI overrides into one temporal config dataclass."""

    values = asdict(config_cls())
    hints = get_type_hints(config_cls)

    for token in argv:
        if "=" not in token:
            raise ValueError(f"Expected key=value argument, got: {token}")
        key, raw_value = token.split("=", 1)
        if key in {"config", "config_path"}:
            loaded = _load_yaml_config(raw_value)
            for loaded_key, loaded_value in loaded.items():
                if loaded_key not in values:
                    raise KeyError(f"Unknown config key in {raw_value}: {loaded_key}")
                values[loaded_key] = _coerce_value(hints[loaded_key], loaded_value)
            continue
        if key not in values:
            raise KeyError(f"Unknown config argument: {key}")
        parsed = yaml.safe_load(raw_value)
        values[key] = _coerce_value(hints[key], parsed)

    return config_cls(**values)
