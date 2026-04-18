"""Configuration parsing for the training pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    """Flat, CLI-friendly configuration for training and inference."""

    data_root: str = "data/makeathon-challenge"
    output_root: str = "output/training_runs"
    preprocessing_dir: str = ""
    reuse_preprocessing: bool = True
    preprocessing_num_workers: int = 4
    run_name: str = ""
    short_tag: str = ""
    seed: int = 42
    num_workers: int = 4
    device: str = "auto"

    val_ratio: float = 0.2
    split_seed: int = 42
    stratify_split: bool = True

    model: str = "resnet18_unet"
    encoder_pretrained: bool = False
    patch_size: int = 128
    train_stride: int = 128
    eval_stride: int = 128
    batch_size: int = 4
    epochs: int = 40
    optimizer: str = "adamw"
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 2
    dropout: float = 0.1
    stochastic_depth: float = 0.0
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True
    ema: bool = False
    ema_decay: float = 0.999
    early_stopping_patience: int = 10

    loss: str = "bce_dice"
    bce_weight: float = 0.7
    dice_weight: float = 0.3
    pos_weight: float = 1.0
    focal_loss_gamma: float = 2.0
    label_smoothing: float = 0.0
    hard_negative_mining: bool = False
    hard_negative_ratio: float = 3.0

    min_valid_pixels_fraction_per_month: float = 0.20
    min_observations_per_pixel: int = 1
    min_valid_patch_fraction: float = 0.05
    aef_pca_dim: int = 8
    pca_num_samples_per_raster: int = 2000
    clip_features: bool = True
    clip_lower_percentile: float = 1.0
    clip_upper_percentile: float = 99.0
    clip_num_samples_per_tile: int = 20000
    normalization: str = "zscore"
    min_normalization_std: float = 1e-2
    normalized_feature_clip: float = 10.0

    year_weight_2020: float = 0.90
    year_weight_2021: float = 0.94
    year_weight_2022: float = 0.98
    year_weight_2023: float = 1.02
    year_weight_2024: float = 1.06
    year_weight_2025: float = 1.10

    label_fusion: str = "consensus_2of3"
    radd_positive_mode: str = "permissive"
    gladl_threshold: int = 2
    glads2_threshold: int = 1
    soft_vote_threshold: float = 0.5
    ignore_uncertain_single_source_positives: bool = False
    ignore_outside_label_extent: bool = True
    vote_weight_3: float = 1.0
    vote_weight_2: float = 0.8
    vote_weight_1: float = 0.3
    vote_weight_0: float = 1.0

    positive_patch_sampling: bool = True
    positive_patch_ratio: float = 0.5
    positive_patch_min_fraction: float = 0.001

    horizontal_flip: bool = True
    horizontal_flip_p: float = 0.5
    vertical_flip: bool = True
    vertical_flip_p: float = 0.5
    rotate90: bool = True
    rotate90_p: float = 0.75
    transpose: bool = False
    transpose_p: float = 0.2
    gaussian_noise: bool = False
    gaussian_noise_p: float = 0.15
    gaussian_noise_std: float = 0.01
    s2_brightness_jitter: bool = False
    s2_brightness_jitter_p: float = 0.15
    s2_brightness_scale: float = 0.05
    s2_contrast_jitter: bool = False
    s2_contrast_jitter_p: float = 0.15
    s2_contrast_scale: float = 0.05
    cutmix: bool = False
    cutmix_p: float = 0.0
    mixup: bool = False
    mixup_p: float = 0.0
    mixup_alpha: float = 0.2

    tta: bool = False
    tta_modes: list[str] = field(default_factory=lambda: ["hflip", "vflip", "rot90"])
    inference_threshold: float = 0.5

    visualization_num_samples: int = 4
    visualization_include_probability: bool = True
    visualization_include_input_preview: bool = True
    visualization_use_best_checkpoint: bool = False
    save_probability_rasters: bool = False

    def year_weights(self) -> dict[int, float]:
        """Return configured year weights as a dictionary."""

        return {
            2020: float(self.year_weight_2020),
            2021: float(self.year_weight_2021),
            2022: float(self.year_weight_2022),
            2023: float(self.year_weight_2023),
            2024: float(self.year_weight_2024),
            2025: float(self.year_weight_2025),
        }


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


def parse_cli_overrides(argv: list[str], config_cls: type[ExperimentConfig] = ExperimentConfig) -> ExperimentConfig:
    """Parse ``key=value`` command-line overrides into a config dataclass."""

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


def save_config_yaml(config: ExperimentConfig, path: str | Path) -> None:
    """Save the resolved config to YAML."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=True)


def config_field_names(config_cls: type[ExperimentConfig] = ExperimentConfig) -> list[str]:
    """Return supported override names."""

    return [item.name for item in fields(config_cls)]
