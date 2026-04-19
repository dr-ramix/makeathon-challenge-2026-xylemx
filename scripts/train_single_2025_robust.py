"""CLI entrypoint for cloud/noise-robust single-date summer-2025 training runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from xylemx.config import parse_cli_overrides
from xylemx.training.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="data/makeathon-challenge")
    parser.add_argument("--preprocessing-dir", type=Path, default=Path("output/preprocessing_single_2025_robust"))
    parser.add_argument("--output-root", type=str, default="output/train_runs_single_2025_robust")
    args, overrides = parser.parse_known_args()

    defaults = [
        "model=resnet18_unet",
        "epochs=100",
        "lr=0.0002",
        f"data_root={args.data_root}",
        f"preprocessing_dir={args.preprocessing_dir}",
        f"output_root={args.output_root}",
        "reuse_preprocessing=true",
        "temporal_feature_mode=single_2025_summer_robust",
        "snapshot_mode=false",
        "loss=bce_dice",
        "bce_weight=0.5",
        "dice_weight=0.5",
        "label_smoothing=0.02",
        "hard_negative_mining=true",
        "hard_negative_ratio=4.0",
        "dropout=0.15",
        "stochastic_depth=0.1",
        "ema=true",
        "gaussian_noise=true",
        "gaussian_noise_p=0.3",
        "gaussian_noise_std=0.02",
        "s2_brightness_jitter=true",
        "s2_brightness_jitter_p=0.3",
        "s2_brightness_scale=0.08",
        "s2_contrast_jitter=true",
        "s2_contrast_jitter_p=0.3",
        "s2_contrast_scale=0.08",
        "mixup=true",
        "mixup_p=0.2",
        "mixup_alpha=0.2",
        "cutmix=true",
        "cutmix_p=0.15",
        "positive_patch_ratio=0.6",
        "inference_threshold=0.45",
    ]
    config = parse_cli_overrides(defaults + overrides)

    summary_path = Path(config.preprocessing_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing preprocessing artifacts at {summary_path}. "
            "Run scripts/preprocessing_single_2025_robust.py first."
        )

    print(json.dumps(asdict(config), indent=2, sort_keys=True), flush=True)
    train_model(config)


if __name__ == "__main__":
    main()
