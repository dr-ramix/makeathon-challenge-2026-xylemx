"""CLI entrypoint for simple snapshot-pair training runs."""

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
    parser.add_argument("--preprocessing-dir", type=Path, default=Path("output/preprocessing/snapshot_pair"))
    parser.add_argument("--output-root", type=str, default="output/train_runs_snapshot_pair")
    args, overrides = parser.parse_known_args()

    defaults = [
        "model=resnet18_unet",
        "epochs=100",
        "lr=0.0002",
        f"data_root={args.data_root}",
        f"preprocessing_dir={args.preprocessing_dir}",
        f"output_root={args.output_root}",
        "reuse_preprocessing=true",
        "temporal_feature_mode=snapshot_pair",
        "early_window_start_year=2020",
        "early_window_end_year=2021",
        "late_window_start_year=2024",
        "late_window_end_year=2025",
        "selection_method=best_valid",
        "fallback_to_nearest_valid_year=true",
        "min_valid_pixels_fraction_per_snapshot=0.2",
        "skip_bad_snapshots=true",
        "require_s2=true",
        "require_s1=false",
        "require_aef=false",
        # Keep this pipeline S2-first by default; can be enabled via overrides.
        "use_s1_features=false",
        "use_aef_features=false",
        "loss=bce_dice",
    ]
    config = parse_cli_overrides(defaults + overrides)

    summary_path = Path(config.preprocessing_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing preprocessing artifacts at {summary_path}. "
            "Run scripts/preprocess_snapshot_pair.py first."
        )

    print(json.dumps(asdict(config), indent=2, sort_keys=True), flush=True)
    train_model(config)


if __name__ == "__main__":
    main()
