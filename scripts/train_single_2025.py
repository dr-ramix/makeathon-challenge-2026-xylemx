"""CLI entrypoint for single-date summer-2025 training runs."""

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
    parser.add_argument("--preprocessing-dir", type=Path, default=Path("output/preprocessing_single_2025"))
    parser.add_argument("--output-root", type=str, default="output/train_runs_single_2025")
    args, overrides = parser.parse_known_args()

    defaults = [
        "model=resnet18_unet",
        "epochs=100",
        "lr=0.0002",
        f"data_root={args.data_root}",
        f"preprocessing_dir={args.preprocessing_dir}",
        f"output_root={args.output_root}",
        "reuse_preprocessing=true",
        "temporal_feature_mode=single_2025_summer",
        "snapshot_mode=false",
    ]
    config = parse_cli_overrides(defaults + overrides)

    summary_path = Path(config.preprocessing_dir) / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing preprocessing artifacts at {summary_path}. "
            "Run scripts/preprocessing_single_2025.py first."
        )

    print(json.dumps(asdict(config), indent=2, sort_keys=True), flush=True)
    train_model(config)


if __name__ == "__main__":
    main()
