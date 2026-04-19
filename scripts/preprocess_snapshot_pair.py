"""Standalone preprocessing entrypoint for the simple early/late snapshot-pair pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from xylemx.config import parse_cli_overrides
from xylemx.preprocessing.pipeline import run_preprocessing


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="data/makeathon-challenge")
    parser.add_argument("--output-dir", type=Path, default=Path("output/preprocessing/snapshot_pair"))
    parser.add_argument("--preprocessing-num-workers", type=int, default=4)
    args, overrides = parser.parse_known_args()

    defaults = [
        f"data_root={args.data_root}",
        f"preprocessing_num_workers={args.preprocessing_num_workers}",
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
    ]
    config = parse_cli_overrides(defaults + overrides)
    print(json.dumps(asdict(config), indent=2, sort_keys=True), flush=True)
    run_preprocessing(config, args.output_dir)


if __name__ == "__main__":
    main()
