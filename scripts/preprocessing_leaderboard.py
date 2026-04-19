"""High-performance preprocessing entrypoint for leaderboard-oriented runs."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

from xylemx.config import ExperimentConfig, parse_cli_overrides
from xylemx.preprocessing.pipeline import run_preprocessing


# Tuned defaults focused on robust cross-region generalization + strong Union IoU.
DEFAULT_LEADERBOARD_OVERRIDES = [
    "temporal_feature_mode=snapshot_quad",
    "selection_method=best_valid",
    "fallback_to_nearest_valid_year=true",
    "skip_bad_snapshots=true",
    "min_valid_pixels_fraction_per_snapshot=0.15",
    "min_valid_pixels_fraction_per_month=0.15",
    "use_s1_features=true",
    "use_aef_features=true",
    "aef_pca_dim=12",
    "pca_num_samples_per_raster=4000",
    "label_fusion=consensus_2of3",
    "radd_positive_mode=permissive",
    "gladl_threshold=2",
    "glads2_threshold=2",
    "ignore_uncertain_single_source_positives=false",
    "clip_features=true",
    "clip_lower_percentile=0.5",
    "clip_upper_percentile=99.5",
    "normalized_feature_clip=8.0",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="data/makeathon-challenge")
    parser.add_argument("--output-dir", type=Path, default=Path("output/preprocessing_leaderboard"))
    parser.add_argument("--preprocessing-num-workers", type=int, default=4)
    args, overrides = parser.parse_known_args()

    config = parse_cli_overrides(DEFAULT_LEADERBOARD_OVERRIDES + overrides, config_cls=ExperimentConfig)
    config = replace(
        config,
        data_root=args.data_root,
        preprocessing_num_workers=args.preprocessing_num_workers,
    )

    print(json.dumps(asdict(config), indent=2, sort_keys=True), flush=True)
    run_preprocessing(config, args.output_dir)


if __name__ == "__main__":
    main()
