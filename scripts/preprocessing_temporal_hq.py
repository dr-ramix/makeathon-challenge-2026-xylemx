"""High-quality preprocessing entrypoint for the stronger temporal FiLM pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

from xylemx.temporal.config import TemporalPreprocessingConfig, parse_temporal_cli_overrides
from xylemx.temporal.preprocessing import run_temporal_preprocessing


DEFAULT_HQ_OVERRIDES = [
    "representation=early_middle_late_deltas",
    "include_sentinel2=true",
    "include_sentinel1=true",
    "include_aef=true",
    "add_s2_indices=true",
    "add_validity_channels=true",
    "add_missing_channel=true",
    "aef_pca_dim=12",
    "pca_num_samples_per_raster=4000",
    "include_condition_vector=true",
    "cond_include_geo=true",
    "cond_include_quality=true",
    "cond_include_aef_summary=true",
    "cond_aef_summary_dim=12",
    "min_label_confidence=0.5",
    "ignore_uncertain_single_source_positives=true",
    "time_bin_mode=year",
    "time_merge_strategy=highest_confidence",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="data/makeathon-challenge")
    parser.add_argument("--output-dir", type=Path, default=Path("output/preprocessing_temporal_hq"))
    args, overrides = parser.parse_known_args()

    config = parse_temporal_cli_overrides(
        DEFAULT_HQ_OVERRIDES + overrides,
        config_cls=TemporalPreprocessingConfig,
    )
    config = replace(config, data_root=args.data_root)
    print(json.dumps(asdict(config), indent=2, sort_keys=True), flush=True)
    run_temporal_preprocessing(config, args.output_dir)


if __name__ == "__main__":
    main()
