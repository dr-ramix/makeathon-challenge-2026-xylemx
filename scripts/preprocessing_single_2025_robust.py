"""Standalone preprocessing entrypoint for a cloud/noise-robust single-date 2025 pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from xylemx.config import parse_cli_overrides
from xylemx.single_2025.preprocessing import run_single_2025_preprocessing


def _parse_months(raw: str) -> tuple[int, ...]:
    months = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not months:
        raise ValueError("summer_months cannot be empty")
    for month in months:
        if month < 1 or month > 12:
            raise ValueError(f"Invalid month in summer_months: {month}")
    return months


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="data/makeathon-challenge")
    parser.add_argument("--output-dir", type=Path, default=Path("output/preprocessing_single_2025_robust"))
    parser.add_argument("--preprocessing-num-workers", type=int, default=4)
    parser.add_argument("--target-year", type=int, default=2025)
    parser.add_argument("--summer-months", type=str, default="6,7,8")
    parser.add_argument(
        "--allow-non-summer-fallback",
        action="store_true",
        help="If set, do not fail tiles when no usable summer image exists.",
    )
    args, overrides = parser.parse_known_args()

    defaults = [
        f"data_root={args.data_root}",
        f"preprocessing_num_workers={args.preprocessing_num_workers}",
        "temporal_feature_mode=single_2025_summer_robust",
        "snapshot_mode=false",
        "ignore_uncertain_single_source_positives=true",
        "min_valid_pixels_fraction_per_snapshot=0.05",
        "skip_bad_snapshots=true",
        "require_s2=true",
        "require_s1=false",
        "require_aef=false",
    ]
    config = parse_cli_overrides(defaults + overrides)
    summer_months = _parse_months(args.summer_months)
    strict_summer_only = not args.allow_non_summer_fallback

    payload = {
        "config": asdict(config),
        "output_dir": str(args.output_dir),
        "target_year": args.target_year,
        "summer_months": list(summer_months),
        "strict_summer_only": strict_summer_only,
        "s2_selection_mode": "summer_median_composite",
        "s1_selection_mode": "summer_median_composite",
        "add_observation_count_channels": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    run_single_2025_preprocessing(
        config,
        args.output_dir,
        target_year=args.target_year,
        summer_months=summer_months,
        strict_summer_only=strict_summer_only,
        s2_selection_mode="summer_median_composite",
        s1_selection_mode="summer_median_composite",
        add_observation_count_channels=True,
    )


if __name__ == "__main__":
    main()
