"""Standalone preprocessing entrypoint for the temporal pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

from xylemx.temporal.config import TemporalPreprocessingConfig, parse_temporal_cli_overrides
from xylemx.temporal.preprocessing import run_temporal_preprocessing


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="data/makeathon-challenge")
    parser.add_argument("--output-dir", type=Path, default=Path("output/preprocessing_temporal"))
    args, overrides = parser.parse_known_args()

    config = parse_temporal_cli_overrides(overrides, config_cls=TemporalPreprocessingConfig) if overrides else TemporalPreprocessingConfig()
    config = replace(config, data_root=args.data_root)
    print(json.dumps(asdict(config), indent=2, sort_keys=True), flush=True)
    run_temporal_preprocessing(config, args.output_dir)


if __name__ == "__main__":
    main()
