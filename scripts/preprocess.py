"""Standalone preprocessing entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

from xylemx.config import ExperimentConfig, parse_cli_overrides
from xylemx.preprocessing.pipeline import run_preprocessing


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="data/makeathon-challenge")
    parser.add_argument("--output-dir", type=Path, default=Path("output/preprocessing"))
    parser.add_argument("--preprocessing-num-workers", type=int, default=4)
    args, overrides = parser.parse_known_args()

    config = parse_cli_overrides(overrides) if overrides else ExperimentConfig()
    config = replace(
        config,
        data_root=args.data_root,
        preprocessing_num_workers=args.preprocessing_num_workers,
    )
    print(json.dumps(asdict(config), indent=2, sort_keys=True), flush=True)
    run_preprocessing(config, args.output_dir)


if __name__ == "__main__":
    main()
