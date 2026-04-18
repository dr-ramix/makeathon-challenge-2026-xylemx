"""Standalone preprocessing entrypoint."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from xylemx.config import ExperimentConfig
from xylemx.preprocessing.pipeline import run_preprocessing


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="data/makeathon-challenge")
    parser.add_argument("--output-dir", type=Path, default=Path("output/preprocessing"))
    args = parser.parse_args()

    config = ExperimentConfig(data_root=args.data_root)
    print(json.dumps(asdict(config), indent=2, sort_keys=True), flush=True)
    run_preprocessing(config, args.output_dir)


if __name__ == "__main__":
    main()
