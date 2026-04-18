"""CLI entrypoint for the temporal segmentation + event-time pipeline."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from xylemx.temporal.config import TemporalTrainConfig, parse_temporal_cli_overrides
from xylemx.temporal.training import train_temporal_model


def main() -> None:
    config = parse_temporal_cli_overrides(sys.argv[1:], config_cls=TemporalTrainConfig)
    print(json.dumps(asdict(config), indent=2, sort_keys=True), flush=True)
    train_temporal_model(config)


if __name__ == "__main__":
    main()
