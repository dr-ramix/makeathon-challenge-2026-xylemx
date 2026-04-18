"""CLI entrypoint for training runs."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from xylemx.config import parse_cli_overrides
from xylemx.training.train import train_model


def main() -> None:
    config = parse_cli_overrides(sys.argv[1:])
    print(json.dumps(asdict(config), indent=2, sort_keys=True), flush=True)
    train_model(config)


if __name__ == "__main__":
    main()
