"""CLI entrypoint for the stronger high-generalization temporal FiLM model."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

from xylemx.temporal.config import TemporalTrainConfig, parse_temporal_cli_overrides
from xylemx.temporal.training import train_temporal_model


DEFAULT_HQ_OVERRIDES = [
    "model=film_temporal_unet_plus",
    "preprocessing_dir=output/preprocessing_temporal_hq",
    "output_root=output/temporal_runs_hq",
    "epochs=60",
    "batch_size=4",
    "lr=2.5e-4",
    "weight_decay=2e-4",
    "warmup_epochs=3",
    "stem_channels=40",
    "base_channels=48",
    "film_hidden_dim=192",
    "dropout=0.1",
    "lambda_time=0.6",
    "positive_patch_sampling=true",
    "positive_patch_ratio=0.6",
    "mixed_precision=true",
    "export_splits=['val']",
]


def main() -> None:
    config = parse_temporal_cli_overrides(
        DEFAULT_HQ_OVERRIDES + sys.argv[1:],
        config_cls=TemporalTrainConfig,
    )
    print(json.dumps(asdict(config), indent=2, sort_keys=True), flush=True)
    train_temporal_model(config)


if __name__ == "__main__":
    main()
