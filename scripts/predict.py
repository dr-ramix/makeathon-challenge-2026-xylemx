"""Standalone prediction/export entrypoint for trained runs."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from xylemx.config import ExperimentConfig
from xylemx.data.io import get_raster_profile, load_json, scan_tiles, write_single_band_geotiff
from xylemx.models.baseline import build_model
from xylemx.training.inference import load_normalized_tile, predict_probability_map


@dataclass(slots=True)
class PredictConfig:
    checkpoint: str = ""
    split: str = "test"
    output_dir: str = ""
    threshold: float | None = None
    tta: bool | None = None


def _coerce_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def parse_predict_args(argv: list[str]) -> PredictConfig:
    values = asdict(PredictConfig())
    for token in argv:
        if "=" not in token:
            raise ValueError(f"Expected key=value argument, got: {token}")
        key, raw_value = token.split("=", 1)
        if key not in values:
            raise KeyError(f"Unknown prediction argument: {key}")
        values[key] = _coerce_value(raw_value)
    if not values["checkpoint"]:
        raise ValueError("checkpoint=... is required")
    return PredictConfig(**values)


def main() -> None:
    predict_config = parse_predict_args(sys.argv[1:])
    checkpoint = torch.load(predict_config.checkpoint, map_location="cpu")
    train_config = ExperimentConfig(**checkpoint["config"])

    device = torch.device("cuda" if torch.cuda.is_available() and train_config.device != "cpu" else "cpu")
    threshold = float(predict_config.threshold if predict_config.threshold is not None else train_config.inference_threshold)
    tta = bool(predict_config.tta if predict_config.tta is not None else train_config.tta)

    run_dir = Path(checkpoint["run_dir"])
    preprocessing_dir = Path(checkpoint["preprocessing_dir"])
    output_dir = Path(predict_config.output_dir) if predict_config.output_dir else run_dir / "predictions" / predict_config.split
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_split = "test" if predict_config.split == "test" else "train"
    records = scan_tiles(train_config.data_root, "test" if predict_config.split == "test" else "train")
    if predict_config.split == "test":
        tile_ids = sorted(records.keys())
    elif predict_config.split == "val":
        tile_ids = load_json(preprocessing_dir / "val_tiles.json")
    elif predict_config.split == "train":
        tile_ids = load_json(preprocessing_dir / "train_tiles.json")
    else:
        raise ValueError(f"Unsupported split: {predict_config.split}")

    model = build_model(
        train_config.model,
        in_channels=int(checkpoint["in_channels"]),
        dropout=train_config.dropout,
        stochastic_depth=train_config.stochastic_depth,
        pretrained=train_config.encoder_pretrained,
    ).to(device)
    state_dict = checkpoint["ema_state_dict"] or checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    exported = []
    with torch.no_grad():
        for tile_id in tile_ids:
            features, valid_mask = load_normalized_tile(preprocessing_dir, split=cache_split, tile_id=tile_id, config=train_config)
            probability = predict_probability_map(
                model,
                features,
                valid_mask,
                device=device,
                patch_size=train_config.patch_size,
                stride=train_config.eval_stride,
                batch_size=train_config.batch_size,
                mixed_precision=train_config.mixed_precision,
                tta=tta,
                tta_modes=train_config.tta_modes,
            )
            binary = ((probability >= threshold) & valid_mask).astype("uint8")
            output_path = output_dir / f"{tile_id}_pred.tif"
            reference_profile = get_raster_profile(records[tile_id].reference_s2_path)
            write_single_band_geotiff(output_path, binary, reference_profile, dtype="uint8", nodata=0)
            exported.append({"tile_id": tile_id, "path": str(output_path), "positive_pixels": int(binary.sum())})

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"tiles": exported, "threshold": threshold}, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
