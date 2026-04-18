"""Batch conversion of prediction rasters into submission GeoJSON files."""

from __future__ import annotations

import json
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from submission_utils import raster_to_geojson


@dataclass(slots=True)
class SubmissionConfig:
    prediction_dir: str = ""
    output_dir: str = ""
    min_area_ha: float = 0.5
    zip_output: bool = True
    allow_empty: bool = True
    merged_filename: str = "submission.geojson"


def _coerce_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def parse_submission_args(argv: list[str]) -> SubmissionConfig:
    values = asdict(SubmissionConfig())
    for token in argv:
        if "=" not in token:
            raise ValueError(f"Expected key=value argument, got: {token}")
        key, raw_value = token.split("=", 1)
        if key not in values:
            raise KeyError(f"Unknown submission argument: {key}")
        values[key] = _coerce_value(raw_value)
    if not values["prediction_dir"]:
        raise ValueError("prediction_dir=... is required")
    if not values["output_dir"]:
        raise ValueError("output_dir=... is required")
    return SubmissionConfig(**values)


def _empty_feature_collection() -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": []}


def main() -> None:
    config = parse_submission_args(sys.argv[1:])
    prediction_dir = Path(config.prediction_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rasters = sorted(prediction_dir.glob("*_pred.tif"))
    if not rasters:
        raise FileNotFoundError(f"No prediction rasters found under {prediction_dir}")

    exported: list[dict[str, Any]] = []
    merged_features: list[dict[str, Any]] = []
    for raster_path in rasters:
        tile_id = raster_path.stem.removesuffix("_pred")
        geojson_path = output_dir / f"{tile_id}.geojson"
        try:
            geojson = raster_to_geojson(raster_path, output_path=geojson_path, min_area_ha=config.min_area_ha)
            exported.append(
                {
                    "tile_id": tile_id,
                    "raster_path": str(raster_path),
                    "geojson_path": str(geojson_path),
                    "num_features": len(geojson.get("features", [])),
                    "status": "ok",
                }
            )
        except ValueError as exc:
            if not config.allow_empty:
                raise
            payload = _empty_feature_collection()
            geojson_path.write_text(json.dumps(payload), encoding="utf-8")
            geojson = payload
            exported.append(
                {
                    "tile_id": tile_id,
                    "raster_path": str(raster_path),
                    "geojson_path": str(geojson_path),
                    "num_features": 0,
                    "status": f"empty:{exc}",
                }
            )
        for feature in geojson.get("features", []):
            properties = dict(feature.get("properties") or {})
            properties.setdefault("tile_id", tile_id)
            merged_features.append(
                {
                    "type": "Feature",
                    "geometry": feature["geometry"],
                    "properties": properties,
                }
            )

    merged_geojson = {"type": "FeatureCollection", "features": merged_features}
    merged_path = output_dir / config.merged_filename
    merged_path.write_text(json.dumps(merged_geojson), encoding="utf-8")

    zip_path: str | None = None
    if config.zip_output:
        zip_file = output_dir / "submission_geojsons.zip"
        with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for item in exported:
                handle.write(item["geojson_path"], arcname=Path(item["geojson_path"]).name)
            handle.write(merged_path, arcname=merged_path.name)
        zip_path = str(zip_file)

    summary = {
        "prediction_dir": str(prediction_dir),
        "output_dir": str(output_dir),
        "min_area_ha": float(config.min_area_ha),
        "zip_output": bool(config.zip_output),
        "allow_empty": bool(config.allow_empty),
        "zip_path": zip_path,
        "merged_geojson_path": str(merged_path),
        "num_merged_features": len(merged_features),
        "tiles": exported,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
