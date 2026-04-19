"""I/O helpers for challenge tiles, rasters, and cached artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

S2_SUFFIX = "__s2_l2a"
S1_SUFFIX = "__s1_rtc"
GLADL_YEARS = [2021, 2022, 2023, 2024, 2025]

S2_PATTERN = re.compile(r"(?P<tile>.+)__s2_l2a_(?P<year>\d{4})_(?P<month>\d{1,2})\.tif$")
S1_PATTERN = re.compile(
    r"(?P<tile>.+)__s1_rtc_(?P<year>\d{4})_(?P<month>\d{1,2})_(?P<orbit>ascending|descending)\.tif$"
)
AEF_PATTERN = re.compile(r"(?P<tile>.+)_(?P<year>\d{4})\.tiff$")


@dataclass(slots=True)
class TileRecord:
    """Resolved input paths for one challenge tile."""

    tile_id: str
    split: str
    sentinel2_dir: Path
    sentinel2_files: list[Path]
    sentinel1_dir: Path | None
    sentinel1_files: list[Path]
    aef_files: list[Path]
    label_paths: dict[str, Any]

    @property
    def reference_s2_path(self) -> Path:
        """Return the Sentinel-2 raster used as the master grid."""

        if not self.sentinel2_files:
            raise FileNotFoundError(f"Tile {self.tile_id} is missing Sentinel-2 files")
        return self.sentinel2_files[0]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record to plain JSON-compatible data."""

        return {
            "tile_id": self.tile_id,
            "split": self.split,
            "sentinel2_dir": str(self.sentinel2_dir),
            "sentinel2_files": [str(path) for path in self.sentinel2_files],
            "sentinel1_dir": str(self.sentinel1_dir) if self.sentinel1_dir else None,
            "sentinel1_files": [str(path) for path in self.sentinel1_files],
            "aef_files": [str(path) for path in self.aef_files],
            "label_paths": stringify_paths(self.label_paths),
        }


def stringify_paths(value: Any) -> Any:
    """Convert nested ``Path`` structures into strings."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: stringify_paths(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [stringify_paths(inner) for inner in value]
    return value


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    """Write a JSON file with deterministic formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def get_raster_profile(path: str | Path) -> dict[str, Any]:
    """Return a raster profile as a plain dictionary."""

    with rasterio.open(path) as src:
        profile = src.profile.copy()
    return profile


def _iter_data_files(directory: Path, suffix: str) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == suffix and not path.name.startswith(".")
    )


def _sort_s2(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: _match_groups(S2_PATTERN, path, ("year", "month")))


def _sort_s1(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: _match_groups(S1_PATTERN, path, ("year", "month", "orbit")))


def _sort_aef(paths: list[Path]) -> list[Path]:
    return sorted(paths, key=lambda path: _match_groups(AEF_PATTERN, path, ("year",)))


def _match_groups(pattern: re.Pattern[str], path: Path, group_names: tuple[str, ...]) -> tuple[Any, ...]:
    match = pattern.match(path.name)
    if match is None:
        raise ValueError(f"Malformed filename: {path}")
    values: list[Any] = []
    for group_name in group_names:
        raw = match.group(group_name)
        values.append(int(raw) if raw.isdigit() else raw)
    return tuple(values)


def resolve_label_paths(data_root: str | Path, tile_id: str) -> dict[str, Any]:
    """Resolve weak-label raster paths for one training tile."""

    label_root = Path(data_root) / "labels" / "train"
    gladl = {
        year: {
            "alert": label_root / "gladl" / f"gladl_{tile_id}_alert{str(year)[-2:]}.tif",
            "alert_date": label_root / "gladl" / f"gladl_{tile_id}_alertDate{str(year)[-2:]}.tif",
        }
        for year in GLADL_YEARS
    }
    return {
        "radd": label_root / "radd" / f"radd_{tile_id}_labels.tif",
        "glads2": {
            "alert": label_root / "glads2" / f"glads2_{tile_id}_alert.tif",
            "alert_date": label_root / "glads2" / f"glads2_{tile_id}_alertDate.tif",
        },
        "gladl": gladl,
    }


def scan_tiles(data_root: str | Path, split: str) -> dict[str, TileRecord]:
    """Scan the dataset layout and build tile records for one split."""

    data_root = Path(data_root)
    s2_root = data_root / "sentinel-2" / split
    s1_root = data_root / "sentinel-1" / split
    aef_root = data_root / "aef-embeddings" / split

    if not s2_root.exists():
        raise FileNotFoundError(f"Missing Sentinel-2 split directory: {s2_root}")

    records: dict[str, TileRecord] = {}
    for tile_dir in sorted(s2_root.iterdir()):
        if not tile_dir.is_dir() or not tile_dir.name.endswith(S2_SUFFIX) or tile_dir.name.startswith("."):
            continue

        tile_id = tile_dir.name[: -len(S2_SUFFIX)]
        s2_files = _sort_s2(_iter_data_files(tile_dir, ".tif"))
        s1_dir = s1_root / f"{tile_id}{S1_SUFFIX}"
        s1_files = _sort_s1(_iter_data_files(s1_dir, ".tif"))
        aef_files = _sort_aef(sorted(aef_root.glob(f"{tile_id}_*.tiff")))

        records[tile_id] = TileRecord(
            tile_id=tile_id,
            split=split,
            sentinel2_dir=tile_dir,
            sentinel2_files=s2_files,
            sentinel1_dir=s1_dir if s1_dir.exists() else None,
            sentinel1_files=s1_files,
            aef_files=aef_files,
            label_paths=resolve_label_paths(data_root, tile_id) if split == "train" else {},
        )

    return records


def validate_tile_record(record: TileRecord) -> list[str]:
    """Return structural issues for a tile record."""

    issues: list[str] = []
    if not record.sentinel2_files:
        issues.append("missing Sentinel-2 files")
    if not record.aef_files:
        issues.append("missing AEF files")
    if record.split == "train":
        if not Path(record.label_paths["radd"]).exists():
            issues.append("missing RADD labels")
        glads2 = record.label_paths["glads2"]
        if not Path(glads2["alert"]).exists():
            issues.append("missing GLAD-S2 alert labels")
        if not Path(glads2["alert_date"]).exists():
            issues.append("missing GLAD-S2 alertDate labels")
        for year, paths in record.label_paths["gladl"].items():
            for key, path in paths.items():
                if not Path(path).exists():
                    issues.append(f"missing GLAD-L {key} for {year}")
    return issues


def read_raster(
    path: str | Path,
    *,
    indexes: int | list[int] | None = None,
    masked: bool = True,
) -> np.ndarray:
    """Read a raster into memory."""

    with rasterio.open(path) as src:
        array = src.read(indexes=indexes, masked=masked)
    return np.asarray(array)


def read_reprojected_raster(
    src_path: str | Path,
    dst_profile: dict[str, Any],
    *,
    resampling: Resampling = Resampling.bilinear,
    out_dtype: np.dtype | str = np.float32,
    fill_value: float | int = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """Read and reproject a raster to match the destination grid."""

    with rasterio.open(src_path) as src:
        same_grid = (
            src.crs == dst_profile["crs"]
            and src.transform == dst_profile["transform"]
            and src.width == int(dst_profile["width"])
            and src.height == int(dst_profile["height"])
        )
        if same_grid:
            source = src.read(masked=True).astype(out_dtype)
            valid_mask = src.dataset_mask() > 0
            array = np.asarray(source.filled(fill_value), dtype=out_dtype)
            if array.ndim == 2:
                array = array[None, ...]
            array[:, ~valid_mask] = fill_value
            return array, valid_mask

        source = src.read(masked=True)
        count = src.count
        source_dtype = np.dtype(out_dtype)
        destination = np.zeros((count, int(dst_profile["height"]), int(dst_profile["width"])), dtype=source_dtype)

        src_nodata = src.nodata
        if src_nodata is None or (isinstance(src_nodata, float) and np.isnan(src_nodata)):
            src_nodata = 0

        for band_index in range(count):
            band = np.asarray(source[band_index].filled(src_nodata), dtype=source_dtype)
            reproject(
                source=band,
                destination=destination[band_index],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=dst_profile["transform"],
                dst_crs=dst_profile["crs"],
                dst_nodata=src_nodata,
                resampling=resampling,
            )

        valid_source = src.dataset_mask().astype(np.uint8)
        valid_destination = np.zeros((int(dst_profile["height"]), int(dst_profile["width"])), dtype=np.uint8)
        reproject(
            source=valid_source,
            destination=valid_destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=0,
            dst_transform=dst_profile["transform"],
            dst_crs=dst_profile["crs"],
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    valid_mask = valid_destination > 0
    if destination.ndim == 2:
        destination = destination[None, ...]
    destination = destination.astype(source_dtype, copy=False)
    destination[:, ~valid_mask] = fill_value
    return destination, valid_mask


def write_single_band_geotiff(
    path: str | Path,
    array: np.ndarray,
    reference_profile: dict[str, Any],
    *,
    dtype: str = "uint8",
    nodata: int | float = 0,
) -> None:
    """Write a single-band GeoTIFF using a reference raster profile."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference_profile.copy()
    profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def get_feature_path(preprocessing_dir: str | Path, split: str, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "features" / split / f"{tile_id}.npy"


def get_feature_metadata_path(preprocessing_dir: str | Path, split: str, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "features" / split / f"{tile_id}.json"


def get_valid_mask_path(preprocessing_dir: str | Path, split: str, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "valid_masks" / split / f"{tile_id}.npy"


def get_target_path(preprocessing_dir: str | Path, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "targets" / f"{tile_id}.npy"


def get_ignore_mask_path(preprocessing_dir: str | Path, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "ignore_masks" / f"{tile_id}.npy"


def get_weight_map_path(preprocessing_dir: str | Path, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "weight_maps" / f"{tile_id}.npy"


def get_vote_count_path(preprocessing_dir: str | Path, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "vote_counts" / f"{tile_id}.npy"


def get_preview_path(preprocessing_dir: str | Path, split: str, tile_id: str) -> Path:
    return Path(preprocessing_dir) / "previews" / split / f"{tile_id}.npy"
