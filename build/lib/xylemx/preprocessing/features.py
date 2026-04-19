"""Multimodal feature engineering for tile-based deforestation segmentation."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from sklearn.decomposition import PCA

from xylemx.config import ExperimentConfig
from xylemx.data.io import AEF_PATTERN, S1_PATTERN, S2_PATTERN, TileRecord, get_raster_profile, read_reprojected_raster

LOGGER = logging.getLogger(__name__)

S2_BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
SNAPSHOT_STAGE_NAMES = ("early", "middle1", "middle2", "late")


class TilePreprocessingError(RuntimeError):
    """Raised when a tile cannot be preprocessed under the active config."""


@dataclass(slots=True)
class FeaturePack:
    """Engineered per-tile features and lightweight metadata."""

    features: np.ndarray
    valid_mask: np.ndarray
    preview: np.ndarray | None
    feature_names: list[str]
    metadata: dict[str, Any]


@dataclass(slots=True)
class AefPcaModel:
    """Serializable PCA projection used for AEF embeddings."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """Project vectors from the original embedding space into PCA space."""

        centered = vectors.astype(np.float32, copy=False) - self.mean[None, :]
        return centered @ self.components.T

    def to_payload(self) -> dict[str, list[list[float]] | list[float]]:
        """Convert the PCA model to a serializable payload."""

        return {
            "mean": self.mean.astype(np.float32).tolist(),
            "components": self.components.astype(np.float32).tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.astype(np.float32).tolist(),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, list[list[float]] | list[float]]) -> "AefPcaModel":
        """Rebuild a PCA model from saved JSON data."""

        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            components=np.asarray(payload["components"], dtype=np.float32),
            explained_variance_ratio=np.asarray(payload["explained_variance_ratio"], dtype=np.float32),
        )


@dataclass(slots=True)
class SnapshotCandidate:
    """One candidate snapshot aligned to the Sentinel-2 grid."""

    path: Path
    year: int
    month: int | None
    array: np.ndarray
    valid_mask: np.ndarray
    valid_fraction: float
    source_paths: list[Path]
    orbit_paths: dict[str, Path] | None = None

    @property
    def time_index(self) -> int:
        month = self.month if self.month is not None else 6
        return self.year * 12 + month

    def describe(self) -> dict[str, Any]:
        """Serialize candidate metadata for audit logs."""

        return {
            "path": str(self.path),
            "year": self.year,
            "month": self.month,
            "valid_fraction": float(self.valid_fraction),
            "source_paths": [str(path) for path in self.source_paths],
            "orbit_paths": {key: str(value) for key, value in (self.orbit_paths or {}).items()},
        }


@dataclass(slots=True)
class SnapshotSelection:
    """Selected snapshots plus selection audit information."""

    selected: dict[str, SnapshotCandidate | None]
    metadata: dict[str, Any]


def build_feature_names(config: ExperimentConfig) -> list[str]:
    """Return the stable feature-channel ordering."""

    if config.temporal_feature_mode not in {"snapshot_pair", "snapshot_quad"}:
        raise ValueError(f"Unsupported temporal_feature_mode: {config.temporal_feature_mode}")
    stage_names = [stage_name for stage_name, *_ in _snapshot_stage_specs(config)]
    names: list[str] = []
    for stage_name in stage_names:
        names.extend(f"s2_{stage_name}_{band}" for band in S2_BAND_NAMES)
    if config.temporal_feature_mode in {"snapshot_pair", "snapshot_quad"}:
        names.extend(f"s2_delta_{band}" for band in S2_BAND_NAMES)
    for stage_name in stage_names:
        names.append(f"s1_{stage_name}")
    if config.temporal_feature_mode in {"snapshot_pair", "snapshot_quad"}:
        names.append("s1_delta")
    if config.aef_pca_dim > 0:
        for stage_name in stage_names:
            names.extend(f"aef_{stage_name}_pc{index + 1:02d}" for index in range(config.aef_pca_dim))
        if config.temporal_feature_mode in {"snapshot_pair", "snapshot_quad"}:
            names.extend(f"aef_delta_pc{index + 1:02d}" for index in range(config.aef_pca_dim))
    return names


def _snapshot_stage_specs(config: ExperimentConfig) -> list[tuple[str, int, int, int]]:
    """Return the ordered snapshot stages for the active temporal feature mode."""

    if config.temporal_feature_mode == "snapshot_pair":
        return [
            ("early", config.early_window_start_year, config.early_window_end_year, config.early_window_start_year),
            ("late", config.late_window_start_year, config.late_window_end_year, config.late_window_end_year),
        ]
    if config.temporal_feature_mode == "snapshot_quad":
        return [
            ("early", config.early_window_start_year, config.early_window_end_year, config.early_window_start_year),
            ("middle1", config.middle1_window_start_year, config.middle1_window_end_year, config.middle1_window_start_year),
            ("middle2", config.middle2_window_start_year, config.middle2_window_end_year, config.middle2_window_start_year),
            ("late", config.late_window_start_year, config.late_window_end_year, config.late_window_end_year),
        ]
    raise ValueError(f"Unsupported temporal_feature_mode: {config.temporal_feature_mode}")


def _parse_match(pattern, path: Path) -> tuple[str, ...]:
    match = pattern.match(path.name)
    if match is None:
        raise ValueError(f"Malformed filename: {path}")
    return match.groups()


def _window_contains(year: int, start_year: int, end_year: int) -> bool:
    return start_year <= year <= end_year


def _window_distance_months(candidate_index: int, *, start_year: int, end_year: int) -> int:
    start_index = start_year * 12 + 1
    end_index = end_year * 12 + 12
    if start_index <= candidate_index <= end_index:
        return 0
    if candidate_index < start_index:
        return start_index - candidate_index
    return candidate_index - end_index


def _year_distance(year: int, *, start_year: int, end_year: int) -> int:
    if start_year <= year <= end_year:
        return 0
    if year < start_year:
        return start_year - year
    return year - end_year


def _choose_candidate(
    candidates: list[SnapshotCandidate],
    *,
    start_year: int,
    end_year: int,
    target_year: int,
    selection_method: str,
    fallback_to_nearest: bool,
) -> tuple[SnapshotCandidate | None, str | None]:
    if selection_method != "best_valid":
        raise ValueError(f"Unsupported selection_method: {selection_method}")

    in_window = [candidate for candidate in candidates if _window_contains(candidate.year, start_year, end_year)]
    if in_window:
        chosen = max(in_window, key=lambda item: (item.valid_fraction, item.time_index))
        return chosen, None

    if not fallback_to_nearest or not candidates:
        return None, None

    chosen = min(
        candidates,
        key=lambda item: (
            _window_distance_months(item.time_index, start_year=start_year, end_year=end_year),
            abs(item.year - target_year),
            -item.valid_fraction,
            -item.time_index,
        ),
    )
    return chosen, f"fallback_to_nearest_valid:{chosen.path.name}"


def _choose_year_path(
    year_to_path: dict[int, Path],
    *,
    start_year: int,
    end_year: int,
    target_year: int,
    fallback_to_nearest: bool,
) -> tuple[tuple[int, Path] | None, str | None]:
    in_window = {year: path for year, path in year_to_path.items() if _window_contains(year, start_year, end_year)}
    if target_year in in_window:
        return (target_year, in_window[target_year]), None
    if in_window:
        year = min(in_window.keys(), key=lambda item: (abs(item - target_year), -item))
        note = None if year == target_year else f"nearest_valid_within_window:{year}"
        return (year, in_window[year]), note
    if not fallback_to_nearest or not year_to_path:
        return None, None
    year = min(year_to_path.keys(), key=lambda item: (_year_distance(item, start_year=start_year, end_year=end_year), abs(item - target_year), -item))
    return (year, year_to_path[year]), f"fallback_to_nearest_valid:{year}"


def _observation_valid_fraction(valid_mask: np.ndarray, array: np.ndarray | None = None) -> float:
    usable = valid_mask.copy()
    if array is not None:
        if array.ndim == 3:
            usable &= np.isfinite(array).all(axis=0)
        else:
            usable &= np.isfinite(array)
    return float(usable.mean())


def _masked_percentile(stack: np.ndarray, percentile: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanpercentile(stack, percentile, axis=0).astype(np.float32)


def aggregate_temporal_stack(
    stack: np.ndarray,
    weights: np.ndarray,
    *,
    min_observations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a temporal stack into weighted mean, std, and quartile delta."""

    finite = np.isfinite(stack)
    weighted_sum = np.nansum(stack * weights[:, None, None], axis=0)
    weight_total = np.sum(finite * weights[:, None, None], axis=0, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted_mean = weighted_sum / np.maximum(weight_total, 1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            std = np.nanstd(stack, axis=0).astype(np.float32)
            delta = (_masked_percentile(stack, 75.0) - _masked_percentile(stack, 25.0)).astype(np.float32)

    valid_counts = finite.sum(axis=0)
    weighted_mean[valid_counts < min_observations] = np.nan
    std[valid_counts < min_observations] = np.nan
    delta[valid_counts < min_observations] = np.nan
    return weighted_mean.astype(np.float32), std, delta


def _empty_block(num_channels: int, height: int, width: int) -> np.ndarray:
    return np.full((num_channels, height, width), np.nan, dtype=np.float32)


def _delta_block(late: np.ndarray, early: np.ndarray) -> np.ndarray:
    delta = late - early
    invalid = ~np.isfinite(late) | ~np.isfinite(early)
    delta[invalid] = np.nan
    return delta.astype(np.float32)


def _load_s2_candidate(
    path: Path,
    dst_profile: dict[str, Any],
    config: ExperimentConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    array, valid_mask = read_reprojected_raster(path, dst_profile, resampling=Resampling.bilinear, out_dtype=np.float32)
    valid = valid_mask & np.isfinite(array).all(axis=0) & np.any(array > 0, axis=0)
    valid_fraction = _observation_valid_fraction(valid)
    if config.skip_bad_snapshots and valid_fraction < config.min_valid_pixels_fraction_per_snapshot:
        return None, None, valid_fraction
    array[:, ~valid] = np.nan
    return array.astype(np.float32, copy=False), valid.astype(bool), valid_fraction


def _load_s1_candidate_band(
    path: Path,
    dst_profile: dict[str, Any],
    config: ExperimentConfig,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    array, valid_mask = read_reprojected_raster(path, dst_profile, resampling=Resampling.bilinear, out_dtype=np.float32)
    band = array[0]
    valid = valid_mask & np.isfinite(band)
    valid_fraction = _observation_valid_fraction(valid)
    if config.skip_bad_snapshots and valid_fraction < config.min_valid_pixels_fraction_per_snapshot:
        return None, None, valid_fraction
    band = band.astype(np.float32, copy=False)
    band[~valid] = np.nan
    return band, valid.astype(bool), valid_fraction


def _select_s2_snapshots(record: TileRecord, dst_profile: dict[str, Any], config: ExperimentConfig) -> SnapshotSelection:
    candidates: list[SnapshotCandidate] = []
    skipped: list[dict[str, Any]] = []
    for path in record.sentinel2_files:
        year_raw, month_raw = _parse_match(S2_PATTERN, path)[1:]
        year = int(year_raw)
        month = int(month_raw)
        array, valid_mask, valid_fraction = _load_s2_candidate(path, dst_profile, config)
        if array is None or valid_mask is None:
            skipped.append({"path": str(path), "year": year, "month": month, "valid_fraction": valid_fraction})
            continue
        candidates.append(
            SnapshotCandidate(
                path=path,
                year=year,
                month=month,
                array=array,
                valid_mask=valid_mask,
                valid_fraction=valid_fraction,
                source_paths=[path],
            )
        )

    selected: dict[str, SnapshotCandidate | None] = {}
    fallback_notes: list[str] = []
    metadata: dict[str, Any] = {}
    for stage_name, start_year, end_year, target_year in _snapshot_stage_specs(config):
        chosen, note = _choose_candidate(
            candidates,
            start_year=start_year,
            end_year=end_year,
            target_year=target_year,
            selection_method=config.selection_method,
            fallback_to_nearest=config.fallback_to_nearest_valid_year,
        )
        selected[stage_name] = chosen
        metadata[f"selected_{stage_name}_s2"] = chosen.describe() if chosen is not None else None
        if note:
            fallback_notes.append(f"{stage_name}:{note}")
    return SnapshotSelection(
        selected=selected,
        metadata={**metadata, "fallback_decisions": fallback_notes, "skipped_bad_snapshots": skipped},
    )


def _select_s1_snapshots(record: TileRecord, dst_profile: dict[str, Any], config: ExperimentConfig) -> SnapshotSelection:
    grouped: dict[tuple[int, int], dict[str, SnapshotCandidate]] = {}
    skipped: list[dict[str, Any]] = []

    for path in record.sentinel1_files:
        year_raw, month_raw, orbit = _parse_match(S1_PATTERN, path)[1:]
        year = int(year_raw)
        month = int(month_raw)
        band, valid_mask, valid_fraction = _load_s1_candidate_band(path, dst_profile, config)
        if band is None or valid_mask is None:
            skipped.append(
                {"path": str(path), "year": year, "month": month, "orbit": orbit, "valid_fraction": valid_fraction}
            )
            continue
        grouped.setdefault((year, month), {})[orbit] = SnapshotCandidate(
            path=path,
            year=year,
            month=month,
            array=band[None, ...],
            valid_mask=valid_mask,
            valid_fraction=valid_fraction,
            source_paths=[path],
            orbit_paths={orbit: path},
        )

    month_candidates: list[SnapshotCandidate] = []
    for (year, month), orbit_candidates in grouped.items():
        arrays = [candidate.array[0] for candidate in orbit_candidates.values()]
        masks = [candidate.valid_mask for candidate in orbit_candidates.values()]
        stacked = np.stack(arrays, axis=0).astype(np.float32)
        finite = np.isfinite(stacked)
        value_sum = np.nansum(stacked, axis=0).astype(np.float32)
        value_count = finite.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            averaged = (value_sum / np.maximum(value_count, 1)).astype(np.float32)
        valid = np.logical_or.reduce(masks)
        averaged[~valid] = np.nan
        month_candidates.append(
            SnapshotCandidate(
                path=next(iter(orbit_candidates.values())).path,
                year=year,
                month=month,
                array=averaged[None, ...],
                valid_mask=valid,
                valid_fraction=float(valid.mean()),
                source_paths=[candidate.path for candidate in orbit_candidates.values()],
                orbit_paths={key: value.path for key, value in orbit_candidates.items()},
            )
        )

    selected: dict[str, SnapshotCandidate | None] = {}
    fallback_notes: list[str] = []
    metadata: dict[str, Any] = {}
    for stage_name, start_year, end_year, target_year in _snapshot_stage_specs(config):
        chosen, note = _choose_candidate(
            month_candidates,
            start_year=start_year,
            end_year=end_year,
            target_year=target_year,
            selection_method=config.selection_method,
            fallback_to_nearest=config.fallback_to_nearest_valid_year,
        )
        selected[stage_name] = chosen
        metadata[f"selected_{stage_name}_s1"] = chosen.describe() if chosen is not None else None
        if note:
            fallback_notes.append(f"{stage_name}:{note}")
    return SnapshotSelection(
        selected=selected,
        metadata={**metadata, "fallback_decisions": fallback_notes, "skipped_bad_snapshots": skipped},
    )


def _collect_aef_paths(record: TileRecord) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for path in record.aef_files:
        year_raw = _parse_match(AEF_PATTERN, path)[1]
        pairs.append((int(year_raw), path))
    return sorted(pairs, key=lambda item: item[0])


def _select_aef_paths(
    record: TileRecord,
    config: ExperimentConfig,
) -> tuple[dict[str, tuple[int, Path] | None], dict[str, Any]]:
    year_to_path = {year: path for year, path in _collect_aef_paths(record)}
    selected: dict[str, tuple[int, Path] | None] = {}
    metadata: dict[str, Any] = {}
    fallback_notes: list[str] = []
    for stage_name, start_year, end_year, target_year in _snapshot_stage_specs(config):
        chosen, note = _choose_year_path(
            year_to_path,
            start_year=start_year,
            end_year=end_year,
            target_year=target_year,
            fallback_to_nearest=config.fallback_to_nearest_valid_year,
        )
        selected[stage_name] = chosen
        metadata[f"selected_{stage_name}_aef"] = (
            {"year": chosen[0], "path": str(chosen[1])} if chosen is not None else None
        )
        if note:
            fallback_notes.append(f"{stage_name}:{note}")
    return selected, {**metadata, "fallback_decisions": fallback_notes, "skipped_bad_snapshots": []}


def fit_aef_pca_model(
    records: dict[str, TileRecord],
    train_tile_ids: list[str],
    config: ExperimentConfig,
) -> AefPcaModel:
    """Fit PCA on train tiles only using selected early and late AEF pixels."""

    if config.aef_pca_dim <= 0:
        raise RuntimeError("AEF PCA fit requested with aef_pca_dim <= 0")

    samples: list[np.ndarray] = []
    rng = np.random.default_rng(config.seed)
    total_rasters = 0
    for tile_id in train_tile_ids:
        record = records[tile_id]
        selected_paths_by_stage, _ = _select_aef_paths(record, config)
        selected_paths = []
        for chosen in selected_paths_by_stage.values():
            if chosen is None:
                continue
            path = chosen[1]
            if path not in selected_paths:
                selected_paths.append(path)
        LOGGER.info("Collecting AEF PCA samples for tile %s from %d snapshot(s)", tile_id, len(selected_paths))
        for path in selected_paths:
            total_rasters += 1
            try:
                with rasterio.open(path) as src:
                    valid_mask = src.dataset_mask() > 0
                    valid_indices = np.argwhere(valid_mask)
                    if valid_indices.size == 0:
                        continue
                    sample_count = min(config.pca_num_samples_per_raster, valid_indices.shape[0])
                    sample_indices = rng.choice(valid_indices.shape[0], size=sample_count, replace=False)
                    rows = valid_indices[sample_indices, 0]
                    cols = valid_indices[sample_indices, 1]
                    xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
                    sampled = np.asarray(list(src.sample(list(zip(xs, ys, strict=True)))), dtype=np.float32)
            except rasterio.errors.RasterioError as exc:
                LOGGER.warning("Skipping unreadable AEF raster during PCA fit: %s (%s)", path, exc)
                continue
            vectors = sampled[np.isfinite(sampled).all(axis=1)]
            if vectors.size == 0:
                continue
            samples.append(vectors.astype(np.float32, copy=False))

    if not samples:
        raise RuntimeError("AEF PCA fit failed because no valid training samples were collected")

    LOGGER.info("Fitting AEF PCA from %d sampled rasters", total_rasters)
    sample_matrix = np.concatenate(samples, axis=0)
    pca = PCA(n_components=config.aef_pca_dim, random_state=config.seed)
    pca.fit(sample_matrix)
    LOGGER.info("Finished fitting AEF PCA with sample matrix shape %s", sample_matrix.shape)
    return AefPcaModel(
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )


def _load_projected_aef_snapshot(
    path: Path,
    dst_profile: dict[str, Any],
    config: ExperimentConfig,
    pca_model: AefPcaModel,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    try:
        with rasterio.open(path) as src:
            source = src.read(masked=True).astype(np.float32)
            source_data = np.asarray(source.filled(np.nan), dtype=np.float32)
            source_valid = np.isfinite(source_data).all(axis=0)
            valid_fraction = _observation_valid_fraction(source_valid)
            if config.skip_bad_snapshots and valid_fraction < config.min_valid_pixels_fraction_per_snapshot:
                return None, None, valid_fraction

            flat = source_data.reshape(source_data.shape[0], -1).T
            projected = pca_model.transform(np.nan_to_num(flat, nan=0.0)).T.reshape(
                config.aef_pca_dim,
                src.height,
                src.width,
            )
            projected[:, ~source_valid] = np.nan

            destination = np.zeros((config.aef_pca_dim, int(dst_profile["height"]), int(dst_profile["width"])), dtype=np.float32)
            for channel_index in range(config.aef_pca_dim):
                reproject(
                    source=np.nan_to_num(projected[channel_index], nan=0.0),
                    destination=destination[channel_index],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=0.0,
                    dst_transform=dst_profile["transform"],
                    dst_crs=dst_profile["crs"],
                    dst_nodata=0.0,
                    resampling=Resampling.bilinear,
                )

            valid_destination = np.zeros((int(dst_profile["height"]), int(dst_profile["width"])), dtype=np.uint8)
            reproject(
                source=source_valid.astype(np.uint8),
                destination=valid_destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=0,
                dst_transform=dst_profile["transform"],
                dst_crs=dst_profile["crs"],
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
    except rasterio.errors.RasterioError as exc:
        LOGGER.warning("Skipping unreadable AEF raster %s (%s)", path, exc)
        return None, None, 0.0

    valid = valid_destination > 0
    valid_fraction = _observation_valid_fraction(valid)
    if config.skip_bad_snapshots and valid_fraction < config.min_valid_pixels_fraction_per_snapshot:
        return None, None, valid_fraction

    destination[:, ~valid] = np.nan
    return destination.astype(np.float32), valid.astype(bool), valid_fraction


def _enforce_required_modality(
    *,
    modality_name: str,
    require_modality: bool,
    selection_present: bool,
    config: ExperimentConfig,
) -> None:
    if selection_present:
        return
    if require_modality and config.skip_tile_if_required_modality_missing:
        raise TilePreprocessingError(f"Missing required {modality_name} snapshots")
    if require_modality:
        LOGGER.warning("Required modality %s is missing but tile skipping is disabled", modality_name)


def _build_s2_snapshot_features(
    record: TileRecord,
    dst_profile: dict[str, Any],
    config: ExperimentConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    selection = _select_s2_snapshots(record, dst_profile, config)
    height = int(dst_profile["height"])
    width = int(dst_profile["width"])
    stage_arrays = [
        selection.selected[stage_name].array
        if selection.selected.get(stage_name) is not None
        else _empty_block(len(S2_BAND_NAMES), height, width)
        for stage_name, *_ in _snapshot_stage_specs(config)
    ]
    _enforce_required_modality(
        modality_name="Sentinel-2",
        require_modality=config.require_s2,
        selection_present=all(selection.selected.get(stage_name) is not None for stage_name, *_ in _snapshot_stage_specs(config)),
        config=config,
    )
    feature_blocks = stage_arrays.copy()
    if config.temporal_feature_mode in {"snapshot_pair", "snapshot_quad"}:
        feature_blocks.append(_delta_block(stage_arrays[-1].copy(), stage_arrays[0].copy()))
    features = np.concatenate(feature_blocks, axis=0).astype(np.float32)
    metadata = {
        "num_s2_candidate_snapshots": len(record.sentinel2_files),
        **selection.metadata,
    }
    return features, metadata


def _build_s1_snapshot_features(
    record: TileRecord,
    dst_profile: dict[str, Any],
    config: ExperimentConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    selection = _select_s1_snapshots(record, dst_profile, config)
    height = int(dst_profile["height"])
    width = int(dst_profile["width"])
    stage_arrays = [
        selection.selected[stage_name].array
        if selection.selected.get(stage_name) is not None
        else _empty_block(1, height, width)
        for stage_name, *_ in _snapshot_stage_specs(config)
    ]
    _enforce_required_modality(
        modality_name="Sentinel-1",
        require_modality=config.require_s1,
        selection_present=any(selection.selected.get(stage_name) is not None for stage_name, *_ in _snapshot_stage_specs(config)),
        config=config,
    )
    feature_blocks = stage_arrays.copy()
    if config.temporal_feature_mode in {"snapshot_pair", "snapshot_quad"}:
        feature_blocks.append(_delta_block(stage_arrays[-1].copy(), stage_arrays[0].copy()))
    features = np.concatenate(feature_blocks, axis=0).astype(np.float32)
    metadata = {
        "num_s1_candidate_snapshots": len(record.sentinel1_files),
        **selection.metadata,
    }
    return features, metadata


def _build_aef_snapshot_features(
    record: TileRecord,
    dst_profile: dict[str, Any],
    config: ExperimentConfig,
    pca_model: AefPcaModel | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    height = int(dst_profile["height"])
    width = int(dst_profile["width"])
    empty = np.empty((0, height, width), dtype=np.float32) if config.aef_pca_dim <= 0 else _empty_block(config.aef_pca_dim, height, width)

    if pca_model is None or config.aef_pca_dim <= 0:
        _enforce_required_modality(
            modality_name="AEF",
            require_modality=config.require_aef,
            selection_present=False,
            config=config,
        )
        stage_count = len(_snapshot_stage_specs(config))
        feature_blocks = [empty.copy() for _ in range(stage_count)]
        if config.temporal_feature_mode in {"snapshot_pair", "snapshot_quad"} and config.aef_pca_dim > 0:
            feature_blocks.append(empty.copy())
        features = np.concatenate(feature_blocks, axis=0) if config.aef_pca_dim > 0 else np.empty((0, height, width), dtype=np.float32)
        return features, {
            "num_aef_candidate_snapshots": len(record.aef_files),
            **{f"selected_{stage_name}_aef": None for stage_name, *_ in _snapshot_stage_specs(config)},
            "fallback_decisions": [],
            "skipped_bad_snapshots": [],
        }

    selected_paths, metadata = _select_aef_paths(record, config)
    stage_arrays = {stage_name: empty.copy() for stage_name, *_ in _snapshot_stage_specs(config)}
    skipped = list(metadata["skipped_bad_snapshots"])
    any_present = False

    for stage_name, chosen in selected_paths.items():
        if chosen is None:
            continue
        projected, _valid_mask, valid_fraction = _load_projected_aef_snapshot(chosen[1], dst_profile, config, pca_model)
        if projected is None:
            skipped.append({"path": str(chosen[1]), "year": chosen[0], "stage": stage_name, "valid_fraction": valid_fraction})
            continue
        stage_arrays[stage_name] = projected
        any_present = True

    _enforce_required_modality(
        modality_name="AEF",
        require_modality=config.require_aef,
        selection_present=any_present,
        config=config,
    )
    metadata["skipped_bad_snapshots"] = skipped
    feature_blocks = [stage_arrays[stage_name] for stage_name, *_ in _snapshot_stage_specs(config)]
    if config.temporal_feature_mode in {"snapshot_pair", "snapshot_quad"}:
        feature_blocks.append(_delta_block(feature_blocks[-1].copy(), feature_blocks[0].copy()))
    features = np.concatenate(feature_blocks, axis=0).astype(np.float32)
    metadata["num_aef_candidate_snapshots"] = len(record.aef_files)
    return features, metadata


def build_preview_from_late_s2_features(s2_features: np.ndarray, config: ExperimentConfig) -> np.ndarray | None:
    """Create a lightweight RGB-like preview from the late Sentinel-2 snapshot."""

    if s2_features.shape[0] < len(S2_BAND_NAMES):
        return None
    if config.temporal_feature_mode == "snapshot_pair":
        offset = len(S2_BAND_NAMES)
    elif config.temporal_feature_mode == "snapshot_quad":
        offset = len(S2_BAND_NAMES) * 3
    else:
        return None
    rgb = np.stack([s2_features[offset + 3], s2_features[offset + 2], s2_features[offset + 1]], axis=0).astype(np.float32)
    preview = np.zeros_like(rgb, dtype=np.uint8)
    for channel_index in range(3):
        channel = rgb[channel_index]
        finite = np.isfinite(channel)
        if not finite.any():
            continue
        lo = float(np.percentile(channel[finite], 2.0))
        hi = float(np.percentile(channel[finite], 98.0))
        scaled = np.clip((channel - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        preview[channel_index] = np.round(np.nan_to_num(scaled, nan=0.0) * 255.0).astype(np.uint8)
    return preview


def build_multimodal_feature_pack(
    record: TileRecord,
    config: ExperimentConfig,
    pca_model: AefPcaModel | None,
) -> FeaturePack:
    """Build the final per-tile feature tensor for one tile."""

    if config.temporal_feature_mode not in {"snapshot_pair", "snapshot_quad"}:
        raise ValueError(f"Unsupported temporal_feature_mode: {config.temporal_feature_mode}")

    dst_profile = get_raster_profile(record.reference_s2_path)
    s2_features, s2_metadata = _build_s2_snapshot_features(record, dst_profile, config)
    s1_features, s1_metadata = _build_s1_snapshot_features(record, dst_profile, config)
    aef_features, aef_metadata = _build_aef_snapshot_features(record, dst_profile, config, pca_model)

    features = np.concatenate([s2_features, s1_features, aef_features], axis=0).astype(np.float32)
    if features.shape[0] == 0:
        raise TilePreprocessingError("No feature channels were produced for the tile")
    valid_mask = np.any(np.isfinite(features), axis=0)
    preview = build_preview_from_late_s2_features(s2_features, config) if config.save_input_previews else None
    feature_names = build_feature_names(config)

    metadata: dict[str, Any] = {
        "height": int(features.shape[1]),
        "width": int(features.shape[2]),
        "temporal_feature_mode": config.temporal_feature_mode,
        "feature_names": feature_names,
        "snapshot_selection": {
            "s2": s2_metadata,
            "s1": s1_metadata,
            "aef": aef_metadata,
        },
    }

    fallback_notes = [
        *s2_metadata.get("fallback_decisions", []),
        *s1_metadata.get("fallback_decisions", []),
        *aef_metadata.get("fallback_decisions", []),
    ]

    LOGGER.info(
        "Tile %s selected snapshots | s2=%s | s1=%s | aef=%s | fallbacks=%s",
        record.tile_id,
        {stage_name: (s2_metadata.get(f"selected_{stage_name}_s2") or {}).get("path") for stage_name, *_ in _snapshot_stage_specs(config)},
        {stage_name: (s1_metadata.get(f"selected_{stage_name}_s1") or {}).get("path") for stage_name, *_ in _snapshot_stage_specs(config)},
        {stage_name: (aef_metadata.get(f"selected_{stage_name}_aef") or {}).get("path") for stage_name, *_ in _snapshot_stage_specs(config)},
        fallback_notes if fallback_notes else "none",
    )

    return FeaturePack(
        features=features,
        valid_mask=valid_mask.astype(bool),
        preview=preview,
        feature_names=feature_names,
        metadata=metadata,
    )
