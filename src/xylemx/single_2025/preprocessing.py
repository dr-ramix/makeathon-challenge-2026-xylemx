"""Single-date summer-2025 preprocessing pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from sklearn.decomposition import PCA

from xylemx.config import ExperimentConfig
from xylemx.data.io import (
    AEF_PATTERN,
    S1_PATTERN,
    S2_PATTERN,
    TileRecord,
    get_feature_metadata_path,
    get_feature_path,
    get_ignore_mask_path,
    get_preview_path,
    get_raster_profile,
    get_target_path,
    get_valid_mask_path,
    get_vote_count_path,
    get_weight_map_path,
    load_json,
    read_reprojected_raster,
    save_json,
    scan_tiles,
)
from xylemx.data.splits import split_train_val_tiles
from xylemx.labels.consensus import (
    LabelFusionResult,
    fuse_binary_masks,
    gladl_positive_mask,
    glads2_positive_mask,
    radd_positive_mask,
)
from xylemx.preprocessing.features import AefPcaModel, S2_BAND_NAMES, TilePreprocessingError
from xylemx.preprocessing.normalize import ReservoirPercentileEstimator, RunningChannelStats, clip_array

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SingleSnapshotCandidate:
    """One candidate snapshot aligned to the Sentinel-2 grid."""

    path: Path
    year: int
    month: int
    array: np.ndarray
    valid_mask: np.ndarray
    valid_fraction: float
    source_paths: list[Path]

    def describe(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "year": self.year,
            "month": self.month,
            "valid_fraction": float(self.valid_fraction),
            "source_paths": [str(path) for path in self.source_paths],
        }


@dataclass(slots=True)
class SingleFeaturePack:
    """Engineered per-tile feature tensor for the single-date workflow."""

    features: np.ndarray
    valid_mask: np.ndarray
    preview: np.ndarray | None
    feature_names: list[str]
    metadata: dict[str, Any]


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _parse_match(pattern, path: Path) -> tuple[str, ...]:
    match = pattern.match(path.name)
    if match is None:
        raise ValueError(f"Malformed filename: {path}")
    return match.groups()


def _empty_block(num_channels: int, height: int, width: int) -> np.ndarray:
    if num_channels <= 0:
        return np.empty((0, height, width), dtype=np.float32)
    return np.full((num_channels, height, width), np.nan, dtype=np.float32)


def _observation_valid_fraction(valid_mask: np.ndarray) -> float:
    return float(valid_mask.mean())


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


def _fuse_tile_labels(record: TileRecord, config: ExperimentConfig) -> tuple[LabelFusionResult, dict[str, Any]]:
    dst_profile = get_raster_profile(record.reference_s2_path)
    source_masks: dict[str, np.ndarray] = {}
    source_valid: dict[str, np.ndarray] = {}

    def _load_single_band(path: Path) -> tuple[np.ndarray, np.ndarray]:
        array, valid_mask = read_reprojected_raster(
            path,
            dst_profile,
            resampling=Resampling.nearest,
            out_dtype=np.float32,
        )
        return array[0], valid_mask

    radd_path = Path(record.label_paths["radd"])
    if radd_path.exists():
        radd_raw, radd_valid = _load_single_band(radd_path)
        source_masks["radd"] = radd_positive_mask(radd_raw, mode=config.radd_positive_mode)
        source_valid["radd"] = radd_valid

    glads2_alert_path = Path(record.label_paths["glads2"]["alert"])
    if glads2_alert_path.exists():
        glads2_alert, glads2_valid = _load_single_band(glads2_alert_path)
        source_masks["glads2"] = glads2_positive_mask(glads2_alert, threshold=config.glads2_threshold)
        source_valid["glads2"] = glads2_valid

    gladl_masks: list[np.ndarray] = []
    gladl_valid_masks: list[np.ndarray] = []
    for _year, paths in record.label_paths["gladl"].items():
        alert_path = Path(paths["alert"])
        if not alert_path.exists():
            continue
        alert, valid = _load_single_band(alert_path)
        gladl_masks.append(alert)
        gladl_valid_masks.append(valid)
    if gladl_masks:
        source_masks["gladl"] = gladl_positive_mask(gladl_masks, threshold=config.gladl_threshold)
        source_valid["gladl"] = np.logical_or.reduce(gladl_valid_masks)

    if not source_masks:
        raise RuntimeError(f"Tile {record.tile_id} has no usable weak labels")

    fused = fuse_binary_masks(
        source_masks,
        source_valid,
        method=config.label_fusion,
        soft_vote_threshold=config.soft_vote_threshold,
        ignore_uncertain_single_source_positives=config.ignore_uncertain_single_source_positives,
        ignore_outside_label_extent=config.ignore_outside_label_extent,
        vote_weight_0=config.vote_weight_0,
        vote_weight_1=config.vote_weight_1,
        vote_weight_2=config.vote_weight_2,
        vote_weight_3=config.vote_weight_3,
    )

    usable = ~fused.ignore_mask
    positive_fraction = float(fused.target[usable].mean()) if usable.any() else 0.0
    summary = {
        "positive_fraction": positive_fraction,
        "ignored_fraction": float(fused.ignore_mask.mean()),
        "valid_extent_fraction": float(fused.valid_extent.mean()),
        "mean_vote_count": float(fused.vote_count.mean()),
        "label_sources": sorted(source_masks.keys()),
    }
    return fused, summary


def _collect_s2_summer_candidates(
    record: TileRecord,
    *,
    dst_profile: dict[str, Any],
    config: ExperimentConfig,
    target_year: int,
    summer_months: set[int],
) -> tuple[list[SingleSnapshotCandidate], list[dict[str, Any]]]:
    candidates: list[SingleSnapshotCandidate] = []
    skipped: list[dict[str, Any]] = []

    for path in record.sentinel2_files:
        _, year_raw, month_raw = _parse_match(S2_PATTERN, path)
        year = int(year_raw)
        month = int(month_raw)
        if year != target_year or month not in summer_months:
            continue
        array, valid_mask, valid_fraction = _load_s2_candidate(path, dst_profile, config)
        if array is None or valid_mask is None:
            skipped.append(
                {"path": str(path), "year": year, "month": month, "valid_fraction": float(valid_fraction)}
            )
            continue
        candidates.append(
            SingleSnapshotCandidate(
                path=path,
                year=year,
                month=month,
                array=array,
                valid_mask=valid_mask,
                valid_fraction=float(valid_fraction),
                source_paths=[path],
            )
        )

    return candidates, skipped


def _choose_single_s2_snapshot(
    record: TileRecord,
    *,
    dst_profile: dict[str, Any],
    config: ExperimentConfig,
    target_year: int,
    summer_months: set[int],
) -> tuple[SingleSnapshotCandidate | None, list[dict[str, Any]]]:
    candidates, skipped = _collect_s2_summer_candidates(
        record,
        dst_profile=dst_profile,
        config=config,
        target_year=target_year,
        summer_months=summer_months,
    )
    if not candidates:
        return None, skipped
    chosen = max(candidates, key=lambda item: (item.valid_fraction, item.month, item.path.name))
    return chosen, skipped


def _collect_s1_summer_month_candidates(
    record: TileRecord,
    *,
    dst_profile: dict[str, Any],
    config: ExperimentConfig,
    target_year: int,
    summer_months: set[int],
) -> tuple[list[SingleSnapshotCandidate], list[dict[str, Any]]]:
    grouped: dict[int, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []

    for path in record.sentinel1_files:
        _, year_raw, month_raw, orbit = _parse_match(S1_PATTERN, path)
        year = int(year_raw)
        month = int(month_raw)
        if year != target_year or month not in summer_months:
            continue
        band, valid_mask, valid_fraction = _load_s1_candidate_band(path, dst_profile, config)
        if band is None or valid_mask is None:
            skipped.append(
                {
                    "path": str(path),
                    "year": year,
                    "month": month,
                    "orbit": orbit,
                    "valid_fraction": float(valid_fraction),
                }
            )
            continue
        bucket = grouped.setdefault(month, {"bands": [], "masks": [], "paths": []})
        bucket["bands"].append(band)
        bucket["masks"].append(valid_mask)
        bucket["paths"].append(path)

    month_candidates: list[SingleSnapshotCandidate] = []
    for month, payload in grouped.items():
        stacked = np.stack(payload["bands"], axis=0).astype(np.float32)
        finite = np.isfinite(stacked)
        value_sum = np.nansum(stacked, axis=0).astype(np.float32)
        value_count = finite.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            averaged = (value_sum / np.maximum(value_count, 1)).astype(np.float32)
        valid = np.logical_or.reduce(payload["masks"])
        averaged[~valid] = np.nan
        month_candidates.append(
            SingleSnapshotCandidate(
                path=payload["paths"][0],
                year=target_year,
                month=int(month),
                array=averaged[None, ...],
                valid_mask=valid,
                valid_fraction=float(valid.mean()),
                source_paths=sorted(payload["paths"]),
            )
        )

    return month_candidates, skipped


def _choose_single_s1_snapshot(
    record: TileRecord,
    *,
    dst_profile: dict[str, Any],
    config: ExperimentConfig,
    target_year: int,
    summer_months: set[int],
    preferred_month: int | None,
) -> tuple[SingleSnapshotCandidate | None, list[dict[str, Any]]]:
    month_candidates, skipped = _collect_s1_summer_month_candidates(
        record,
        dst_profile=dst_profile,
        config=config,
        target_year=target_year,
        summer_months=summer_months,
    )

    if not month_candidates:
        return None, skipped

    if preferred_month is None:
        chosen = max(month_candidates, key=lambda item: (item.valid_fraction, item.month, item.path.name))
        return chosen, skipped

    chosen = max(
        month_candidates,
        key=lambda item: (
            int(item.month == preferred_month),
            item.valid_fraction,
            -abs(item.month - preferred_month),
            item.month,
            item.path.name,
        ),
    )
    return chosen, skipped


def _nanmedian_composite(
    candidates: list[SingleSnapshotCandidate],
) -> tuple[np.ndarray, np.ndarray]:
    if not candidates:
        raise ValueError("Expected at least one candidate for compositing")
    stacked = np.stack([candidate.array for candidate in candidates], axis=0).astype(np.float32)
    valid_stack = np.stack([candidate.valid_mask for candidate in candidates], axis=0)
    with np.errstate(invalid="ignore"):
        composite = np.nanmedian(stacked, axis=0).astype(np.float32)
    observation_count = valid_stack.sum(axis=0).astype(np.float32)
    composite[:, observation_count <= 0] = np.nan
    return composite, observation_count


def _find_aef_year_path(record: TileRecord, target_year: int) -> Path | None:
    for path in record.aef_files:
        _, year_raw = _parse_match(AEF_PATTERN, path)
        if int(year_raw) == target_year:
            return path
    return None


def _fit_single_year_aef_pca(
    records: dict[str, TileRecord],
    train_tile_ids: list[str],
    *,
    target_year: int,
    config: ExperimentConfig,
) -> AefPcaModel | None:
    if not config.use_aef_features or config.aef_pca_dim <= 0:
        return None

    rng = np.random.default_rng(config.seed)
    samples: list[np.ndarray] = []
    sampled_rasters = 0
    for tile_id in train_tile_ids:
        path = _find_aef_year_path(records[tile_id], target_year)
        if path is None:
            continue
        try:
            with rasterio.open(path) as src:
                source = src.read(masked=True).astype(np.float32)
                source_data = np.asarray(source.filled(np.nan), dtype=np.float32)
                valid_mask = np.isfinite(source_data).all(axis=0)
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
        sampled_rasters += 1

    if not samples:
        if config.require_aef:
            raise RuntimeError("AEF features are required but no valid 2025 AEF samples were found")
        LOGGER.warning("No valid 2025 AEF samples found; disabling AEF channels for this preprocessing run")
        return None

    sample_matrix = np.concatenate(samples, axis=0)
    pca = PCA(n_components=config.aef_pca_dim, random_state=config.seed)
    pca.fit(sample_matrix)
    LOGGER.info(
        "Fitted single-2025 AEF PCA from %d raster(s) with sample matrix shape %s",
        sampled_rasters,
        sample_matrix.shape,
    )
    return AefPcaModel(
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
    )


def _load_projected_aef_snapshot(
    path: Path,
    *,
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
                pca_model.components.shape[0],
                src.height,
                src.width,
            )
            projected[:, ~source_valid] = np.nan

            destination = np.zeros(
                (projected.shape[0], int(dst_profile["height"]), int(dst_profile["width"])),
                dtype=np.float32,
            )
            for channel_index in range(projected.shape[0]):
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


def _build_preview_from_single_s2(s2_features: np.ndarray) -> np.ndarray | None:
    if s2_features.shape[0] < len(S2_BAND_NAMES):
        return None
    rgb = np.stack([s2_features[3], s2_features[2], s2_features[1]], axis=0).astype(np.float32)
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


def _build_feature_names(
    *,
    include_s1: bool,
    aef_dim: int,
    add_observation_count_channels: bool,
) -> list[str]:
    names = [f"s2_summer2025_{band}" for band in S2_BAND_NAMES]
    if include_s1:
        names.append("s1_summer2025")
    if aef_dim > 0:
        names.extend(f"aef_summer2025_pc{index + 1:02d}" for index in range(aef_dim))
    if add_observation_count_channels:
        names.append("s2_summer2025_obs_fraction")
        if include_s1:
            names.append("s1_summer2025_obs_fraction")
    return names


def _cache_feature_pack(tile_id: str, split: str, preprocessing_dir: Path, feature_pack: SingleFeaturePack) -> None:
    _save_array(get_feature_path(preprocessing_dir, split, tile_id), feature_pack.features.astype(np.float16))
    save_json(get_feature_metadata_path(preprocessing_dir, split, tile_id), feature_pack.metadata)
    _save_array(get_valid_mask_path(preprocessing_dir, split, tile_id), feature_pack.valid_mask.astype(np.bool_))
    if feature_pack.preview is not None:
        _save_array(get_preview_path(preprocessing_dir, split, tile_id), feature_pack.preview.astype(np.uint8))


def _compute_normalization_stats(
    train_tile_ids: list[str],
    preprocessing_dir: Path,
    *,
    feature_names: list[str],
    config: ExperimentConfig,
) -> dict[str, Any]:
    first_feature = np.load(get_feature_path(preprocessing_dir, "train", train_tile_ids[0])).astype(np.float32)
    num_channels = int(first_feature.shape[0])

    if config.clip_features:
        estimator = ReservoirPercentileEstimator(
            num_channels=num_channels,
            max_samples_per_channel=config.clip_num_samples_per_tile,
            seed=config.seed,
        )
        for tile_id in train_tile_ids:
            feature = np.load(get_feature_path(preprocessing_dir, "train", tile_id)).astype(np.float32)
            valid_mask = np.load(get_valid_mask_path(preprocessing_dir, "train", tile_id)).astype(bool)
            estimator.update(feature, valid_mask=valid_mask)
        clip_lower, clip_upper = estimator.finalize(config.clip_lower_percentile, config.clip_upper_percentile)
    else:
        clip_lower = [-1e12] * num_channels
        clip_upper = [1e12] * num_channels

    stats = RunningChannelStats(num_channels=num_channels)
    for tile_id in train_tile_ids:
        feature = np.load(get_feature_path(preprocessing_dir, "train", tile_id)).astype(np.float32)
        valid_mask = np.load(get_valid_mask_path(preprocessing_dir, "train", tile_id)).astype(bool)
        clipped = clip_array(feature, clip_lower, clip_upper)
        stats.update(clipped, valid_mask=valid_mask)

    payload = stats.finalize(
        feature_names,
        clip_lower,
        clip_upper,
        min_std=config.min_normalization_std,
    )
    payload["normalization"] = config.normalization
    payload["clip_features"] = bool(config.clip_features)
    payload["clip_lower_percentile"] = float(config.clip_lower_percentile)
    payload["clip_upper_percentile"] = float(config.clip_upper_percentile)
    payload["min_normalization_std"] = float(config.min_normalization_std)
    return payload


def _build_single_feature_pack(
    record: TileRecord,
    *,
    config: ExperimentConfig,
    target_year: int,
    summer_months: set[int],
    strict_summer_only: bool,
    aef_pca_model: AefPcaModel | None,
    aef_output_dim: int,
    s2_selection_mode: str,
    s1_selection_mode: str,
    add_observation_count_channels: bool,
) -> SingleFeaturePack:
    dst_profile = get_raster_profile(record.reference_s2_path)
    height = int(dst_profile["height"])
    width = int(dst_profile["width"])

    if s2_selection_mode not in {"best_single", "summer_median_composite"}:
        raise ValueError(f"Unsupported s2_selection_mode: {s2_selection_mode}")
    if s1_selection_mode not in {"best_single", "summer_median_composite"}:
        raise ValueError(f"Unsupported s1_selection_mode: {s1_selection_mode}")

    selected_s2: SingleSnapshotCandidate | None
    selected_s2_month: int | None
    s2_candidates: list[SingleSnapshotCandidate]
    s2_obs_total: int
    s2_obs_count: np.ndarray

    if s2_selection_mode == "summer_median_composite":
        s2_candidates, skipped_s2 = _collect_s2_summer_candidates(
            record,
            dst_profile=dst_profile,
            config=config,
            target_year=target_year,
            summer_months=summer_months,
        )
        if s2_candidates:
            s2_features, s2_obs_count = _nanmedian_composite(s2_candidates)
            selected_s2 = max(s2_candidates, key=lambda item: (item.valid_fraction, item.month, item.path.name))
            selected_s2_month = int(selected_s2.month)
            s2_obs_total = len(s2_candidates)
        else:
            s2_features = _empty_block(len(S2_BAND_NAMES), height, width)
            selected_s2 = None
            selected_s2_month = None
            s2_obs_total = 1
            s2_obs_count = np.zeros((height, width), dtype=np.float32)
    else:
        selected_s2, skipped_s2 = _choose_single_s2_snapshot(
            record,
            dst_profile=dst_profile,
            config=config,
            target_year=target_year,
            summer_months=summer_months,
        )
        s2_candidates = [selected_s2] if selected_s2 is not None else []
        if selected_s2 is None:
            s2_features = _empty_block(len(S2_BAND_NAMES), height, width)
            selected_s2_month = None
            s2_obs_total = 1
            s2_obs_count = np.zeros((height, width), dtype=np.float32)
        else:
            s2_features = selected_s2.array
            selected_s2_month = int(selected_s2.month)
            s2_obs_total = 1
            s2_obs_count = selected_s2.valid_mask.astype(np.float32)

    if selected_s2 is None and strict_summer_only:
        if config.require_s2 and config.skip_tile_if_required_modality_missing:
            raise TilePreprocessingError("Missing usable Sentinel-2 summer 2025 snapshot")
        LOGGER.warning("Tile %s has no usable Sentinel-2 summer 2025 snapshot", record.tile_id)

    if config.use_s1_features:
        selected_s1: SingleSnapshotCandidate | None
        s1_obs_total: int
        s1_obs_count: np.ndarray
        s1_candidates: list[SingleSnapshotCandidate]
        if s1_selection_mode == "summer_median_composite":
            s1_candidates, skipped_s1 = _collect_s1_summer_month_candidates(
                record,
                dst_profile=dst_profile,
                config=config,
                target_year=target_year,
                summer_months=summer_months,
            )
            if s1_candidates:
                s1_features, s1_obs_count = _nanmedian_composite(s1_candidates)
                selected_s1 = max(
                    s1_candidates,
                    key=lambda item: (
                        int(selected_s2_month is not None and item.month == selected_s2_month),
                        item.valid_fraction,
                        item.month,
                        item.path.name,
                    ),
                )
                s1_obs_total = len(s1_candidates)
            else:
                selected_s1 = None
                s1_features = _empty_block(1, height, width)
                s1_obs_total = 1
                s1_obs_count = np.zeros((height, width), dtype=np.float32)
        else:
            selected_s1, skipped_s1 = _choose_single_s1_snapshot(
                record,
                dst_profile=dst_profile,
                config=config,
                target_year=target_year,
                summer_months=summer_months,
                preferred_month=selected_s2_month,
            )
            s1_candidates = [selected_s1] if selected_s1 is not None else []
            if selected_s1 is None:
                s1_features = _empty_block(1, height, width)
                s1_obs_total = 1
                s1_obs_count = np.zeros((height, width), dtype=np.float32)
            else:
                s1_features = selected_s1.array
                s1_obs_total = 1
                s1_obs_count = selected_s1.valid_mask.astype(np.float32)

        if selected_s1 is None and config.require_s1 and config.skip_tile_if_required_modality_missing:
            raise TilePreprocessingError("Missing usable Sentinel-1 summer 2025 snapshot")
    else:
        selected_s1 = None
        s1_candidates = []
        skipped_s1 = []
        s1_features = _empty_block(0, height, width)
        s1_obs_total = 1
        s1_obs_count = np.zeros((height, width), dtype=np.float32)

    if aef_output_dim > 0 and config.use_aef_features and aef_pca_model is not None:
        aef_path = _find_aef_year_path(record, target_year)
        skipped_aef: list[dict[str, Any]] = []
        if aef_path is None:
            if config.require_aef and config.skip_tile_if_required_modality_missing:
                raise TilePreprocessingError("Missing 2025 AEF embedding raster")
            aef_features = _empty_block(aef_output_dim, height, width)
            selected_aef: dict[str, Any] | None = None
        else:
            projected, _valid, valid_fraction = _load_projected_aef_snapshot(
                aef_path,
                dst_profile=dst_profile,
                config=config,
                pca_model=aef_pca_model,
            )
            if projected is None:
                if config.require_aef and config.skip_tile_if_required_modality_missing:
                    raise TilePreprocessingError("Missing usable 2025 AEF embedding raster")
                skipped_aef.append({"path": str(aef_path), "year": target_year, "valid_fraction": float(valid_fraction)})
                aef_features = _empty_block(aef_output_dim, height, width)
                selected_aef = None
            else:
                aef_features = projected
                selected_aef = {"year": target_year, "path": str(aef_path), "valid_fraction": float(valid_fraction)}
    else:
        aef_features = _empty_block(0, height, width)
        selected_aef = None
        skipped_aef = []

    extra_channels: list[np.ndarray] = []
    if add_observation_count_channels:
        s2_obs_fraction = np.clip(s2_obs_count / max(float(s2_obs_total), 1.0), 0.0, 1.0).astype(np.float32)
        extra_channels.append(s2_obs_fraction[None, ...])
        if config.use_s1_features:
            s1_obs_fraction = np.clip(s1_obs_count / max(float(s1_obs_total), 1.0), 0.0, 1.0).astype(np.float32)
            extra_channels.append(s1_obs_fraction[None, ...])

    features = np.concatenate([s2_features, s1_features, aef_features, *extra_channels], axis=0).astype(np.float32)
    if features.shape[0] == 0:
        raise TilePreprocessingError("No feature channels were produced for the tile")
    valid_mask = np.any(np.isfinite(features), axis=0)
    if not valid_mask.any():
        raise TilePreprocessingError("No valid pixels available after feature construction")

    preview = _build_preview_from_single_s2(s2_features) if config.save_input_previews else None
    feature_names = _build_feature_names(
        include_s1=bool(config.use_s1_features),
        aef_dim=aef_output_dim,
        add_observation_count_channels=add_observation_count_channels,
    )
    metadata: dict[str, Any] = {
        "height": int(features.shape[1]),
        "width": int(features.shape[2]),
        "temporal_feature_mode": "single_2025_summer",
        "feature_names": feature_names,
        "single_snapshot_selection": {
            "target_year": int(target_year),
            "summer_months": sorted(int(value) for value in summer_months),
            "s2": {
                "selection_mode": s2_selection_mode,
                "num_candidates": len(s2_candidates),
                "selected": selected_s2.describe() if selected_s2 is not None else None,
                "skipped_bad_snapshots": skipped_s2,
            },
            "s1": {
                "selection_mode": s1_selection_mode,
                "num_candidates": len(s1_candidates),
                "selected": selected_s1.describe() if selected_s1 is not None else None,
                "skipped_bad_snapshots": skipped_s1,
            },
            "aef": {
                "selected": selected_aef,
                "skipped_bad_snapshots": skipped_aef,
                "pca_dim": int(aef_output_dim),
            },
            "observation_count_channels": bool(add_observation_count_channels),
        },
    }
    return SingleFeaturePack(
        features=features,
        valid_mask=valid_mask.astype(bool),
        preview=preview,
        feature_names=feature_names,
        metadata=metadata,
    )


def run_single_2025_preprocessing(
    config: ExperimentConfig,
    preprocessing_dir: str | Path,
    *,
    target_year: int = 2025,
    summer_months: tuple[int, ...] = (6, 7, 8),
    strict_summer_only: bool = True,
    s2_selection_mode: str = "best_single",
    s1_selection_mode: str = "best_single",
    add_observation_count_channels: bool = False,
) -> dict[str, Any]:
    """Build reusable preprocessing artifacts for the single-image 2025 workflow."""

    preprocessing_dir = Path(preprocessing_dir)
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    summary_path = preprocessing_dir / "summary.json"
    if config.cache_preprocessed and not config.rebuild_cache and summary_path.exists():
        LOGGER.info("Reusing cached single-2025 preprocessing artifacts from %s", preprocessing_dir)
        return load_json(summary_path)

    summer_months_set = {int(month) for month in summer_months}
    if not summer_months_set:
        raise ValueError("summer_months must include at least one month")

    LOGGER.info(
        (
            "Single-2025 preprocessing | target_year=%d summer_months=%s strict_summer_only=%s "
            "s2_selection_mode=%s s1_selection_mode=%s add_observation_count_channels=%s"
        ),
        target_year,
        sorted(summer_months_set),
        strict_summer_only,
        s2_selection_mode,
        s1_selection_mode,
        add_observation_count_channels,
    )

    train_records = scan_tiles(config.data_root, "train")
    test_records = scan_tiles(config.data_root, "test")

    label_summaries: dict[str, dict[str, Any]] = {}
    positive_fractions: dict[str, float] = {}
    LOGGER.info("Fusing weak labels for %d train tiles", len(train_records))
    for tile_id, record in train_records.items():
        fused, summary = _fuse_tile_labels(record, config)
        label_summaries[tile_id] = summary
        positive_fractions[tile_id] = summary["positive_fraction"]
        _save_array(get_target_path(preprocessing_dir, tile_id), fused.target.astype(np.float32))
        _save_array(get_ignore_mask_path(preprocessing_dir, tile_id), fused.ignore_mask.astype(np.bool_))
        _save_array(get_weight_map_path(preprocessing_dir, tile_id), fused.weight_map.astype(np.float32))
        _save_array(get_vote_count_path(preprocessing_dir, tile_id), fused.vote_count.astype(np.uint8))

    initial_train_tile_ids, _initial_val_tile_ids = split_train_val_tiles(
        sorted(train_records.keys()),
        val_ratio=config.val_ratio,
        seed=config.split_seed,
        positive_fractions=positive_fractions,
        stratify=config.stratify_split,
    )

    aef_pca_model = _fit_single_year_aef_pca(
        train_records,
        initial_train_tile_ids,
        target_year=target_year,
        config=config,
    )
    if aef_pca_model is None:
        save_json(preprocessing_dir / "aef_pca.json", {"disabled": True, "aef_pca_dim": 0})
        aef_output_dim = 0
    else:
        save_json(preprocessing_dir / "aef_pca.json", aef_pca_model.to_payload())
        aef_output_dim = int(config.aef_pca_dim)

    tile_metadata: dict[str, dict[str, Any]] = {"train": {}, "test": {}}
    skipped_tiles: dict[str, list[str]] = {"train": [], "test": []}
    feature_names = _build_feature_names(
        include_s1=bool(config.use_s1_features),
        aef_dim=aef_output_dim,
        add_observation_count_channels=add_observation_count_channels,
    )

    processed_train_tile_ids: list[str] = []
    processed_test_tile_ids: list[str] = []

    for split, records in (("train", train_records), ("test", test_records)):
        start_time = time.perf_counter()
        tile_ids = sorted(records.keys())
        for index, tile_id in enumerate(tile_ids, start=1):
            try:
                feature_pack = _build_single_feature_pack(
                    records[tile_id],
                    config=config,
                    target_year=target_year,
                    summer_months=summer_months_set,
                    strict_summer_only=strict_summer_only,
                    aef_pca_model=aef_pca_model,
                    aef_output_dim=aef_output_dim,
                    s2_selection_mode=s2_selection_mode,
                    s1_selection_mode=s1_selection_mode,
                    add_observation_count_channels=add_observation_count_channels,
                )
            except TilePreprocessingError as exc:
                skipped_tiles[split].append(tile_id)
                LOGGER.warning("Skipping %s tile %s in single-2025 preprocessing: %s", split, tile_id, exc)
                continue

            _cache_feature_pack(tile_id, split, preprocessing_dir, feature_pack)
            if split == "train":
                processed_train_tile_ids.append(tile_id)
                tile_metadata["train"][tile_id] = {
                    **records[tile_id].to_dict(),
                    **feature_pack.metadata,
                    **label_summaries[tile_id],
                }
            else:
                processed_test_tile_ids.append(tile_id)
                tile_metadata["test"][tile_id] = {
                    **records[tile_id].to_dict(),
                    **feature_pack.metadata,
                }
            elapsed = time.perf_counter() - start_time
            LOGGER.info(
                "Finished single-2025 features for %s tile %s (%d/%d) | elapsed=%.1fs",
                split,
                tile_id,
                index,
                len(tile_ids),
                elapsed,
            )

    processed_train_tile_ids = sorted(processed_train_tile_ids)
    if not processed_train_tile_ids:
        raise RuntimeError("No train tiles remained after single-2025 preprocessing")

    filtered_positive_fractions = {
        tile_id: positive_fractions[tile_id]
        for tile_id in processed_train_tile_ids
        if tile_id in positive_fractions
    }
    train_tile_ids, val_tile_ids = split_train_val_tiles(
        processed_train_tile_ids,
        val_ratio=config.val_ratio,
        seed=config.split_seed,
        positive_fractions=filtered_positive_fractions,
        stratify=config.stratify_split,
    )

    normalization_stats = _compute_normalization_stats(
        train_tile_ids,
        preprocessing_dir,
        feature_names=feature_names,
        config=config,
    )

    save_json(preprocessing_dir / "feature_spec.json", {"feature_names": feature_names})
    save_json(preprocessing_dir / "normalization_stats.json", normalization_stats)
    save_json(preprocessing_dir / "train_tiles.json", train_tile_ids)
    save_json(preprocessing_dir / "val_tiles.json", val_tile_ids)
    save_json(preprocessing_dir / "tile_metadata.json", tile_metadata)

    summary = {
        "config": asdict(config),
        "preprocessing_dir": str(preprocessing_dir),
        "pipeline": "single_2025_summer",
        "target_year": int(target_year),
        "summer_months": sorted(int(value) for value in summer_months_set),
        "strict_summer_only": bool(strict_summer_only),
        "s2_selection_mode": s2_selection_mode,
        "s1_selection_mode": s1_selection_mode,
        "add_observation_count_channels": bool(add_observation_count_channels),
        "num_train_tiles": len(train_records),
        "num_test_tiles": len(test_records),
        "num_cached_train_tiles": len(processed_train_tile_ids),
        "num_cached_test_tiles": len(processed_test_tile_ids),
        "train_tiles": train_tile_ids,
        "val_tiles": val_tile_ids,
        "skipped_tiles": skipped_tiles,
        "feature_names": feature_names,
        "positive_fractions": filtered_positive_fractions,
        "pca_explained_variance_ratio": (
            aef_pca_model.explained_variance_ratio.astype(np.float32).tolist() if aef_pca_model is not None else []
        ),
    }
    save_json(summary_path, summary)
    return summary
