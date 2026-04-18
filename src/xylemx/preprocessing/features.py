"""Multimodal feature engineering for tile-based deforestation segmentation."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from sklearn.decomposition import PCA

from xylemx.config import ExperimentConfig
from xylemx.data.io import AEF_PATTERN, S1_PATTERN, S2_PATTERN, TileRecord, get_raster_profile, read_reprojected_raster

LOGGER = logging.getLogger(__name__)

S2_BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
S2_MEAN_CHANNELS = [f"s2_mean_{band}" for band in S2_BAND_NAMES]
S2_STD_CHANNELS = [f"s2_std_{band}" for band in S2_BAND_NAMES]
S2_DELTA_CHANNELS = [f"s2_delta_{band}" for band in S2_BAND_NAMES]
S1_CHANNELS = ["s1_mean", "s1_std", "s1_delta"]


@dataclass(slots=True)
class FeaturePack:
    """Engineered per-tile features and lightweight metadata."""

    features: np.ndarray
    valid_mask: np.ndarray
    preview: np.ndarray | None
    feature_names: list[str]
    metadata: dict[str, int | float]


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
class TemporalRaster:
    """One valid temporal observation for a modality."""

    path: Path
    year: int
    month: int | None
    weight: float
    valid_mask: np.ndarray


def build_feature_names(aef_pca_dim: int) -> list[str]:
    """Return the stable feature-channel ordering."""

    return S2_MEAN_CHANNELS + S2_STD_CHANNELS + S2_DELTA_CHANNELS + S1_CHANNELS + [
        f"aef_mean_pc{index + 1:02d}" for index in range(aef_pca_dim)
    ]


def _parse_match(pattern, path: Path) -> tuple[str, ...]:
    match = pattern.match(path.name)
    if match is None:
        raise ValueError(f"Malformed filename: {path}")
    return match.groups()


def _read_aligned_band(
    path: Path,
    dst_profile: dict,
    *,
    band_index: int,
    resampling: Resampling,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        masked_band = src.read(band_index, masked=True)
        src_nodata = src.nodata
        if src_nodata is None or (isinstance(src_nodata, float) and np.isnan(src_nodata)):
            src_nodata = 0

        destination = np.zeros((int(dst_profile["height"]), int(dst_profile["width"])), dtype=np.float32)
        reproject(
            source=np.asarray(masked_band.filled(src_nodata), dtype=np.float32),
            destination=destination,
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
    destination[~valid_mask] = np.nan
    return destination.astype(np.float32), valid_mask


def _observation_valid_fraction(valid_mask: np.ndarray, array: np.ndarray | None = None) -> float:
    usable = valid_mask.copy()
    if array is not None:
        if array.ndim == 3:
            usable &= np.isfinite(array).all(axis=0)
        else:
            usable &= np.isfinite(array)
    return float(usable.mean())


def _collect_s2_observations(record: TileRecord, dst_profile: dict, config: ExperimentConfig) -> list[TemporalRaster]:
    observations: list[TemporalRaster] = []
    for path in record.sentinel2_files:
        year_raw, month_raw = _parse_match(S2_PATTERN, path)[1:]
        year = int(year_raw)
        month = int(month_raw)
        array, valid_mask = read_reprojected_raster(path, dst_profile, resampling=Resampling.bilinear, out_dtype=np.float32)
        valid_mask = valid_mask & np.isfinite(array).all(axis=0) & np.any(array > 0, axis=0)
        if _observation_valid_fraction(valid_mask) < config.min_valid_pixels_fraction_per_month:
            continue
        observations.append(
            TemporalRaster(
                path=path,
                year=year,
                month=month,
                weight=float(config.year_weights().get(year, 1.0)),
                valid_mask=valid_mask,
            )
        )
    return observations


def _collect_s1_observations(
    record: TileRecord,
    dst_profile: dict,
    config: ExperimentConfig,
) -> dict[str, list[TemporalRaster]]:
    grouped = {"ascending": [], "descending": []}
    for path in record.sentinel1_files:
        year_raw, month_raw, orbit = _parse_match(S1_PATTERN, path)[1:]
        year = int(year_raw)
        month = int(month_raw)
        band, valid_mask = _read_aligned_band(path, dst_profile, band_index=1, resampling=Resampling.bilinear)
        valid_mask = valid_mask & np.isfinite(band)
        if _observation_valid_fraction(valid_mask) < config.min_valid_pixels_fraction_per_month:
            continue
        grouped[orbit].append(
            TemporalRaster(
                path=path,
                year=year,
                month=month,
                weight=float(config.year_weights().get(year, 1.0)),
                valid_mask=valid_mask,
            )
        )
    return grouped


def _collect_aef_paths(record: TileRecord) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for path in record.aef_files:
        year_raw = _parse_match(AEF_PATTERN, path)[1]
        pairs.append((int(year_raw), path))
    return sorted(pairs, key=lambda item: item[0])


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


def aggregate_temporal_multiband_stack(
    stack: np.ndarray,
    weights: np.ndarray,
    *,
    min_observations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a temporal stack of shape [T, C, H, W]."""

    finite = np.isfinite(stack)
    weighted_sum = np.nansum(stack * weights[:, None, None, None], axis=0)
    weight_total = np.sum(finite * weights[:, None, None, None], axis=0, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted_mean = weighted_sum / np.maximum(weight_total, 1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            std = np.nanstd(stack, axis=0).astype(np.float32)
            delta = (
                np.nanpercentile(stack, 75.0, axis=0).astype(np.float32)
                - np.nanpercentile(stack, 25.0, axis=0).astype(np.float32)
            )

    valid_counts = finite.sum(axis=0)
    weighted_mean[valid_counts < min_observations] = np.nan
    std[valid_counts < min_observations] = np.nan
    delta[valid_counts < min_observations] = np.nan
    return weighted_mean.astype(np.float32), std, delta.astype(np.float32)


def _aggregate_temporal_band(
    observations: Iterable[TemporalRaster],
    dst_profile: dict,
    *,
    band_index: int,
    resampling: Resampling,
    min_observations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_list = list(observations)
    height = int(dst_profile["height"])
    width = int(dst_profile["width"])
    if not obs_list:
        nan = np.full((height, width), np.nan, dtype=np.float32)
        return nan, nan.copy(), nan.copy()

    stack = np.full((len(obs_list), height, width), np.nan, dtype=np.float32)
    weights = np.asarray([obs.weight for obs in obs_list], dtype=np.float32)
    for obs_index, obs in enumerate(obs_list):
        band, band_valid = _read_aligned_band(obs.path, dst_profile, band_index=band_index, resampling=resampling)
        valid = obs.valid_mask & band_valid & np.isfinite(band)
        stack[obs_index] = np.where(valid, band, np.nan)

    return aggregate_temporal_stack(stack, weights, min_observations=min_observations)


def _aggregate_s2_features(record: TileRecord, dst_profile: dict, config: ExperimentConfig) -> tuple[np.ndarray, dict[str, int]]:
    stacks: list[np.ndarray] = []
    weights: list[float] = []
    for path in record.sentinel2_files:
        year_raw, month_raw = _parse_match(S2_PATTERN, path)[1:]
        year = int(year_raw)
        _month = int(month_raw)
        array, valid_mask = read_reprojected_raster(path, dst_profile, resampling=Resampling.bilinear, out_dtype=np.float32)
        valid = valid_mask & np.isfinite(array).all(axis=0) & np.any(array > 0, axis=0)
        if _observation_valid_fraction(valid) < config.min_valid_pixels_fraction_per_month:
            continue
        array[:, ~valid] = np.nan
        stacks.append(array.astype(np.float32, copy=False))
        weights.append(float(config.year_weights().get(year, 1.0)))

    if not stacks:
        height = int(dst_profile["height"])
        width = int(dst_profile["width"])
        nan_block = np.full((len(S2_BAND_NAMES), height, width), np.nan, dtype=np.float32)
        features = np.concatenate([nan_block, nan_block.copy(), nan_block.copy()], axis=0)
        return features, {"num_s2_observations": 0}

    stack = np.stack(stacks, axis=0)
    mean, std, delta = aggregate_temporal_multiband_stack(
        stack,
        np.asarray(weights, dtype=np.float32),
        min_observations=config.min_observations_per_pixel,
    )
    features = np.concatenate([mean, std, delta], axis=0).astype(np.float32)
    return features, {"num_s2_observations": len(stacks)}


def _aggregate_s1_orbit_features(
    observations: list[TemporalRaster],
    dst_profile: dict,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _aggregate_temporal_band(
        observations,
        dst_profile,
        band_index=1,
        resampling=Resampling.bilinear,
        min_observations=config.min_observations_per_pixel,
    )


def _aggregate_s1_features(record: TileRecord, dst_profile: dict, config: ExperimentConfig) -> tuple[np.ndarray, dict[str, int]]:
    grouped = _collect_s1_observations(record, dst_profile, config)
    orbit_features: list[np.ndarray] = []
    metadata = {
        "num_s1_ascending_observations": len(grouped["ascending"]),
        "num_s1_descending_observations": len(grouped["descending"]),
    }
    for orbit in ["ascending", "descending"]:
        mean, std, delta = _aggregate_s1_orbit_features(grouped[orbit], dst_profile, config)
        orbit_features.append(np.stack([mean, std, delta], axis=0))

    stacked = np.stack(orbit_features, axis=0)
    with np.errstate(all="ignore"):
        averaged = np.nanmean(stacked, axis=0).astype(np.float32)
    return averaged, metadata


def fit_aef_pca_model(
    records: dict[str, TileRecord],
    train_tile_ids: list[str],
    config: ExperimentConfig,
) -> AefPcaModel:
    """Fit PCA on train tiles only using sampled AEF pixels."""

    samples: list[np.ndarray] = []
    rng = np.random.default_rng(config.seed)
    total_rasters = 0
    for tile_id in train_tile_ids:
        record = records[tile_id]
        LOGGER.info("Collecting AEF PCA samples for tile %s", tile_id)
        for _year, path in _collect_aef_paths(record):
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


def _aggregate_aef_features(
    record: TileRecord,
    dst_profile: dict,
    config: ExperimentConfig,
    pca_model: AefPcaModel,
) -> tuple[np.ndarray, dict[str, int]]:
    height = int(dst_profile["height"])
    width = int(dst_profile["width"])
    numerator = np.zeros((config.aef_pca_dim, height, width), dtype=np.float32)
    denominator = np.zeros((height, width), dtype=np.float32)
    usable_years = 0

    for year, path in _collect_aef_paths(record):
        try:
            with rasterio.open(path) as src:
                source = src.read(masked=True).astype(np.float32)
                source_data = np.asarray(source.filled(np.nan), dtype=np.float32)
                source_valid = np.isfinite(source_data).all(axis=0)
                if _observation_valid_fraction(source_valid) < config.min_valid_pixels_fraction_per_month:
                    continue

                flat = source_data.reshape(source_data.shape[0], -1).T
                projected = pca_model.transform(np.nan_to_num(flat, nan=0.0)).T.reshape(
                    config.aef_pca_dim,
                    src.height,
                    src.width,
                )
                projected[:, ~source_valid] = np.nan

                destination = np.zeros(
                    (config.aef_pca_dim, int(dst_profile["height"]), int(dst_profile["width"])),
                    dtype=np.float32,
                )
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
            LOGGER.warning("Skipping unreadable AEF raster for %s: %s (%s)", record.tile_id, path, exc)
            continue

        valid = valid_destination > 0
        if _observation_valid_fraction(valid) < config.min_valid_pixels_fraction_per_month:
            continue

        projected = destination
        projected[:, ~valid] = np.nan

        weight = float(config.year_weights().get(year, 1.0))
        numerator += np.where(valid[None, ...], projected * weight, 0.0)
        denominator += valid.astype(np.float32) * weight
        usable_years += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = numerator / np.maximum(denominator[None, ...], 1e-6)
    mean[:, denominator < config.min_observations_per_pixel] = np.nan
    return mean.astype(np.float32), {"num_aef_observations": usable_years}


def build_preview_from_s2_features(s2_features: np.ndarray) -> np.ndarray | None:
    """Create a lightweight RGB-like preview from S2 weighted-mean channels."""

    if s2_features.shape[0] < 4:
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
        preview[channel_index] = np.round(scaled * 255.0).astype(np.uint8)
    return preview


def build_multimodal_feature_pack(record: TileRecord, config: ExperimentConfig, pca_model: AefPcaModel) -> FeaturePack:
    """Build the final 47-channel feature tensor for one tile."""

    dst_profile = get_raster_profile(record.reference_s2_path)
    s2_features, s2_metadata = _aggregate_s2_features(record, dst_profile, config)
    s1_features, s1_metadata = _aggregate_s1_features(record, dst_profile, config)
    aef_features, aef_metadata = _aggregate_aef_features(record, dst_profile, config, pca_model)

    feature_names = build_feature_names(config.aef_pca_dim)
    features = np.concatenate([s2_features, s1_features, aef_features], axis=0).astype(np.float32)
    valid_mask = np.any(np.isfinite(features), axis=0)
    preview = build_preview_from_s2_features(s2_features[: len(S2_BAND_NAMES)])
    metadata = {
        **s2_metadata,
        **s1_metadata,
        **aef_metadata,
        "height": int(features.shape[1]),
        "width": int(features.shape[2]),
    }
    return FeaturePack(
        features=features,
        valid_mask=valid_mask.astype(bool),
        preview=preview,
        feature_names=feature_names,
        metadata=metadata,
    )
