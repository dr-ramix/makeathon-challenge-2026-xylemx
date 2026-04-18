"""Temporal preprocessing for joint segmentation and event-time prediction."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from xylemx.data.io import (
    AEF_PATTERN,
    S1_PATTERN,
    S2_PATTERN,
    TileRecord,
    get_raster_profile,
    load_json,
    read_reprojected_raster,
    save_json,
    scan_tiles,
)
from xylemx.data.splits import split_train_val_tiles
from xylemx.labels.consensus import fuse_binary_masks
from xylemx.labels.decode import DecodedLabel, combine_decoded_labels, decode_gladl_year, decode_glads2_array, decode_radd_array
from xylemx.preprocessing.features import AefPcaModel, S2_BAND_NAMES
from xylemx.temporal.config import TemporalPreprocessingConfig
from xylemx.temporal.io import (
    get_temporal_ignore_mask_path,
    get_temporal_input_path,
    get_temporal_mask_target_path,
    get_temporal_time_target_path,
    get_temporal_valid_mask_path,
    get_temporal_weight_map_path,
)
from xylemx.temporal.labels import (
    TimeBinSpec,
    build_time_bin_spec,
    dates_to_bin_indices,
    filter_dates_to_range,
    iter_months,
    merge_event_dates,
)

LOGGER = logging.getLogger(__name__)

S2_RED_INDEX = 3
S2_NIR_INDEX = 7
S2_SWIR2_INDEX = 11


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _load_single_band(path: Path, dst_profile: dict[str, Any], *, resampling: Resampling) -> tuple[np.ndarray, np.ndarray]:
    array, valid_mask = read_reprojected_raster(path, dst_profile, resampling=resampling, out_dtype=np.float32)
    return array[0], valid_mask


def _parse_time_steps(config: TemporalPreprocessingConfig) -> list[tuple[int, int]]:
    return iter_months(config.time_start, config.time_end)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full_like(numerator, np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-6)
    output[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
    return output


def _compute_s2_indices(array: np.ndarray) -> np.ndarray:
    ndvi = _safe_divide(array[S2_NIR_INDEX] - array[S2_RED_INDEX], array[S2_NIR_INDEX] + array[S2_RED_INDEX])
    nbr = _safe_divide(array[S2_NIR_INDEX] - array[S2_SWIR2_INDEX], array[S2_NIR_INDEX] + array[S2_SWIR2_INDEX])
    return np.stack([ndvi, nbr], axis=0).astype(np.float32)


def _empty_channels(num_channels: int, height: int, width: int) -> np.ndarray:
    return np.full((num_channels, height, width), np.nan, dtype=np.float32)


def _flatten_channels(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        time_steps, channels, height, width = array.shape
        return array.reshape(time_steps * channels, height, width)
    raise ValueError(f"Unsupported temporal input ndim={array.ndim}")


def _temporal_channel_names(config: TemporalPreprocessingConfig) -> list[str]:
    names: list[str] = []
    if config.include_sentinel2:
        names.extend(f"s2_{band}" for band in S2_BAND_NAMES)
        if config.add_s2_indices:
            names.extend(["s2_ndvi", "s2_nbr"])
        if config.add_validity_channels:
            names.append("s2_valid")
    if config.include_sentinel1:
        names.append("s1_vv")
        if config.add_validity_channels:
            names.append("s1_valid")
    if config.include_aef and config.aef_pca_dim > 0:
        names.extend(f"aef_pc{index + 1:02d}" for index in range(config.aef_pca_dim))
        if config.add_validity_channels:
            names.append("aef_valid")
    if not names:
        raise ValueError("At least one modality must be enabled for temporal preprocessing")
    return names


def _window_names(num_windows: int) -> list[str]:
    if num_windows == 3:
        return ["early", "middle", "late"]
    return [f"window_{index + 1:02d}" for index in range(num_windows)]


def _sequence_to_summary(sequence: np.ndarray, num_windows: int) -> tuple[np.ndarray, list[str]]:
    indices = np.array_split(np.arange(sequence.shape[0]), num_windows)
    windows = []
    for subset in indices:
        with np.errstate(all="ignore"):
            windows.append(np.nanmean(sequence[subset], axis=0).astype(np.float32))
    delta = (windows[-1] - windows[0]).astype(np.float32)
    invalid = ~np.isfinite(windows[-1]) | ~np.isfinite(windows[0])
    delta[invalid] = np.nan
    names = _window_names(num_windows)
    return np.concatenate(windows + [delta], axis=0).astype(np.float32), names + ["delta"]


def _normalization_stats(
    tile_ids: list[str],
    preprocessing_dir: Path,
    *,
    split: str,
    min_std: float,
) -> dict[str, Any]:
    first = np.load(get_temporal_input_path(preprocessing_dir, split, tile_ids[0])).astype(np.float32)
    num_channels = int(_flatten_channels(first).shape[0])
    sums = np.zeros(num_channels, dtype=np.float64)
    sq_sums = np.zeros(num_channels, dtype=np.float64)
    counts = np.zeros(num_channels, dtype=np.int64)

    for tile_id in tile_ids:
        flattened = _flatten_channels(np.load(get_temporal_input_path(preprocessing_dir, split, tile_id)).astype(np.float32))
        for channel_index in range(num_channels):
            channel = flattened[channel_index]
            valid = np.isfinite(channel)
            if not valid.any():
                continue
            values = channel[valid].astype(np.float64)
            sums[channel_index] += values.sum()
            sq_sums[channel_index] += np.square(values).sum()
            counts[channel_index] += int(values.size)

    mean = np.divide(sums, np.maximum(counts, 1), dtype=np.float64)
    variance = np.divide(sq_sums, np.maximum(counts, 1), dtype=np.float64) - np.square(mean)
    std = np.sqrt(np.maximum(variance, float(min_std) ** 2))
    return {
        "mean": mean.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "num_channels": num_channels,
        "min_std": float(min_std),
    }


def _fit_temporal_aef_pca(
    records: dict[str, TileRecord],
    tile_ids: list[str],
    config: TemporalPreprocessingConfig,
) -> AefPcaModel | None:
    if not config.include_aef or config.aef_pca_dim <= 0:
        return None

    rng = np.random.default_rng(config.seed)
    samples: list[np.ndarray] = []
    for tile_id in tile_ids:
        for path in records[tile_id].aef_files:
            try:
                with rasterio.open(path) as src:
                    valid_mask = src.dataset_mask() > 0
                    valid_indices = np.argwhere(valid_mask)
                    if valid_indices.size == 0:
                        continue
                    sample_count = min(config.pca_num_samples_per_raster, valid_indices.shape[0])
                    chosen = rng.choice(valid_indices.shape[0], size=sample_count, replace=False)
                    rows = valid_indices[chosen, 0]
                    cols = valid_indices[chosen, 1]
                    xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
                    sampled = np.asarray(list(src.sample(list(zip(xs, ys, strict=True)))), dtype=np.float32)
            except rasterio.errors.RasterioError as exc:
                LOGGER.warning("Skipping unreadable AEF raster during temporal PCA fit: %s (%s)", path, exc)
                continue
            vectors = sampled[np.isfinite(sampled).all(axis=1)]
            if vectors.size > 0:
                samples.append(vectors)

    if not samples:
        LOGGER.warning("No valid AEF samples were found for temporal PCA; disabling AEF features")
        return None

    matrix = np.concatenate(samples, axis=0)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[: config.aef_pca_dim].astype(np.float32)
    explained = (singular_values[: config.aef_pca_dim] ** 2) / max(np.sum(singular_values**2), 1e-6)
    return AefPcaModel(
        mean=matrix.mean(axis=0).astype(np.float32),
        components=components,
        explained_variance_ratio=explained.astype(np.float32),
    )


def _load_projected_aef_year(
    path: Path,
    dst_profile: dict[str, Any],
    pca_model: AefPcaModel,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        source = src.read(masked=True).astype(np.float32)
        source_data = np.asarray(source.filled(np.nan), dtype=np.float32)
        source_valid = np.isfinite(source_data).all(axis=0)
        flat = source_data.reshape(source_data.shape[0], -1).T
        projected = pca_model.transform(np.nan_to_num(flat, nan=0.0)).T.reshape(
            pca_model.components.shape[0],
            src.height,
            src.width,
        )
        projected[:, ~source_valid] = np.nan
        destination = np.zeros((projected.shape[0], int(dst_profile["height"]), int(dst_profile["width"])), dtype=np.float32)
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
    valid = valid_destination > 0
    destination[:, ~valid] = np.nan
    return destination.astype(np.float32), valid.astype(bool)


def _apply_source_thresholds(source_name: str, decoded: DecodedLabel, config: TemporalPreprocessingConfig) -> DecodedLabel:
    is_positive = decoded.is_positive.copy()
    if source_name == "radd" and config.radd_positive_mode == "conservative":
        is_positive &= decoded.raw_class >= 3
    if source_name == "gladl":
        is_positive &= decoded.raw_class >= config.gladl_threshold
    if source_name == "glads2":
        is_positive &= decoded.raw_class >= config.glads2_threshold
    return DecodedLabel(
        is_positive=is_positive,
        confidence_score=decoded.confidence_score,
        event_date=decoded.event_date,
        raw_class=decoded.raw_class,
        is_uncertain=decoded.is_uncertain,
        valid_mask=decoded.valid_mask,
    )


def _decode_temporal_labels(
    record: TileRecord,
    dst_profile: dict[str, Any],
    config: TemporalPreprocessingConfig,
) -> dict[str, DecodedLabel]:
    decoded: dict[str, DecodedLabel] = {}

    radd_path = Path(record.label_paths["radd"])
    if radd_path.exists():
        raw, valid_mask = _load_single_band(radd_path, dst_profile, resampling=Resampling.nearest)
        label = decode_radd_array(np.nan_to_num(raw, nan=0.0).astype(np.uint16))
        label = _apply_source_thresholds("radd", label, config)
        decoded["radd"] = DecodedLabel(
            is_positive=label.is_positive,
            confidence_score=label.confidence_score,
            event_date=label.event_date,
            raw_class=label.raw_class,
            is_uncertain=label.is_uncertain,
            valid_mask=valid_mask,
        )

    glads2_alert = Path(record.label_paths["glads2"]["alert"])
    glads2_date = Path(record.label_paths["glads2"]["alert_date"])
    if glads2_alert.exists() and glads2_date.exists():
        alert, valid_a = _load_single_band(glads2_alert, dst_profile, resampling=Resampling.nearest)
        alert_date, valid_b = _load_single_band(glads2_date, dst_profile, resampling=Resampling.nearest)
        label = decode_glads2_array(
            np.nan_to_num(alert, nan=0.0).astype(np.uint8),
            np.nan_to_num(alert_date, nan=0.0).astype(np.uint16),
        )
        label = _apply_source_thresholds("glads2", label, config)
        decoded["glads2"] = DecodedLabel(
            is_positive=label.is_positive,
            confidence_score=label.confidence_score,
            event_date=label.event_date,
            raw_class=label.raw_class,
            is_uncertain=label.is_uncertain,
            valid_mask=valid_a & valid_b,
        )

    gladl_years: list[DecodedLabel] = []
    for year, paths in record.label_paths["gladl"].items():
        alert_path = Path(paths["alert"])
        date_path = Path(paths["alert_date"])
        if not alert_path.exists() or not date_path.exists():
            continue
        alert, valid_a = _load_single_band(alert_path, dst_profile, resampling=Resampling.nearest)
        alert_date, valid_b = _load_single_band(date_path, dst_profile, resampling=Resampling.nearest)
        label = decode_gladl_year(
            np.nan_to_num(alert, nan=0.0).astype(np.uint8),
            np.nan_to_num(alert_date, nan=0.0).astype(np.uint16),
            year,
        )
        label = _apply_source_thresholds("gladl", label, config)
        gladl_years.append(
            DecodedLabel(
                is_positive=label.is_positive,
                confidence_score=label.confidence_score,
                event_date=label.event_date,
                raw_class=label.raw_class,
                is_uncertain=label.is_uncertain,
                valid_mask=valid_a & valid_b,
            )
        )
    if gladl_years:
        decoded["gladl"] = combine_decoded_labels(gladl_years)
    return decoded


def _build_temporal_targets(
    record: TileRecord,
    dst_profile: dict[str, Any],
    config: TemporalPreprocessingConfig,
    bin_spec: TimeBinSpec,
    feature_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    decoded = _decode_temporal_labels(record, dst_profile, config)
    if not decoded:
        raise RuntimeError(f"Tile {record.tile_id} has no usable temporal weak labels")

    source_masks = {name: label.is_positive & (label.confidence_score >= config.min_label_confidence) for name, label in decoded.items()}
    source_valid = {name: label.valid_mask for name, label in decoded.items()}
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

    candidate_dates: list[np.ndarray] = []
    candidate_confidences: list[np.ndarray] = []
    candidate_validity: list[np.ndarray] = []
    for label in decoded.values():
        filtered_dates = filter_dates_to_range(label.event_date, bin_spec)
        usable = label.valid_mask & label.is_positive & (label.confidence_score >= config.min_label_confidence) & ~np.isnat(filtered_dates)
        candidate_dates.append(filtered_dates)
        candidate_confidences.append(label.confidence_score.astype(np.float32))
        candidate_validity.append(usable)

    merged_dates = merge_event_dates(
        candidate_dates,
        candidate_confidences,
        candidate_validity,
        strategy=config.time_merge_strategy,
    )
    time_target = np.full(fused.target.shape, config.time_ignore_index, dtype=np.int64)
    positive = fused.target >= 0.5
    positive_with_date = positive & ~np.isnat(merged_dates)
    if positive_with_date.any():
        time_target[positive_with_date] = dates_to_bin_indices(merged_dates, bin_spec)[positive_with_date]

    ignore_mask = fused.ignore_mask | ~feature_valid_mask
    weight_map = fused.weight_map.astype(np.float32)
    weight_map[ignore_mask] = 0.0
    summary = {
        "label_sources": sorted(decoded.keys()),
        "positive_fraction": float(fused.target[~ignore_mask].mean()) if (~ignore_mask).any() else 0.0,
        "ignored_fraction": float(ignore_mask.mean()),
        "dated_positive_fraction": float(positive_with_date.sum() / max(int(positive.sum()), 1)),
    }
    return (
        fused.target.astype(np.float32),
        time_target,
        ignore_mask.astype(bool),
        weight_map,
        summary,
    )


def _build_s2_cache(
    record: TileRecord,
    dst_profile: dict[str, Any],
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for path in record.sentinel2_files:
        match = S2_PATTERN.match(path.name)
        if match is None:
            continue
        year = int(match.group("year"))
        month = int(match.group("month"))
        array, valid = read_reprojected_raster(path, dst_profile, resampling=Resampling.bilinear, out_dtype=np.float32)
        valid = valid & np.isfinite(array).all(axis=0)
        array[:, ~valid] = np.nan
        cache[(year, month)] = (array.astype(np.float32), valid.astype(bool))
    return cache


def _build_s1_cache(
    record: TileRecord,
    dst_profile: dict[str, Any],
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
    for path in record.sentinel1_files:
        match = S1_PATTERN.match(path.name)
        if match is None:
            continue
        year = int(match.group("year"))
        month = int(match.group("month"))
        array, valid = read_reprojected_raster(path, dst_profile, resampling=Resampling.bilinear, out_dtype=np.float32)
        band = array[0]
        valid = valid & np.isfinite(band)
        band[~valid] = np.nan
        grouped.setdefault((year, month), []).append((band.astype(np.float32), valid.astype(bool)))

    cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for key, observations in grouped.items():
        stacked = np.stack([item[0] for item in observations], axis=0)
        valid_stack = np.stack([item[1] for item in observations], axis=0)
        with np.errstate(all="ignore"):
            averaged = np.nanmean(stacked, axis=0).astype(np.float32)
        valid = np.logical_or.reduce(valid_stack)
        averaged[~valid] = np.nan
        cache[key] = (averaged[None, ...], valid.astype(bool))
    return cache


def _build_aef_cache(
    record: TileRecord,
    dst_profile: dict[str, Any],
    pca_model: AefPcaModel | None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if pca_model is None:
        return {}
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for path in record.aef_files:
        match = AEF_PATTERN.match(path.name)
        if match is None:
            continue
        year = int(match.group("year"))
        cache[year] = _load_projected_aef_year(path, dst_profile, pca_model)
    return cache


def _build_temporal_input(
    record: TileRecord,
    config: TemporalPreprocessingConfig,
    time_steps: list[tuple[int, int]],
    pca_model: AefPcaModel | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[str]]:
    dst_profile = get_raster_profile(record.reference_s2_path)
    height = int(dst_profile["height"])
    width = int(dst_profile["width"])
    step_channel_names = _temporal_channel_names(config)
    s2_cache = _build_s2_cache(record, dst_profile) if config.include_sentinel2 else {}
    s1_cache = _build_s1_cache(record, dst_profile) if config.include_sentinel1 else {}
    aef_cache = _build_aef_cache(record, dst_profile, pca_model) if config.include_aef and config.aef_pca_dim > 0 else {}

    sequence_steps: list[np.ndarray] = []
    overall_valid = np.zeros((height, width), dtype=bool)
    step_metadata: list[dict[str, Any]] = []

    for year, month in time_steps:
        blocks: list[np.ndarray] = []
        metadata: dict[str, Any] = {"year": year, "month": month}
        if config.include_sentinel2:
            s2_array, s2_valid = s2_cache.get((year, month), (_empty_channels(len(S2_BAND_NAMES), height, width), np.zeros((height, width), dtype=bool)))
            blocks.append(s2_array)
            if config.add_s2_indices:
                blocks.append(_compute_s2_indices(s2_array))
            if config.add_validity_channels:
                blocks.append(s2_valid[None, ...].astype(np.float32))
            metadata["s2_available"] = bool(s2_valid.any())
            overall_valid |= s2_valid
        if config.include_sentinel1:
            s1_array, s1_valid = s1_cache.get((year, month), (_empty_channels(1, height, width), np.zeros((height, width), dtype=bool)))
            blocks.append(s1_array)
            if config.add_validity_channels:
                blocks.append(s1_valid[None, ...].astype(np.float32))
            metadata["s1_available"] = bool(s1_valid.any())
            overall_valid |= s1_valid
        if config.include_aef and config.aef_pca_dim > 0:
            aef_array, aef_valid = aef_cache.get(year, (_empty_channels(config.aef_pca_dim, height, width), np.zeros((height, width), dtype=bool)))
            blocks.append(aef_array)
            if config.add_validity_channels:
                blocks.append(aef_valid[None, ...].astype(np.float32))
            metadata["aef_available"] = bool(aef_valid.any())
            overall_valid |= aef_valid
        sequence_steps.append(np.concatenate(blocks, axis=0).astype(np.float32))
        step_metadata.append(metadata)

    sequence = np.stack(sequence_steps, axis=0).astype(np.float32)
    if config.representation == "sequence":
        inputs = sequence.reshape(-1, height, width).astype(np.float32) if config.flatten_time else sequence
        group_names = [f"{year:04d}-{month:02d}" for year, month in time_steps]
    elif config.representation == "summary":
        inputs, group_names = _sequence_to_summary(sequence, config.summary_window_count)
    else:
        raise ValueError(f"Unsupported temporal representation: {config.representation}")

    metadata = {
        "step_metadata": step_metadata,
        "representation": config.representation,
        "flatten_time": bool(config.flatten_time),
        "shape": list(inputs.shape),
        "time_steps": [f"{year:04d}-{month:02d}" for year, month in time_steps],
        "groups": group_names,
    }
    return inputs, overall_valid.astype(bool), metadata, step_channel_names


def run_temporal_preprocessing(
    config: TemporalPreprocessingConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build reusable temporal tiles, targets, and metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        LOGGER.info("Overwriting existing temporal preprocessing summary at %s", summary_path)

    train_records = scan_tiles(config.data_root, "train")
    test_records = scan_tiles(config.data_root, "test")
    time_steps = _parse_time_steps(config)
    bin_spec = build_time_bin_spec(config.time_bin_mode, start=config.time_start, end=config.time_end, ignore_index=config.time_ignore_index)

    positive_fractions: dict[str, float] = {}
    pca_model = _fit_temporal_aef_pca(train_records, sorted(train_records.keys()), config)
    if pca_model is not None:
        save_json(output_dir / "aef_pca.json", pca_model.to_payload())
    else:
        save_json(output_dir / "aef_pca.json", {"disabled": True})

    tile_metadata: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "test": {}}
    train_input_shapes: list[list[int]] = []
    step_channel_names: list[str] | None = None

    for split, records in (("train", train_records), ("test", test_records)):
        for tile_id, record in records.items():
            dst_profile = get_raster_profile(record.reference_s2_path)
            inputs, feature_valid_mask, input_metadata, step_channel_names = _build_temporal_input(
                record,
                config,
                time_steps,
                pca_model,
            )
            _save_array(get_temporal_input_path(output_dir, split, tile_id), inputs.astype(np.float32))
            _save_array(get_temporal_valid_mask_path(output_dir, split, tile_id), feature_valid_mask.astype(np.bool_))
            tile_payload = {**record.to_dict(), **input_metadata}
            if split == "train":
                mask_target, time_target, ignore_mask, weight_map, label_summary = _build_temporal_targets(
                    record,
                    dst_profile,
                    config,
                    bin_spec,
                    feature_valid_mask,
                )
                _save_array(get_temporal_mask_target_path(output_dir, tile_id), mask_target.astype(np.float32))
                _save_array(get_temporal_time_target_path(output_dir, tile_id), time_target.astype(np.int16))
                _save_array(get_temporal_ignore_mask_path(output_dir, tile_id), ignore_mask.astype(np.bool_))
                _save_array(get_temporal_weight_map_path(output_dir, tile_id), weight_map.astype(np.float32))
                tile_payload.update(label_summary)
                usable = ~ignore_mask
                positive_fractions[tile_id] = float(mask_target[usable].mean()) if usable.any() else 0.0
                train_input_shapes.append(list(inputs.shape))
            tile_metadata[split][tile_id] = tile_payload

    train_tile_ids, val_tile_ids = split_train_val_tiles(
        sorted(train_records.keys()),
        val_ratio=config.val_ratio,
        seed=config.split_seed,
        positive_fractions=positive_fractions,
        stratify=config.stratify_split,
    )

    save_json(output_dir / "train_tiles.json", train_tile_ids)
    save_json(output_dir / "val_tiles.json", val_tile_ids)
    save_json(output_dir / "tile_metadata.json", tile_metadata)
    save_json(output_dir / "time_bins.json", bin_spec.to_payload())

    temporal_spec = {
        "representation": config.representation,
        "flatten_time": bool(config.flatten_time),
        "time_steps": [f"{year:04d}-{month:02d}" for year, month in time_steps],
        "step_channel_names": step_channel_names or [],
        "train_input_shapes": train_input_shapes[:4],
        "time_bin_spec": bin_spec.to_payload(),
    }
    save_json(output_dir / "temporal_spec.json", temporal_spec)

    normalization_stats = _normalization_stats(train_tile_ids, output_dir, split="train", min_std=config.min_normalization_std)
    save_json(output_dir / "normalization_stats.json", normalization_stats)

    summary = {
        "config": asdict(config),
        "preprocessing_dir": str(output_dir),
        "num_train_tiles": len(train_records),
        "num_test_tiles": len(test_records),
        "train_tiles": train_tile_ids,
        "val_tiles": val_tile_ids,
        "time_bins": bin_spec.to_payload(),
        "temporal_spec": temporal_spec,
        "positive_fractions": positive_fractions,
    }
    save_json(summary_path, summary)
    return summary
