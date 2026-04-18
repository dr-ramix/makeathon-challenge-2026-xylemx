"""End-to-end preprocessing pipeline for multimodal training artifacts."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.enums import Resampling

from xylemx.config import ExperimentConfig
from xylemx.data.io import (
    TileRecord,
    get_feature_path,
    get_feature_metadata_path,
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
from xylemx.preprocessing.features import (
    AefPcaModel,
    TilePreprocessingError,
    build_multimodal_feature_pack,
    fit_aef_pca_model,
)
from xylemx.preprocessing.normalize import ReservoirPercentileEstimator, RunningChannelStats, clip_array

LOGGER = logging.getLogger(__name__)


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _cache_feature_pack(tile_id: str, split: str, preprocessing_dir: Path, feature_pack: Any) -> None:
    _save_array(get_feature_path(preprocessing_dir, split, tile_id), feature_pack.features.astype(np.float16))
    save_json(get_feature_metadata_path(preprocessing_dir, split, tile_id), feature_pack.metadata)
    _save_array(get_valid_mask_path(preprocessing_dir, split, tile_id), feature_pack.valid_mask.astype(np.bool_))
    if feature_pack.preview is not None:
        _save_array(get_preview_path(preprocessing_dir, split, tile_id), feature_pack.preview.astype(np.uint8))


def _build_and_cache_tile_features(
    record: TileRecord,
    split: str,
    preprocessing_dir: Path,
    config: ExperimentConfig,
    aef_pca: AefPcaModel | None,
) -> dict[str, Any]:
    start = time.perf_counter()
    feature_pack = build_multimodal_feature_pack(record, config, aef_pca)
    _cache_feature_pack(record.tile_id, split, preprocessing_dir, feature_pack)
    return {
        "tile_id": record.tile_id,
        "feature_names": feature_pack.feature_names,
        "metadata": feature_pack.metadata,
        "elapsed_seconds": time.perf_counter() - start,
    }


def _load_single_band(path: Path, dst_profile: dict, *, resampling: Resampling) -> tuple[np.ndarray, np.ndarray]:
    array, valid_mask = read_reprojected_raster(
        path,
        dst_profile,
        resampling=resampling,
        out_dtype=np.float32,
    )
    return array[0], valid_mask


def _fuse_tile_labels(record: TileRecord, config: ExperimentConfig) -> tuple[LabelFusionResult, dict[str, Any]]:
    dst_profile = get_raster_profile(record.reference_s2_path)
    source_masks: dict[str, np.ndarray] = {}
    source_valid: dict[str, np.ndarray] = {}

    radd_path = Path(record.label_paths["radd"])
    if radd_path.exists():
        radd_raw, radd_valid = _load_single_band(radd_path, dst_profile, resampling=Resampling.nearest)
        source_masks["radd"] = radd_positive_mask(radd_raw, mode=config.radd_positive_mode)
        source_valid["radd"] = radd_valid

    glads2_alert_path = Path(record.label_paths["glads2"]["alert"])
    if glads2_alert_path.exists():
        glads2_alert, glads2_valid = _load_single_band(glads2_alert_path, dst_profile, resampling=Resampling.nearest)
        source_masks["glads2"] = glads2_positive_mask(glads2_alert, threshold=config.glads2_threshold)
        source_valid["glads2"] = glads2_valid

    gladl_masks: list[np.ndarray] = []
    gladl_valid_masks: list[np.ndarray] = []
    for year, paths in record.label_paths["gladl"].items():
        alert_path = Path(paths["alert"])
        if not alert_path.exists():
            continue
        alert, valid = _load_single_band(alert_path, dst_profile, resampling=Resampling.nearest)
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


def _run_feature_stage(
    records: dict[str, TileRecord],
    *,
    split: str,
    preprocessing_dir: Path,
    config: ExperimentConfig,
    aef_pca: AefPcaModel | None,
) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    tile_ids = sorted(records.keys())
    metadata_by_tile: dict[str, dict[str, Any]] = {}
    feature_names: list[str] | None = None
    skipped_tile_ids: list[str] = []
    stage_start = time.perf_counter()

    def _log_progress(tile_id: str, completed: int, total: int, tile_seconds: float) -> None:
        elapsed = time.perf_counter() - stage_start
        avg_per_tile = elapsed / max(completed, 1)
        remaining = max(total - completed, 0)
        eta_seconds = avg_per_tile * remaining
        LOGGER.info(
            "Finished %s features for %s (%d/%d) in %.1fs | elapsed=%.1fs avg=%.1fs/tile eta=%.1fs",
            split,
            tile_id,
            completed,
            total,
            tile_seconds,
            elapsed,
            avg_per_tile,
            eta_seconds,
        )

    if config.preprocessing_num_workers <= 1:
        for index, tile_id in enumerate(tile_ids, start=1):
            try:
                result = _build_and_cache_tile_features(records[tile_id], split, preprocessing_dir, config, aef_pca)
            except TilePreprocessingError as exc:
                skipped_tile_ids.append(tile_id)
                LOGGER.warning("Skipping %s tile %s during preprocessing: %s", split, tile_id, exc)
                continue
            feature_names = result["feature_names"]
            metadata_by_tile[tile_id] = result["metadata"]
            _log_progress(tile_id, len(metadata_by_tile), len(tile_ids), result["elapsed_seconds"])
    else:
        LOGGER.info("Using %d preprocessing workers for %s tiles", config.preprocessing_num_workers, split)
        with ProcessPoolExecutor(max_workers=config.preprocessing_num_workers) as executor:
            futures = {
                executor.submit(
                    _build_and_cache_tile_features,
                    records[tile_id],
                    split,
                    preprocessing_dir,
                    config,
                    aef_pca,
                ): tile_id
                for tile_id in tile_ids
            }
            completed = 0
            for future in as_completed(futures):
                tile_id = futures[future]
                try:
                    result = future.result()
                except TilePreprocessingError as exc:
                    skipped_tile_ids.append(tile_id)
                    LOGGER.warning("Skipping %s tile %s during preprocessing: %s", split, tile_id, exc)
                    continue
                feature_names = result["feature_names"]
                metadata_by_tile[tile_id] = result["metadata"]
                completed += 1
                _log_progress(tile_id, completed, len(tile_ids), result["elapsed_seconds"])

    if feature_names is None:
        raise RuntimeError(f"No {split} features were generated during preprocessing")
    return metadata_by_tile, feature_names, skipped_tile_ids


def run_preprocessing(config: ExperimentConfig, preprocessing_dir: str | Path) -> dict[str, Any]:
    """Build reusable preprocessing artifacts under the run directory."""

    preprocessing_dir = Path(preprocessing_dir)
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    summary_path = preprocessing_dir / "summary.json"
    if config.cache_preprocessed and not config.rebuild_cache and summary_path.exists():
        LOGGER.info("Reusing cached preprocessing artifacts from %s", preprocessing_dir)
        return load_json(summary_path)
    LOGGER.info(
        (
            "Preprocessing mode=%s snapshot_mode=%s windows early=%d-%d late=%d-%d"
        ),
        config.temporal_feature_mode,
        config.snapshot_mode,
        config.early_window_start_year,
        config.early_window_end_year,
        config.late_window_start_year,
        config.late_window_end_year,
    )
    LOGGER.info(
        (
            "Selection method=%s fallback_to_nearest_valid_year=%s "
            "min_valid_pixels_fraction_per_snapshot=%.2f skip_bad_snapshots=%s"
        ),
        config.selection_method,
        config.fallback_to_nearest_valid_year,
        config.min_valid_pixels_fraction_per_snapshot,
        config.skip_bad_snapshots,
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

    train_tile_ids, val_tile_ids = split_train_val_tiles(
        sorted(train_records.keys()),
        val_ratio=config.val_ratio,
        seed=config.split_seed,
        positive_fractions=positive_fractions,
        stratify=config.stratify_split,
    )

    aef_pca: AefPcaModel | None = None
    if config.use_aef_features and config.aef_pca_dim > 0:
        LOGGER.info("Fitting AEF PCA on %d train tiles", len(train_tile_ids))
        aef_pca = fit_aef_pca_model(train_records, train_tile_ids, config)
        save_json(preprocessing_dir / "aef_pca.json", aef_pca.to_payload())
    else:
        LOGGER.info("Skipping AEF PCA because AEF features are disabled")
        save_json(preprocessing_dir / "aef_pca.json", {"disabled": True, "aef_pca_dim": 0})

    tile_metadata: dict[str, dict[str, Any]] = {"train": {}, "test": {}}
    skipped_tiles: dict[str, list[str]] = {"train": [], "test": []}

    LOGGER.info("Building train feature caches")
    train_feature_metadata, feature_names, skipped_train_tiles = _run_feature_stage(
        train_records,
        split="train",
        preprocessing_dir=preprocessing_dir,
        config=config,
        aef_pca=aef_pca,
    )
    skipped_tiles["train"] = skipped_train_tiles
    train_tile_ids = [tile_id for tile_id in train_tile_ids if tile_id in train_feature_metadata]
    val_tile_ids = [tile_id for tile_id in val_tile_ids if tile_id in train_feature_metadata]
    if not train_tile_ids:
        raise RuntimeError("No train tiles remained after preprocessing")
    if not val_tile_ids:
        LOGGER.warning("No validation tiles remained after preprocessing")
    for tile_id, record in train_records.items():
        if tile_id not in train_feature_metadata:
            continue
        tile_metadata["train"][tile_id] = {
            **record.to_dict(),
            **train_feature_metadata[tile_id],
            **label_summaries[tile_id],
        }

    LOGGER.info("Building test feature caches")
    test_feature_metadata, _, skipped_test_tiles = _run_feature_stage(
        test_records,
        split="test",
        preprocessing_dir=preprocessing_dir,
        config=config,
        aef_pca=aef_pca,
    )
    skipped_tiles["test"] = skipped_test_tiles
    for tile_id, record in test_records.items():
        if tile_id not in test_feature_metadata:
            continue
        tile_metadata["test"][tile_id] = {
            **record.to_dict(),
            **test_feature_metadata[tile_id],
        }

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
        "num_train_tiles": len(train_records),
        "num_test_tiles": len(test_records),
        "num_cached_train_tiles": len(train_feature_metadata),
        "num_cached_test_tiles": len(test_feature_metadata),
        "train_tiles": train_tile_ids,
        "val_tiles": val_tile_ids,
        "skipped_tiles": skipped_tiles,
        "feature_names": feature_names,
        "positive_fractions": positive_fractions,
        "pca_explained_variance_ratio": aef_pca.explained_variance_ratio.tolist() if aef_pca is not None else [],
    }
    save_json(summary_path, summary)
    return summary


def load_aef_pca(preprocessing_dir: str | Path) -> AefPcaModel:
    """Load a previously fitted AEF PCA model."""

    path = Path(preprocessing_dir) / "aef_pca.json"
    with path.open("r", encoding="utf-8") as handle:
        import json

        payload = json.load(handle)
    if payload.get("disabled"):
        raise RuntimeError(f"AEF PCA is disabled in preprocessing artifacts under {preprocessing_dir}")
    return AefPcaModel.from_payload(payload)
