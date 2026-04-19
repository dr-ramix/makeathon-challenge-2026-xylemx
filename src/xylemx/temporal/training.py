"""Training loop for the temporal pipeline."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from submission_utils import raster_to_geojson
from xylemx.data.io import get_raster_profile, load_json, scan_tiles, write_single_band_geotiff
from xylemx.experiment import append_run_record, configure_logging, create_run_directory, save_json, save_resolved_config
from xylemx.temporal.config import TemporalTrainConfig
from xylemx.temporal.dataset import TemporalPatchDataset
from xylemx.temporal.inference import load_temporal_tile, predict_temporal_tile
from xylemx.temporal.losses import TemporalMultiTaskLoss
from xylemx.temporal.model import DualHeadUNet, DualHeadUNetPlus
from xylemx.temporal.io import get_temporal_cond_path, get_temporal_ignore_mask_path, get_temporal_mask_target_path
from xylemx.training.metrics import add_confusion_counts, compute_confusion_counts, empty_confusion_counts, metrics_from_confusion_counts

LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.amp.autocast("cpu", enabled=False)


def _amp_scaler(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.amp.GradScaler("cpu", enabled=False)


def _build_dataloaders(
    config: TemporalTrainConfig,
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    preprocessing_dir = Path(config.preprocessing_dir)
    train_tile_ids = load_json(preprocessing_dir / "train_tiles.json")
    val_tile_ids = load_json(preprocessing_dir / "val_tiles.json")

    train_dataset = TemporalPatchDataset(
        tile_ids=train_tile_ids,
        preprocessing_dir=preprocessing_dir,
        split="train",
        patch_size=config.patch_size,
        stride=config.train_stride,
        config=config,
        training=True,
    )
    val_dataset = TemporalPatchDataset(
        tile_ids=val_tile_ids,
        preprocessing_dir=preprocessing_dir,
        split="train",
        patch_size=config.patch_size,
        stride=config.eval_stride,
        config=config,
        training=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_tile_ids, val_tile_ids


def _time_metrics(time_logits: torch.Tensor, time_target: torch.Tensor, ignore_index: int) -> dict[str, float]:
    predictions = torch.argmax(time_logits, dim=1)
    valid = time_target != ignore_index
    if not valid.any():
        return {"time_accuracy": 0.0, "time_mae_bins": 0.0, "time_valid_pixels": 0.0}
    accuracy = float((predictions[valid] == time_target[valid]).float().mean().item())
    mae = float(torch.abs(predictions[valid].float() - time_target[valid].float()).mean().item())
    return {"time_accuracy": accuracy, "time_mae_bins": mae, "time_valid_pixels": float(valid.sum().item())}


def _train_or_eval_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: TemporalMultiTaskLoss,
    *,
    device: torch.device,
    config: TemporalTrainConfig,
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    mask_losses: list[float] = []
    time_losses: list[float] = []
    confusion = empty_confusion_counts()
    time_accuracy_values: list[float] = []
    time_mae_values: list[float] = []
    time_valid_values: list[float] = []

    for batch in dataloader:
        inputs = batch["inputs"].to(device)
        cond = batch["cond"].to(device)
        mask_target = batch["mask_target"].to(device)
        time_target = batch["time_target"].to(device)
        ignore_mask = batch["ignore_mask"].to(device)
        weight_map = batch["weight_map"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, enabled=config.mixed_precision and device.type == "cuda"):
            outputs = model(inputs, cond)
            batch_loss = loss_fn(outputs, mask_target, time_target, ignore_mask=ignore_mask, weight_map=weight_map)
        loss = batch_loss["loss"]
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        losses.append(float(loss.item()))
        mask_losses.append(float(batch_loss["mask_loss"].item()))
        time_losses.append(float(batch_loss["time_loss"].item()))
        confusion = add_confusion_counts(
            confusion,
            compute_confusion_counts(outputs["mask_logits"].detach(), mask_target, ignore_mask=ignore_mask, threshold=config.inference_threshold),
        )
        timing = _time_metrics(outputs["time_logits"].detach(), time_target, config.time_ignore_index)
        time_accuracy_values.append(timing["time_accuracy"])
        time_mae_values.append(timing["time_mae_bins"])
        time_valid_values.append(timing["time_valid_pixels"])

    summary = metrics_from_confusion_counts(confusion)
    summary.update(
        {
            "loss": float(sum(losses) / max(len(losses), 1)),
            "mask_loss": float(sum(mask_losses) / max(len(mask_losses), 1)),
            "time_loss": float(sum(time_losses) / max(len(time_losses), 1)),
            "time_accuracy": float(sum(time_accuracy_values) / max(len(time_accuracy_values), 1)),
            "time_mae_bins": float(sum(time_mae_values) / max(len(time_mae_values), 1)),
            "time_valid_pixels": float(sum(time_valid_values)),
            "valid_pixels": float(confusion["valid_pixels"]),
        }
    )
    return summary


def _build_model(config: TemporalTrainConfig, preprocessing_dir: Path) -> tuple[torch.nn.Module, dict[str, object]]:
    temporal_spec = load_json(preprocessing_dir / "temporal_spec.json")
    time_bins = load_json(preprocessing_dir / "time_bins.json")
    sample_tile = load_json(preprocessing_dir / "train_tiles.json")[0]
    sample_inputs = np.load(preprocessing_dir / "inputs" / "train" / f"{sample_tile}.npy").astype(np.float32)
    if sample_inputs.ndim == 4:
        sample_inputs = sample_inputs.reshape(-1, sample_inputs.shape[-2], sample_inputs.shape[-1])
    sample_cond_path = get_temporal_cond_path(preprocessing_dir, "train", sample_tile)
    sample_cond = np.load(sample_cond_path).astype(np.float32).reshape(-1) if sample_cond_path.exists() else np.zeros((0,), dtype=np.float32)
    model_name = str(config.model).lower().strip()
    shared_kwargs = {
        "in_channels": int(sample_inputs.shape[0]),
        "num_time_classes": int(time_bins["num_classes"]),
        "cond_dim": int(sample_cond.shape[0]),
        "input_is_sequence": False,
        "stem_channels": config.stem_channels,
        "base_channels": config.base_channels,
        "dropout": config.dropout,
        "temporal_kernel_size": config.temporal_kernel_size,
        "film_hidden_dim": config.film_hidden_dim,
    }
    if model_name in {"film_temporal_unet", "temporal_unet"}:
        model = DualHeadUNet(**shared_kwargs)
    elif model_name in {"film_temporal_unet_plus", "film_temporal_plus", "temporal_unet_plus"}:
        model = DualHeadUNetPlus(**shared_kwargs)
    else:
        raise ValueError(
            f"Unsupported temporal model '{config.model}'. "
            "Supported: film_temporal_unet, film_temporal_unet_plus"
        )
    return model, {
        "model_name": model_name,
        "temporal_spec": temporal_spec,
        "time_bins": time_bins,
        "sample_shape": list(sample_inputs.shape),
        "cond_dim": int(sample_cond.shape[0]),
    }


def _export_predictions(
    model: torch.nn.Module,
    *,
    split: str,
    tile_ids: list[str],
    records,
    preprocessing_dir: Path,
    output_dir: Path,
    config: TemporalTrainConfig,
    device: torch.device,
    time_labels: list[str],
) -> list[dict[str, object]]:
    exported: list[dict[str, object]] = []
    cache_split = "test" if split == "test" else "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    for tile_id in tile_ids:
        inputs, cond, valid_mask = load_temporal_tile(
            preprocessing_dir,
            split=cache_split,
            tile_id=tile_id,
            clip=config.normalized_feature_clip,
        )
        probability, time_prediction = predict_temporal_tile(
            model,
            inputs,
            cond,
            valid_mask,
            device=device,
            patch_size=config.patch_size,
            stride=config.eval_stride,
            batch_size=config.batch_size,
            mixed_precision=config.mixed_precision,
        )
        binary = ((probability >= config.inference_threshold) & valid_mask).astype(np.uint8)
        reference_profile = get_raster_profile(records[tile_id].reference_s2_path)
        raster_path = output_dir / f"{tile_id}_pred.tif"
        write_single_band_geotiff(raster_path, binary, reference_profile, dtype="uint8", nodata=0)
        time_raster_path = output_dir / f"{tile_id}_time.tif"
        if config.save_time_rasters:
            write_single_band_geotiff(time_raster_path, time_prediction, reference_profile, dtype="int16", nodata=-1)
        payload = {
            "tile_id": tile_id,
            "path": str(raster_path),
            "positive_pixels": int(binary.sum()),
            "time_raster_path": str(time_raster_path) if config.save_time_rasters else "",
        }
        if config.create_submission_geojson:
            geojson_path = output_dir / f"{tile_id}.geojson"
            raster_to_geojson(
                raster_path,
                output_path=geojson_path,
                time_raster_path=time_raster_path if config.save_time_rasters else None,
                time_labels=time_labels,
                time_strategy=config.polygon_time_strategy,
            )
            payload["geojson_path"] = str(geojson_path)
        if split == "val":
            target = np.load(get_temporal_mask_target_path(preprocessing_dir, tile_id)).astype(np.float32)
            ignore = np.load(get_temporal_ignore_mask_path(preprocessing_dir, tile_id)).astype(bool)
            counts = compute_confusion_counts(
                torch.from_numpy(np.log(np.clip(probability, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probability, 1e-6, 1.0))).float()[None, None],
                torch.from_numpy(target).float()[None, None],
                ignore_mask=torch.from_numpy(ignore)[None, None],
                threshold=config.inference_threshold,
            )
            payload.update(metrics_from_confusion_counts(counts))
        exported.append(payload)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"tiles": exported}, handle, indent=2, sort_keys=True)
    return exported


def train_temporal_model(config: TemporalTrainConfig) -> dict[str, object]:
    """Train the temporal model and optionally export temporal predictions."""

    seed_everything(config.seed)
    preprocessing_dir = Path(config.preprocessing_dir)
    time_bins = load_json(preprocessing_dir / "time_bins.json")
    run_dir = create_run_directory(config)
    configure_logging(run_dir / "train.log")
    save_resolved_config(config, run_dir)
    save_json(run_dir / "time_bins.json", time_bins)

    train_loader, val_loader, train_tile_ids, val_tile_ids = _build_dataloaders(config)
    model, model_summary = _build_model(config, preprocessing_dir)
    save_json(run_dir / "model_summary.json", model_summary)
    device = _resolve_device(config.device)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = _amp_scaler(device, config.mixed_precision and device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0
        if config.epochs <= 1
        else (
            float(epoch + 1) / max(config.warmup_epochs, 1)
            if epoch < config.warmup_epochs
            else max(
                0.5 * (1.0 + math.cos(math.pi * ((epoch - config.warmup_epochs) / max(config.epochs - config.warmup_epochs, 1)))),
                config.min_lr / max(config.lr, 1e-12),
            )
        ),
    )
    loss_fn = TemporalMultiTaskLoss(
        mask_loss_name=config.mask_loss,
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
        pos_weight=config.pos_weight,
        lambda_time=config.lambda_time,
        time_loss_weight=config.time_loss_weight,
        time_ignore_index=config.time_ignore_index,
    )

    history: list[dict[str, float | int]] = []
    best_score = float("-inf")
    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"
    start_time = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        train_summary = _train_or_eval_epoch(
            model,
            train_loader,
            loss_fn,
            device=device,
            config=config,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_summary = _train_or_eval_epoch(
            model,
            val_loader,
            loss_fn,
            device=device,
            config=config,
        )
        scheduler.step()
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_summary["loss"],
            "train_dice": train_summary["dice"],
            "train_time_accuracy": train_summary["time_accuracy"],
            "val_loss": val_summary["loss"],
            "val_dice": val_summary["dice"],
            "val_iou": val_summary["iou"],
            "val_time_accuracy": val_summary["time_accuracy"],
            "val_time_mae_bins": val_summary["time_mae_bins"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_summary)
        save_json(run_dir / "metrics" / "history.json", history)
        LOGGER.info("Epoch %03d | train_loss=%.4f val_loss=%.4f val_dice=%.4f val_time_acc=%.4f", epoch, train_summary["loss"], val_summary["loss"], val_summary["dice"], val_summary["time_accuracy"])
        score = float(val_summary["dice"] + 0.25 * val_summary["time_accuracy"])
        checkpoint = {
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "run_dir": str(run_dir),
            "preprocessing_dir": str(preprocessing_dir),
            "model_summary": model_summary,
            "epoch": epoch,
            "history": history,
        }
        torch.save(checkpoint, run_dir / "checkpoints" / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(checkpoint, best_checkpoint_path)

    best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    records_train = scan_tiles(config.data_root, "train")
    records_test = scan_tiles(config.data_root, "test")
    export_results: dict[str, list[dict[str, object]]] = {}
    for split in config.export_splits:
        if split == "val":
            export_results["val"] = _export_predictions(
                model,
                split="val",
                tile_ids=val_tile_ids,
                records=records_train,
                preprocessing_dir=preprocessing_dir,
                output_dir=run_dir / "predictions" / "val",
                config=config,
                device=device,
                time_labels=list(time_bins["labels"]),
            )
        elif split == "train":
            export_results["train"] = _export_predictions(
                model,
                split="train",
                tile_ids=train_tile_ids,
                records=records_train,
                preprocessing_dir=preprocessing_dir,
                output_dir=run_dir / "predictions" / "train",
                config=config,
                device=device,
                time_labels=list(time_bins["labels"]),
            )
        elif split == "test":
            export_results["test"] = _export_predictions(
                model,
                split="test",
                tile_ids=sorted(records_test.keys()),
                records=records_test,
                preprocessing_dir=preprocessing_dir,
                output_dir=run_dir / "predictions" / "test",
                config=config,
                device=device,
                time_labels=list(time_bins["labels"]),
            )
        else:
            raise ValueError(f"Unsupported export split: {split}")

    duration_seconds = time.perf_counter() - start_time
    leaderboard_record = {
        "run_name": run_dir.name,
        "model": config.model,
        "best_val_dice": max(float(item["val_dice"]) for item in history) if history else 0.0,
        "best_val_iou": max(float(item["val_iou"]) for item in history) if history else 0.0,
        "best_val_time_accuracy": max(float(item["val_time_accuracy"]) for item in history) if history else 0.0,
        "duration_seconds": duration_seconds,
        "run_dir": str(run_dir),
    }
    append_run_record(config.output_root, leaderboard_record)
    final_summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(best_checkpoint_path),
        "history": history,
        "exports": export_results,
        "leaderboard_record": leaderboard_record,
    }
    save_json(run_dir / "summary.json", final_summary)
    return final_summary
