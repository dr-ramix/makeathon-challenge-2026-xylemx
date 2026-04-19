"""Training orchestration for the osapiens deforestation baseline."""

from __future__ import annotations

import copy
import errno
import json
import logging
import math
import random
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from xylemx.config import ExperimentConfig
from xylemx.data.dataset import SegmentationPatchDataset
from xylemx.data.io import (
    get_ignore_mask_path,
    get_preview_path,
    get_raster_profile,
    get_target_path,
    load_json,
    scan_tiles,
    write_single_band_geotiff,
)
from xylemx.experiment import append_run_record, configure_logging, create_run_directory, save_json, save_resolved_config
from xylemx.models.baseline import build_model
from xylemx.models.losses import build_loss
from xylemx.preprocessing.pipeline import run_preprocessing
from xylemx.training.augmentations import apply_cutmix, apply_mixup
from xylemx.training.inference import load_normalized_tile, predict_probability_map
from xylemx.training.metrics import (
    add_confusion_counts,
    compute_confusion_counts,
    empty_confusion_counts,
    metrics_from_confusion_counts,
)
from xylemx.visualization.render import save_bw_mask, save_panel, save_probability_map
from xylemx.visualization.render import save_confusion_map, save_overlay_image, save_preview_image

LOGGER = logging.getLogger(__name__)


def _sanitize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Replace non-finite values before they hit the model or loss."""

    if torch.isfinite(tensor).all():
        return tensor
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _has_nonfinite_parameters(model: torch.nn.Module) -> bool:
    """Check whether any parameter already contains NaN or Inf."""

    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            return True
    return False


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _amp_scaler(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.amp.GradScaler("cpu", enabled=False)


def _amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return nullcontext()


class ModelEma:
    """Exponential moving average of model weights."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            ema_state = self.module.state_dict()
            model_state = model.state_dict()
            for key, value in ema_state.items():
                if not value.is_floating_point():
                    value.copy_(model_state[key])
                    continue
                value.mul_(self.decay).add_(model_state[key], alpha=1.0 - self.decay)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _build_optimizer(config: ExperimentConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if config.optimizer.lower() != "adamw":
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def _build_scheduler(
    config: ExperimentConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LambdaLR:
    if config.scheduler.lower() != "cosine":
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")

    def lr_lambda(epoch_index: int) -> float:
        if config.epochs <= 1:
            return 1.0
        if epoch_index < config.warmup_epochs:
            return float(epoch_index + 1) / max(config.warmup_epochs, 1)
        progress = (epoch_index - config.warmup_epochs) / max(config.epochs - config.warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_scale = config.min_lr / max(config.lr, 1e-12)
        return max(cosine, min_scale)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _plot_history(history: list[dict[str, float | int]], path: Path) -> None:
    if not history:
        return
    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    val_loss = [float(item["val_loss"]) for item in history]
    val_dice = [float(item["val_dice"]) for item in history]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, val_dice, label="val_dice")
    axes[1].set_title("Validation Dice")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def _build_dataloaders(
    config: ExperimentConfig,
    preprocessing_dir: Path,
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    train_tile_ids = load_json(preprocessing_dir / "train_tiles.json")
    val_tile_ids = load_json(preprocessing_dir / "val_tiles.json")

    train_dataset = SegmentationPatchDataset(
        tile_ids=train_tile_ids,
        preprocessing_dir=preprocessing_dir,
        split="train",
        patch_size=config.patch_size,
        stride=config.train_stride,
        config=config,
        training=True,
    )
    val_dataset = SegmentationPatchDataset(
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


def _maybe_fallback_num_workers(
    config: ExperimentConfig,
    preprocessing_dir: Path,
) -> tuple[DataLoader, DataLoader, list[str], list[str], ExperimentConfig]:
    try:
        train_loader, val_loader, train_tile_ids, val_tile_ids = _build_dataloaders(config, preprocessing_dir)
        _ = next(iter(train_loader))
        return train_loader, val_loader, train_tile_ids, val_tile_ids, config
    except RuntimeError as exc:
        if config.num_workers == 0:
            raise
        message = str(exc)
        if any(token in message for token in ["torch_shm_manager", "Operation not permitted", "DataLoader worker process"]):
            LOGGER.warning("Falling back to num_workers=0 due to multiprocessing restrictions")
            fallback = copy.deepcopy(config)
            fallback.num_workers = 0
            train_loader, val_loader, train_tile_ids, val_tile_ids = _build_dataloaders(fallback, preprocessing_dir)
            return train_loader, val_loader, train_tile_ids, val_tile_ids, fallback
        raise


def _train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    scaler,
    *,
    device: torch.device,
    config: ExperimentConfig,
    ema_model: ModelEma | None = None,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    confusion = empty_confusion_counts()

    for batch in dataloader:
        inputs = _sanitize_tensor(batch["inputs"].to(device))
        targets = _sanitize_tensor(batch["target"].to(device))
        ignore_masks = batch["ignore_mask"].to(device)
        weight_maps = _sanitize_tensor(batch["weight_map"].to(device))

        if config.cutmix and config.cutmix_p > 0 and np.random.random() < config.cutmix_p:
            inputs, targets, weight_maps, ignore_masks = apply_cutmix(
                inputs,
                targets,
                weight_maps,
                ignore_masks,
                alpha=config.mixup_alpha,
            )
        elif config.mixup and config.mixup_p > 0 and np.random.random() < config.mixup_p:
            inputs, targets, weight_maps, ignore_masks = apply_mixup(
                inputs,
                targets,
                weight_maps,
                ignore_masks,
                alpha=config.mixup_alpha,
            )

        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, enabled=config.mixed_precision and device.type == "cuda"):
            logits = model(inputs)
            logits = _sanitize_tensor(logits)
            loss = loss_fn(logits, targets, ignore_mask=ignore_masks, weight_map=weight_maps)

        if not torch.isfinite(loss):
            LOGGER.warning("Skipping non-finite training batch loss")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if _has_nonfinite_parameters(model):
            raise RuntimeError("Model parameters became non-finite during training")

        if ema_model is not None:
            ema_model.update(model)

        losses.append(float(loss.item()))
        confusion = add_confusion_counts(
            confusion,
            compute_confusion_counts(logits.detach(), targets, ignore_mask=ignore_masks),
        )

    summary = metrics_from_confusion_counts(confusion)
    summary["loss"] = float(sum(losses) / max(len(losses), 1))
    summary["valid_pixels"] = float(confusion["valid_pixels"])
    return summary


def _evaluate_patch_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn,
    *,
    device: torch.device,
    config: ExperimentConfig,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    confusion = empty_confusion_counts()

    with torch.no_grad():
        for batch in dataloader:
            inputs = _sanitize_tensor(batch["inputs"].to(device))
            targets = _sanitize_tensor(batch["target"].to(device))
            ignore_masks = batch["ignore_mask"].to(device)
            weight_maps = _sanitize_tensor(batch["weight_map"].to(device))

            with _amp_context(device, enabled=config.mixed_precision and device.type == "cuda"):
                logits = _sanitize_tensor(model(inputs))
                loss = loss_fn(logits, targets, ignore_mask=ignore_masks, weight_map=weight_maps)
            if not torch.isfinite(loss):
                LOGGER.warning("Encountered non-finite validation loss; skipping batch in aggregate")
                continue
            losses.append(float(loss.item()))
            confusion = add_confusion_counts(
                confusion,
                compute_confusion_counts(logits, targets, ignore_mask=ignore_masks),
            )

    summary = metrics_from_confusion_counts(confusion)
    summary["loss"] = float(sum(losses) / max(len(losses), 1))
    summary["valid_pixels"] = float(confusion["valid_pixels"])
    return summary


def _compute_counts_from_probability(
    probability_map: np.ndarray,
    target: np.ndarray,
    ignore_mask: np.ndarray,
    *,
    threshold: float,
) -> dict[str, int]:
    logits = torch.from_numpy(np.log(np.clip(probability_map, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - probability_map, 1e-6, 1.0))).float()[None, None]
    targets = torch.from_numpy(target).float()[None, None]
    ignore = torch.from_numpy(ignore_mask).bool()[None, None]
    return compute_confusion_counts(logits, targets, ignore_mask=ignore, threshold=threshold)


def _evaluate_tiles(
    model: torch.nn.Module,
    *,
    tile_ids: list[str],
    records: dict[str, Any],
    preprocessing_dir: Path,
    config: ExperimentConfig,
    device: torch.device,
    prediction_dir: Path | None = None,
) -> dict[str, float]:
    confusion = empty_confusion_counts()
    for tile_id in tile_ids:
        features, valid_mask = load_normalized_tile(preprocessing_dir, split="train", tile_id=tile_id, config=config)
        probability = predict_probability_map(
            model,
            features,
            valid_mask,
            device=device,
            patch_size=config.patch_size,
            stride=config.eval_stride,
            batch_size=config.batch_size,
            mixed_precision=config.mixed_precision,
            tta=config.tta,
            tta_modes=config.tta_modes,
        )
        target = np.load(get_target_path(preprocessing_dir, tile_id)).astype(np.float32)
        ignore_mask = np.load(get_ignore_mask_path(preprocessing_dir, tile_id)).astype(bool) | ~valid_mask
        counts = _compute_counts_from_probability(probability, target, ignore_mask, threshold=config.inference_threshold)
        confusion = add_confusion_counts(confusion, counts)

        if prediction_dir is not None:
            binary = ((probability >= config.inference_threshold) & valid_mask).astype(np.uint8)
            reference_profile = get_raster_profile(records[tile_id].reference_s2_path)
            write_single_band_geotiff(prediction_dir / f"{tile_id}_pred.tif", binary, reference_profile, dtype="uint8", nodata=0)

    summary = metrics_from_confusion_counts(confusion)
    summary["valid_pixels"] = float(confusion["valid_pixels"])
    return summary


def _save_tile_visualizations(
    model: torch.nn.Module,
    *,
    tile_ids: list[str],
    split: str,
    preprocessing_dir: Path,
    output_dir: Path,
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
) -> None:
    for tile_id in tile_ids:
        features, valid_mask = load_normalized_tile(preprocessing_dir, split="train", tile_id=tile_id, config=config)
        probability = predict_probability_map(
            model,
            features,
            valid_mask,
            device=device,
            patch_size=config.patch_size,
            stride=config.eval_stride,
            batch_size=config.batch_size,
            mixed_precision=config.mixed_precision,
            tta=config.tta,
            tta_modes=config.tta_modes,
        )
        target = np.load(get_target_path(preprocessing_dir, tile_id)).astype(np.float32)
        ignore_mask = np.load(get_ignore_mask_path(preprocessing_dir, tile_id)).astype(bool) | ~valid_mask
        binary = ((probability >= config.inference_threshold) & valid_mask).astype(np.float32)

        preview_path = get_preview_path(preprocessing_dir, "train", tile_id)
        preview = np.load(preview_path) if preview_path.exists() else None

        split_dir = output_dir / split
        if config.visualization_save_preview_png and config.visualization_include_input_preview:
            save_preview_image(split_dir / f"sample_{tile_id}_epoch_{epoch:03d}_preview.png", preview)
        save_bw_mask(split_dir / f"sample_{tile_id}_epoch_{epoch:03d}_pred.png", binary)
        save_bw_mask(split_dir / f"sample_{tile_id}_epoch_{epoch:03d}_true.png", target)
        if config.visualization_include_probability:
            save_probability_map(split_dir / f"sample_{tile_id}_epoch_{epoch:03d}_prob.png", probability)
        if config.visualization_include_overlays and config.visualization_include_input_preview:
            save_overlay_image(
                split_dir / f"sample_{tile_id}_epoch_{epoch:03d}_true_overlay.png",
                preview=preview,
                mask=target,
                color=(59, 130, 246),
            )
            save_overlay_image(
                split_dir / f"sample_{tile_id}_epoch_{epoch:03d}_pred_overlay.png",
                preview=preview,
                mask=binary,
                color=(236, 72, 153),
            )
        if config.visualization_include_error_map:
            save_confusion_map(
                split_dir / f"sample_{tile_id}_epoch_{epoch:03d}_error.png",
                true_mask=target,
                pred_mask=binary,
                ignore_mask=ignore_mask,
            )
        save_panel(
            split_dir / f"sample_{tile_id}_epoch_{epoch:03d}_panel.png",
            preview=preview if config.visualization_include_input_preview else None,
            true_mask=target,
            probability=probability if config.visualization_include_probability else None,
            pred_mask=binary,
            ignore_mask=ignore_mask,
            dpi=config.visualization_dpi,
        )


def _should_save_epoch_visualizations(epoch: int, config: ExperimentConfig) -> bool:
    """Decide whether to save qualitative epoch visualizations."""

    if config.visualization_every_n_epochs <= 0:
        return False
    return epoch == 1 or epoch % config.visualization_every_n_epochs == 0 or epoch == config.epochs


def _try_save_epoch_visualizations(
    model: torch.nn.Module,
    *,
    tile_ids: list[str],
    split: str,
    preprocessing_dir: Path,
    output_dir: Path,
    config: ExperimentConfig,
    device: torch.device,
    epoch: int,
) -> bool:
    """Save visualizations, returning False when disk exhaustion forces them off."""

    try:
        _save_tile_visualizations(
            model,
            tile_ids=tile_ids,
            split=split,
            preprocessing_dir=preprocessing_dir,
            output_dir=output_dir,
            config=config,
            device=device,
            epoch=epoch,
        )
        return True
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            LOGGER.warning("Disabling further visualizations because the disk is full: %s", exc)
            return False
        raise


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema_model: ModelEma | None,
    config: ExperimentConfig,
    epoch: int,
    in_channels: int,
    run_dir: Path,
    preprocessing_dir: Path,
    best_val_dice: float,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.module.state_dict() if ema_model is not None else None,
        "config": asdict(config),
        "epoch": epoch,
        "in_channels": in_channels,
        "run_dir": str(run_dir),
        "preprocessing_dir": str(preprocessing_dir),
        "best_val_dice": best_val_dice,
    }
    torch.save(payload, path)


def train_model(config: ExperimentConfig) -> dict[str, Any]:
    """Run the full training pipeline and return the final metrics summary."""

    started_at = datetime.now(timezone.utc)
    run_dir = create_run_directory(config)
    configure_logging(run_dir / "train.log")
    save_resolved_config(config, run_dir)

    seed_everything(config.seed)
    preprocessing_dir = Path(config.preprocessing_dir) if config.preprocessing_dir else run_dir / "artifacts" / "preprocessing"
    if config.reuse_preprocessing and (preprocessing_dir / "summary.json").exists():
        LOGGER.info("Reusing existing preprocessing artifacts from %s", preprocessing_dir)
        preprocessing_summary = load_json(preprocessing_dir / "summary.json")
    else:
        LOGGER.info("Running preprocessing into %s", preprocessing_dir)
        preprocessing_summary = run_preprocessing(config, preprocessing_dir)
    save_json(run_dir / "artifacts" / "preprocessing_summary.json", preprocessing_summary)

    train_loader, val_loader, train_tile_ids, val_tile_ids, effective_config = _maybe_fallback_num_workers(config, preprocessing_dir)
    if effective_config.num_workers != config.num_workers:
        save_resolved_config(effective_config, run_dir)

    sample_batch = next(iter(train_loader))
    in_channels = int(sample_batch["inputs"].shape[1])
    device = _resolve_device(effective_config.device)

    model = build_model(
        effective_config.model,
        in_channels=in_channels,
        dropout=effective_config.dropout,
        stochastic_depth=effective_config.stochastic_depth,
        pretrained=effective_config.encoder_pretrained,
    ).to(device)
    optimizer = _build_optimizer(effective_config, model)
    scheduler = _build_scheduler(effective_config, optimizer)
    scaler = _amp_scaler(device, enabled=effective_config.mixed_precision and device.type == "cuda")
    loss_fn = build_loss(
        effective_config.loss,
        bce_weight=effective_config.bce_weight,
        dice_weight=effective_config.dice_weight,
        pos_weight=effective_config.pos_weight,
        focal_gamma=effective_config.focal_loss_gamma,
        label_smoothing=effective_config.label_smoothing,
        hard_negative_mining=effective_config.hard_negative_mining,
        hard_negative_ratio=effective_config.hard_negative_ratio,
    )
    ema_model = ModelEma(model, decay=effective_config.ema_decay) if effective_config.ema else None

    train_records = scan_tiles(effective_config.data_root, "train")
    fixed_train_tiles = train_tile_ids[: min(effective_config.visualization_num_samples, len(train_tile_ids))]
    fixed_val_tiles = val_tile_ids[: min(effective_config.visualization_num_samples, len(val_tile_ids))]

    best_val_dice = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    visualization_enabled = True

    for epoch in range(1, effective_config.epochs + 1):
        train_summary = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            scaler,
            device=device,
            config=effective_config,
            ema_model=ema_model,
        )
        eval_model = ema_model.module if ema_model is not None else model
        val_patch_summary = _evaluate_patch_loader(eval_model, val_loader, loss_fn, device=device, config=effective_config)
        val_tile_summary = _evaluate_tiles(
            eval_model,
            tile_ids=val_tile_ids,
            records=train_records,
            preprocessing_dir=preprocessing_dir,
            config=effective_config,
            device=device,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_summary: dict[str, float | int] = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_summary["loss"],
            "train_iou": train_summary["iou"],
            "train_dice": train_summary["dice"],
            "train_f1": train_summary["f1"],
            "train_accuracy": train_summary["accuracy"],
            "train_precision": train_summary["precision"],
            "train_recall": train_summary["recall"],
            "val_loss": val_patch_summary["loss"],
            "val_iou": val_tile_summary["iou"],
            "val_dice": val_tile_summary["dice"],
            "val_f1": val_tile_summary["f1"],
            "val_accuracy": val_tile_summary["accuracy"],
            "val_precision": val_tile_summary["precision"],
            "val_recall": val_tile_summary["recall"],
            "val_patch_dice": val_patch_summary["dice"],
            "train_valid_pixels": int(train_summary["valid_pixels"]),
            "val_valid_pixels": int(val_tile_summary["valid_pixels"]),
        }
        history.append(epoch_summary)

        LOGGER.info(
            "epoch=%d lr=%.6f train_loss=%.4f val_loss=%.4f val_dice=%.4f val_iou=%.4f",
            epoch,
            current_lr,
            train_summary["loss"],
            val_patch_summary["loss"],
            val_tile_summary["dice"],
            val_tile_summary["iou"],
        )

        _save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            model=model,
            ema_model=ema_model,
            config=effective_config,
            epoch=epoch,
            in_channels=in_channels,
            run_dir=run_dir,
            preprocessing_dir=preprocessing_dir,
            best_val_dice=best_val_dice,
        )

        if val_tile_summary["dice"] > best_val_dice:
            best_val_dice = float(val_tile_summary["dice"])
            best_epoch = epoch
            best_state = copy.deepcopy(eval_model.state_dict())
            _save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                model=model,
                ema_model=ema_model,
                config=effective_config,
                epoch=epoch,
                in_channels=in_channels,
                run_dir=run_dir,
                preprocessing_dir=preprocessing_dir,
                best_val_dice=best_val_dice,
            )
            save_json(
                run_dir / "metrics" / "best_metrics.json",
                {"best_epoch": best_epoch, "best_val_dice": best_val_dice, "epoch_summary": epoch_summary},
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step()
        save_json(run_dir / "metrics" / "history.json", {"history": history})
        _plot_history(history, run_dir / "metrics" / "learning_curves.png")

        visualization_model = eval_model
        if effective_config.visualization_use_best_checkpoint and best_state is not None:
            visualization_model = copy.deepcopy(eval_model)
            visualization_model.load_state_dict(best_state)
            visualization_model.to(device)
            visualization_model.eval()

        if visualization_enabled and _should_save_epoch_visualizations(epoch, effective_config):
            epoch_visual_dir = run_dir / "visualizations" / f"epoch_{epoch:03d}"
            visualization_enabled = _try_save_epoch_visualizations(
                visualization_model,
                tile_ids=fixed_train_tiles,
                split="train",
                preprocessing_dir=preprocessing_dir,
                output_dir=epoch_visual_dir,
                config=effective_config,
                device=device,
                epoch=epoch,
            )
            if visualization_enabled:
                visualization_enabled = _try_save_epoch_visualizations(
                    visualization_model,
                    tile_ids=fixed_val_tiles,
                    split="val",
                    preprocessing_dir=preprocessing_dir,
                    output_dir=epoch_visual_dir,
                    config=effective_config,
                    device=device,
                    epoch=epoch,
                )

        if epochs_without_improvement >= effective_config.early_stopping_patience:
            LOGGER.info("Early stopping triggered after epoch %d", epoch)
            break

    if best_state is None:
        raise RuntimeError("Training finished without producing a best checkpoint")

    best_model = build_model(
        effective_config.model,
        in_channels=in_channels,
        dropout=effective_config.dropout,
        stochastic_depth=effective_config.stochastic_depth,
        pretrained=effective_config.encoder_pretrained,
    ).to(device)
    best_model.load_state_dict(best_state)
    best_model.eval()

    best_val_summary = _evaluate_tiles(
        best_model,
        tile_ids=val_tile_ids,
        records=train_records,
        preprocessing_dir=preprocessing_dir,
        config=effective_config,
        device=device,
        prediction_dir=run_dir / "predictions" / "val",
    )
    if visualization_enabled:
        visualization_enabled = _try_save_epoch_visualizations(
            best_model,
            tile_ids=fixed_val_tiles,
            split="val",
            preprocessing_dir=preprocessing_dir,
            output_dir=run_dir / "visualizations" / "best",
            config=effective_config,
            device=device,
            epoch=best_epoch,
        )

    final_summary = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "preprocessing_dir": str(preprocessing_dir),
        "best_epoch": best_epoch,
        "best_val_iou": float(best_val_summary["iou"]),
        "best_val_f1": float(best_val_summary["f1"]),
        "best_val_accuracy": float(best_val_summary["accuracy"]),
        "best_val_precision": float(best_val_summary["precision"]),
        "best_val_recall": float(best_val_summary["recall"]),
        "best_val_dice": float(best_val_summary["dice"]),
        "history": history,
    }
    finished_at = datetime.now(timezone.utc)
    duration_seconds = float((finished_at - started_at).total_seconds())
    final_summary["started_at_utc"] = started_at.isoformat()
    final_summary["finished_at_utc"] = finished_at.isoformat()
    final_summary["duration_seconds"] = duration_seconds
    final_summary["duration_minutes"] = duration_seconds / 60.0
    final_summary["epochs_completed"] = len(history)
    final_summary["epochs_requested"] = int(effective_config.epochs)
    final_summary["stopped_early"] = len(history) < int(effective_config.epochs)
    final_summary["visualizations_completed"] = visualization_enabled
    save_json(run_dir / "metrics" / "summary.json", final_summary)
    save_json(
        run_dir / "metrics" / "run_record.json",
        {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "model": effective_config.model,
            "temporal_feature_mode": effective_config.temporal_feature_mode,
            "preprocessing_dir": str(preprocessing_dir),
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_seconds / 60.0,
            "best_epoch": int(best_epoch),
            "best_val_dice": float(best_val_summary["dice"]),
            "best_val_iou": float(best_val_summary["iou"]),
            "best_val_f1": float(best_val_summary["f1"]),
            "best_val_accuracy": float(best_val_summary["accuracy"]),
            "best_val_precision": float(best_val_summary["precision"]),
            "best_val_recall": float(best_val_summary["recall"]),
            "epochs_completed": len(history),
            "epochs_requested": int(effective_config.epochs),
            "stopped_early": len(history) < int(effective_config.epochs),
            "batch_size": int(effective_config.batch_size),
            "patch_size": int(effective_config.patch_size),
            "lr": float(effective_config.lr),
            "min_lr": float(effective_config.min_lr),
            "scheduler": effective_config.scheduler,
            "weight_decay": float(effective_config.weight_decay),
            "dropout": float(effective_config.dropout),
            "stochastic_depth": float(effective_config.stochastic_depth),
            "ema": bool(effective_config.ema),
        },
    )
    append_run_record(
        effective_config.output_root,
        {
            "run_name": run_dir.name,
            "model": effective_config.model,
            "temporal_feature_mode": effective_config.temporal_feature_mode,
            "preprocessing_dir": str(preprocessing_dir),
            "run_dir": str(run_dir),
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_seconds / 60.0,
            "best_epoch": int(best_epoch),
            "best_val_dice": float(best_val_summary["dice"]),
            "best_val_iou": float(best_val_summary["iou"]),
            "best_val_f1": float(best_val_summary["f1"]),
            "best_val_accuracy": float(best_val_summary["accuracy"]),
            "best_val_precision": float(best_val_summary["precision"]),
            "best_val_recall": float(best_val_summary["recall"]),
            "epochs_completed": len(history),
            "epochs_requested": int(effective_config.epochs),
            "stopped_early": len(history) < int(effective_config.epochs),
            "batch_size": int(effective_config.batch_size),
            "patch_size": int(effective_config.patch_size),
            "lr": float(effective_config.lr),
            "min_lr": float(effective_config.min_lr),
            "scheduler": effective_config.scheduler,
            "weight_decay": float(effective_config.weight_decay),
            "dropout": float(effective_config.dropout),
            "stochastic_depth": float(effective_config.stochastic_depth),
            "ema": bool(effective_config.ema),
        },
    )
    LOGGER.info("Training complete. Best epoch=%d best_val_dice=%.4f", best_epoch, best_val_summary["dice"])
    return final_summary
