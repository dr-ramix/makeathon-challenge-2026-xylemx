"""Leaderboard-oriented training entrypoint with optional model search."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from xylemx.config import ExperimentConfig, parse_cli_overrides
from xylemx.data.io import get_ignore_mask_path, get_target_path, load_json
from xylemx.experiment import save_json
from xylemx.models.baseline import build_model
from xylemx.training.inference import load_normalized_tile, predict_probability_map
from xylemx.training.train import train_model

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_CANDIDATES = [
    "resnet34_unetpp",
    "resnet50_fpn",
    "convnext_tiny_fpn",
    "convnextv2_tiny_unetpp",
]

# Tuned defaults focused on robust cross-region performance while staying practical on one GPU.
DEFAULT_LEADERBOARD_TRAIN_OVERRIDES = [
    "model=resnet34_unetpp",
    "preprocessing_dir=output/preprocessing_leaderboard",
    "output_root=output/training_runs_leaderboard",
    "batch_size=4",
    "epochs=50",
    "lr=2e-4",
    "min_lr=1e-6",
    "warmup_epochs=3",
    "weight_decay=2e-4",
    "dropout=0.15",
    "loss=bce_dice",
    "bce_weight=0.6",
    "dice_weight=0.4",
    "pos_weight=1.2",
    "positive_patch_sampling=true",
    "positive_patch_ratio=0.65",
    "positive_patch_min_fraction=0.001",
    "patch_size=128",
    "train_stride=128",
    "eval_stride=128",
    "mixed_precision=true",
    "tta=true",
    "tta_modes=['hflip','vflip','rot90']",
    "gaussian_noise=true",
    "gaussian_noise_p=0.10",
    "gaussian_noise_std=0.01",
    "s2_brightness_jitter=true",
    "s2_brightness_jitter_p=0.15",
    "s2_contrast_jitter=true",
    "s2_contrast_jitter_p=0.15",
    "horizontal_flip=true",
    "vertical_flip=true",
    "rotate90=true",
    "early_stopping_patience=18",
    "visualization_every_n_epochs=0",
    "visualization_num_samples=0",
    "save_input_previews=false",
]


def _parse_model_candidates(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _metric_value(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key, float("-inf"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _counts_for_threshold(probability: np.ndarray, target: np.ndarray, ignore_mask: np.ndarray, threshold: float) -> dict[str, int]:
    valid = ~ignore_mask
    pred = (probability >= threshold) & valid
    truth = (target >= 0.5) & valid
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = float(counts["tp"])
    fp = float(counts["fp"])
    fn = float(counts["fn"])
    tn = float(counts["tn"])
    iou = _safe_div(tp, tp + fp + fn)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * tp, 2.0 * tp + fp + fn)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def _parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        value = float(stripped)
        if not (0.0 < value < 1.0):
            raise ValueError(f"Threshold must be in (0,1): {value}")
        values.append(value)
    unique = sorted(set(values))
    if not unique:
        raise ValueError("No thresholds were parsed")
    return unique


def evaluate_checkpoint_thresholds(
    *,
    checkpoint_path: Path,
    thresholds: list[float],
    use_tta: bool | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = ExperimentConfig(**checkpoint["config"])
    preprocessing_dir = Path(checkpoint["preprocessing_dir"])
    val_tile_ids = load_json(preprocessing_dir / "val_tiles.json")

    device = _resolve_device(config.device)
    model = build_model(
        config.model,
        in_channels=int(checkpoint["in_channels"]),
        dropout=config.dropout,
        stochastic_depth=config.stochastic_depth,
        pretrained=config.encoder_pretrained,
    ).to(device)
    state_dict = checkpoint["ema_state_dict"] or checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    threshold_counts: dict[float, dict[str, int]] = {threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for threshold in thresholds}
    tta = bool(config.tta if use_tta is None else use_tta)

    with torch.no_grad():
        for tile_id in val_tile_ids:
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
                tta=tta,
                tta_modes=config.tta_modes,
            )
            target = np.load(get_target_path(preprocessing_dir, tile_id)).astype(np.float32)
            ignore_mask = np.load(get_ignore_mask_path(preprocessing_dir, tile_id)).astype(bool) | ~valid_mask
            for threshold in thresholds:
                counts = _counts_for_threshold(probability, target, ignore_mask, threshold)
                current = threshold_counts[threshold]
                current["tp"] += counts["tp"]
                current["fp"] += counts["fp"]
                current["fn"] += counts["fn"]
                current["tn"] += counts["tn"]

    per_threshold = []
    for threshold in thresholds:
        metrics = _metrics_from_counts(threshold_counts[threshold])
        per_threshold.append({"threshold": threshold, **metrics})
    per_threshold.sort(key=lambda item: (float(item["iou"]), float(item["f1"]), float(item["precision"])), reverse=True)
    best = per_threshold[0]
    return {
        "best_threshold": float(best["threshold"]),
        "best_iou": float(best["iou"]),
        "best_f1": float(best["f1"]),
        "best_precision": float(best["precision"]),
        "best_recall": float(best["recall"]),
        "best_accuracy": float(best["accuracy"]),
        "threshold_metrics": per_threshold,
    }


def _ranking_key(summary: dict[str, Any], metric: str) -> tuple[float, float, float, float]:
    metric_name = f"best_val_{metric}"
    return (
        _metric_value(summary, metric_name),
        _metric_value(summary, "best_val_iou"),
        _metric_value(summary, "best_val_f1"),
        _metric_value(summary, "best_val_dice"),
    )


def _print_config(config: ExperimentConfig, label: str) -> None:
    payload = asdict(config)
    payload["stage"] = label
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="data/makeathon-challenge")
    parser.add_argument("--preprocessing-dir", type=Path, default=Path("output/preprocessing_leaderboard"))
    parser.add_argument("--output-root", type=Path, default=Path("output/training_runs_leaderboard"))
    parser.add_argument("--search-output-root", type=Path, default=Path("output/training_runs_leaderboard/search"))
    parser.add_argument("--final-output-root", type=Path, default=Path("output/training_runs_leaderboard/final"))
    parser.add_argument("--model-candidates", type=str, default=",".join(DEFAULT_MODEL_CANDIDATES))
    parser.add_argument("--selection-metric", type=str, choices=["iou", "f1", "dice"], default="iou")
    parser.add_argument("--calibrate-threshold", action="store_true")
    parser.add_argument("--thresholds", type=str, default="0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--search-epochs", type=int, default=12)
    parser.add_argument("--final-epochs", type=int, default=0)
    parser.add_argument("--search-tta", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=0)
    args, overrides = parser.parse_known_args()
    thresholds = _parse_thresholds(args.thresholds)

    base_config = parse_cli_overrides(
        DEFAULT_LEADERBOARD_TRAIN_OVERRIDES + overrides,
        config_cls=ExperimentConfig,
    )
    base_config = replace(
        base_config,
        data_root=args.data_root,
        preprocessing_dir=str(args.preprocessing_dir),
        output_root=str(args.output_root),
    )

    candidate_models = _parse_model_candidates(args.model_candidates)
    if args.max_candidates > 0:
        candidate_models = candidate_models[: args.max_candidates]
    if not candidate_models:
        raise ValueError("No model candidates were provided")

    args.output_root.mkdir(parents=True, exist_ok=True)

    search_results: list[dict[str, Any]] = []
    successful_search_results: list[dict[str, Any]] = []

    if not args.skip_search:
        LOGGER.info("Starting model search over %d candidate(s)", len(candidate_models))
        for model_name in candidate_models:
            search_config = replace(
                base_config,
                model=model_name,
                output_root=str(args.search_output_root),
                epochs=int(args.search_epochs),
                tta=bool(args.search_tta),
                run_name=f"search_{model_name}_{int(args.search_epochs)}ep",
                visualization_every_n_epochs=0,
                visualization_num_samples=0,
                save_input_previews=False,
            )
            _print_config(search_config, label=f"search:{model_name}")

            try:
                summary = train_model(search_config)
                calibrated: dict[str, Any] | None = None
                if args.calibrate_threshold:
                    best_checkpoint = Path(summary["run_dir"]) / "checkpoints" / "best.pt"
                    calibrated = evaluate_checkpoint_thresholds(
                        checkpoint_path=best_checkpoint,
                        thresholds=thresholds,
                        use_tta=search_config.tta,
                    )
                record = {
                    "status": "ok",
                    "model": model_name,
                    "summary": summary,
                    "threshold_calibration": calibrated,
                    "ranking_key": (
                        (
                            float(calibrated["best_iou"]),
                            float(calibrated["best_f1"]),
                            float(calibrated["best_precision"]),
                            float(calibrated["best_recall"]),
                        )
                        if calibrated is not None
                        else _ranking_key(summary, args.selection_metric)
                    ),
                }
                successful_search_results.append(record)
                search_results.append(record)
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                LOGGER.exception("Model search failed for %s", model_name)
                search_results.append(
                    {
                        "status": "failed",
                        "model": model_name,
                        "error": repr(exc),
                    }
                )

        search_payload = {
            "selection_metric": args.selection_metric,
            "candidates": candidate_models,
            "results": search_results,
        }
        save_json(args.output_root / "model_search_summary.json", search_payload)

        if not successful_search_results:
            raise RuntimeError("Model search completed but all candidates failed")

        successful_search_results.sort(key=lambda item: item["ranking_key"], reverse=True)
        best_search = successful_search_results[0]
        selected_model = str(best_search["model"])
        LOGGER.info(
            "Selected model from search: %s (best_val_iou=%.4f best_val_f1=%.4f best_val_dice=%.4f)",
            selected_model,
            _metric_value(best_search["summary"], "best_val_iou"),
            _metric_value(best_search["summary"], "best_val_f1"),
            _metric_value(best_search["summary"], "best_val_dice"),
        )
    else:
        selected_model = candidate_models[0]
        LOGGER.info("Search skipped; selected first candidate model: %s", selected_model)

    if args.search_only:
        save_json(
            args.output_root / "selection_summary.json",
            {
                "mode": "search_only",
                "selected_model": selected_model,
                "selection_metric": args.selection_metric,
            },
        )
        return

    final_epochs = int(args.final_epochs) if int(args.final_epochs) > 0 else int(base_config.epochs)
    final_config = replace(
        base_config,
        model=selected_model,
        output_root=str(args.final_output_root),
        epochs=final_epochs,
        run_name=f"final_{selected_model}_{final_epochs}ep",
    )
    _print_config(final_config, label="final")
    final_summary = train_model(final_config)
    final_calibrated: dict[str, Any] | None = None
    if args.calibrate_threshold:
        final_checkpoint = Path(final_summary["run_dir"]) / "checkpoints" / "best.pt"
        final_calibrated = evaluate_checkpoint_thresholds(
            checkpoint_path=final_checkpoint,
            thresholds=thresholds,
            use_tta=final_config.tta,
        )

    save_json(
        args.output_root / "selection_summary.json",
        {
            "mode": "search_then_final" if not args.skip_search else "final_only",
            "selected_model": selected_model,
            "selection_metric": args.selection_metric,
            "final_epochs": final_epochs,
            "final_summary": final_summary,
            "final_threshold_calibration": final_calibrated,
        },
    )


if __name__ == "__main__":
    main()
