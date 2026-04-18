"""Experiment utilities for run directories, logging, and serialization."""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from xylemx.config import ExperimentConfig


def slugify(value: str) -> str:
    """Create a filesystem-friendly slug."""

    text = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower()


def utc_timestamp() -> str:
    """Return a UTC timestamp for run names."""

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_run_name(config: ExperimentConfig) -> str:
    """Build a run name that matches the required timestamped format."""

    prefix = utc_timestamp()
    if config.run_name:
        return f"{prefix}_{slugify(config.run_name)}"

    parts = [prefix, slugify(config.model)]
    if config.short_tag:
        parts.append(slugify(config.short_tag))
    return "_".join(part for part in parts if part)


def create_run_directory(config: ExperimentConfig) -> Path:
    """Create the full run directory tree and return the run path."""

    run_dir = Path(config.output_root) / build_run_name(config)
    for child in [
        run_dir,
        run_dir / "artifacts",
        run_dir / "artifacts" / "preprocessing",
        run_dir / "checkpoints",
        run_dir / "metrics",
        run_dir / "predictions" / "val",
        run_dir / "predictions" / "test",
        run_dir / "visualizations",
        run_dir / "visualizations" / "best",
    ]:
        child.mkdir(parents=True, exist_ok=True)
    return run_dir


def configure_logging(log_path: str | Path) -> None:
    """Configure root logging for both console and file output."""

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def save_json(path: str | Path, payload: Any) -> None:
    """Write JSON with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_yaml(path: str | Path, payload: Any) -> None:
    """Write YAML with stable formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)


def save_resolved_config(config: ExperimentConfig, run_dir: str | Path) -> None:
    """Persist config snapshots in YAML and JSON form."""

    run_dir = Path(run_dir)
    save_yaml(run_dir / "config.yaml", asdict(config))
    save_json(run_dir / "config.json", asdict(config))


def append_run_record(output_root: str | Path, record: dict[str, Any]) -> None:
    """Append a finished-run record to shared JSON and CSV leaderboards."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    leaderboard_json = output_root / "leaderboard.json"
    existing: list[dict[str, Any]]
    if leaderboard_json.exists():
        with leaderboard_json.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        existing = loaded if isinstance(loaded, list) else []
    else:
        existing = []

    run_name = str(record.get("run_name", ""))
    existing = [item for item in existing if str(item.get("run_name", "")) != run_name]
    existing.append(record)
    existing.sort(
        key=lambda item: (
            -float(item.get("best_val_dice", float("-inf"))),
            -float(item.get("best_val_iou", float("-inf"))),
            float(item.get("duration_seconds", float("inf"))),
            str(item.get("run_name", "")),
        )
    )
    save_json(leaderboard_json, existing)

    fieldnames: list[str] = []
    for item in existing:
        for key in item:
            if key not in fieldnames:
                fieldnames.append(key)

    leaderboard_csv = output_root / "leaderboard.csv"
    with leaderboard_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in existing:
            writer.writerow(item)
