"""Tests for experiment leaderboard tracking."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from xylemx.data.io import load_json
from xylemx.experiment import append_run_record


class ExperimentTrackingTests(unittest.TestCase):
    def test_append_run_record_updates_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            append_run_record(
                output_root,
                {
                    "run_name": "run_b",
                    "model": "resnet18_unet",
                    "best_val_dice": 0.70,
                    "best_val_iou": 0.55,
                    "duration_seconds": 100.0,
                },
            )
            append_run_record(
                output_root,
                {
                    "run_name": "run_a",
                    "model": "resnet34_fpn",
                    "best_val_dice": 0.75,
                    "best_val_iou": 0.60,
                    "duration_seconds": 120.0,
                },
            )

            leaderboard = load_json(output_root / "leaderboard.json")
            self.assertEqual([item["run_name"] for item in leaderboard], ["run_a", "run_b"])

            with (output_root / "leaderboard.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["run_name"] for row in rows], ["run_a", "run_b"])


if __name__ == "__main__":
    unittest.main()
