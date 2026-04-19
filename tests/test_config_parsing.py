"""Regression tests for CLI config parsing."""

from __future__ import annotations

import unittest

from xylemx.config import ExperimentConfig, parse_cli_overrides
from xylemx.temporal.config import TemporalTrainConfig, parse_temporal_cli_overrides


class ConfigParsingTests(unittest.TestCase):
    def test_main_config_list_field_stays_list(self) -> None:
        config = parse_cli_overrides(["tta_modes=[\"hflip\",\"rot90\"]"], config_cls=ExperimentConfig)
        self.assertEqual(config.tta_modes, ["hflip", "rot90"])
        self.assertIsInstance(config.tta_modes, list)

    def test_temporal_config_list_field_stays_list(self) -> None:
        config = parse_temporal_cli_overrides(["export_splits=[\"val\",\"test\"]"], config_cls=TemporalTrainConfig)
        self.assertEqual(config.export_splits, ["val", "test"])
        self.assertIsInstance(config.export_splits, list)

    def test_temporal_config_handles_escaped_string_items(self) -> None:
        config = parse_temporal_cli_overrides(["export_splits=[\\\"val\\\"]"], config_cls=TemporalTrainConfig)
        self.assertEqual(config.export_splits, ["val"])


if __name__ == "__main__":
    unittest.main()
