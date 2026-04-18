"""Unit tests for temporal feature aggregation math."""

from __future__ import annotations

import unittest

import numpy as np

from xylemx.config import ExperimentConfig
from xylemx.preprocessing.features import aggregate_temporal_stack, build_feature_names


class FeatureAggregationTests(unittest.TestCase):
    def test_temporal_stack_aggregation(self) -> None:
        stack = np.array(
            [
                [[1.0, 2.0], [np.nan, 4.0]],
                [[3.0, 4.0], [5.0, np.nan]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float32,
        )
        weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mean, std, delta = aggregate_temporal_stack(stack, weights, min_observations=2)

        expected_mean = np.array(
            [
                [(1 * 1 + 3 * 2 + 5 * 3) / 6, (2 * 1 + 4 * 2 + 6 * 3) / 6],
                [(5 * 2 + 7 * 3) / 5, (4 * 1 + 8 * 3) / 4],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(mean, expected_mean, rtol=1e-5, atol=1e-5)
        self.assertTrue(np.isfinite(std[0, 0]))
        self.assertGreater(float(delta[0, 0]), 0.0)

    def test_feature_name_count_matches_spec(self) -> None:
        names = build_feature_names(ExperimentConfig(aef_pca_dim=8, temporal_feature_mode="snapshot_pair"))
        self.assertEqual(len(names), 63)
        self.assertEqual(names[0], "s2_early_B01")
        self.assertEqual(names[-1], "aef_delta_pc08")

    def test_feature_name_count_respects_aef_dimension(self) -> None:
        config = ExperimentConfig(aef_pca_dim=4, temporal_feature_mode="snapshot_pair")
        names = build_feature_names(config)
        self.assertEqual(len(names), 51)
        self.assertEqual(names[-1], "aef_delta_pc04")


if __name__ == "__main__":
    unittest.main()
