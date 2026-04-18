"""Unit tests for weak-label fusion."""

from __future__ import annotations

import unittest

import numpy as np

from xylemx.labels.consensus import (
    fuse_binary_masks,
    gladl_positive_mask,
    glads2_positive_mask,
    radd_positive_mask,
)


class LabelFusionTests(unittest.TestCase):
    def test_radd_modes(self) -> None:
        raw = np.array([[0, 21001, 31001]], dtype=np.uint16)
        np.testing.assert_array_equal(radd_positive_mask(raw, mode="permissive"), [[False, True, True]])
        np.testing.assert_array_equal(radd_positive_mask(raw, mode="conservative"), [[False, False, True]])

    def test_glad_thresholds(self) -> None:
        gladl = gladl_positive_mask([np.array([[0, 1], [2, 3]], dtype=np.uint8)], threshold=2)
        glads2 = glads2_positive_mask(np.array([[0, 1], [2, 4]], dtype=np.uint8), threshold=1)
        np.testing.assert_array_equal(gladl, [[False, False], [True, True]])
        np.testing.assert_array_equal(glads2, [[False, True], [True, True]])

    def test_consensus_weights_and_ignore(self) -> None:
        source_masks = {
            "radd": np.array([[1, 0], [0, 0]], dtype=bool),
            "gladl": np.array([[1, 1], [0, 0]], dtype=bool),
            "glads2": np.array([[0, 0], [0, 1]], dtype=bool),
        }
        source_valid = {name: np.ones((2, 2), dtype=bool) for name in source_masks}
        fused = fuse_binary_masks(
            source_masks,
            source_valid,
            method="consensus_2of3",
            ignore_uncertain_single_source_positives=True,
            vote_weight_0=1.0,
            vote_weight_1=0.3,
            vote_weight_2=0.8,
            vote_weight_3=1.0,
        )

        np.testing.assert_array_equal(fused.target, np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32))
        np.testing.assert_array_equal(fused.vote_count, np.array([[2, 1], [0, 1]], dtype=np.uint8))
        np.testing.assert_array_equal(fused.ignore_mask, np.array([[False, True], [False, True]]))
        self.assertAlmostEqual(float(fused.weight_map[0, 0]), 0.8)
        self.assertAlmostEqual(float(fused.weight_map[1, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
