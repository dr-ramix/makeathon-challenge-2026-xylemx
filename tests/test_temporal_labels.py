"""Unit tests for temporal label helpers."""

from __future__ import annotations

import unittest

import numpy as np

from xylemx.temporal.labels import build_time_bin_spec, dates_to_bin_indices, filter_dates_to_range, iter_months, merge_event_dates


class TemporalLabelTests(unittest.TestCase):
    def test_iter_months_can_stride_quarterly(self) -> None:
        steps = iter_months("2020-01", "2020-12", step_months=3)
        self.assertEqual(steps, [(2020, 1), (2020, 4), (2020, 7), (2020, 10)])

    def test_highest_confidence_merge_prefers_stronger_source(self) -> None:
        first = np.array([["2022-06-15", "2023-02-10"]], dtype="datetime64[D]")
        second = np.array([["2022-05-01", "2023-03-01"]], dtype="datetime64[D]")
        confidences_a = np.array([[0.5, 0.9]], dtype=np.float32)
        confidences_b = np.array([[0.8, 0.7]], dtype=np.float32)
        merged = merge_event_dates(
            [first, second],
            [confidences_a, confidences_b],
            [np.ones((1, 2), dtype=bool), np.ones((1, 2), dtype=bool)],
            strategy="highest_confidence",
        )
        np.testing.assert_array_equal(
            merged,
            np.array([["2022-05-01", "2023-02-10"]], dtype="datetime64[D]"),
        )

    def test_month_binning_filters_dates_outside_range(self) -> None:
        spec = build_time_bin_spec("month", start="2022-01", end="2022-03", ignore_index=-1)
        dates = np.array([["2021-12-31", "2022-02-15", "2022-03-01"]], dtype="datetime64[D]")
        filtered = filter_dates_to_range(dates, spec)
        indices = dates_to_bin_indices(filtered, spec)
        np.testing.assert_array_equal(filtered[0, 0], np.datetime64("NaT"))
        np.testing.assert_array_equal(indices, np.array([[-1, 1, 2]], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
