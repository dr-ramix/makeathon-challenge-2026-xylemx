"""Unit tests for submission export helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from submission_utils import raster_to_geojson


class SubmissionUtilsTests(unittest.TestCase):
    def test_raster_to_geojson_preserves_default_time_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raster_path = Path(tmpdir) / "mask.tif"
            transform = from_origin(0, 4, 1, 1)
            profile = {
                "driver": "GTiff",
                "height": 4,
                "width": 4,
                "count": 1,
                "dtype": "uint8",
                "crs": "EPSG:3857",
                "transform": transform,
                "nodata": 0,
            }
            data = np.zeros((4, 4), dtype=np.uint8)
            data[1:3, 1:3] = 1
            with rasterio.open(raster_path, "w", **profile) as dst:
                dst.write(data, 1)
            geojson = raster_to_geojson(raster_path, min_area_ha=0.0)
            self.assertEqual(geojson["features"][0]["properties"]["time_step"], None)

    def test_raster_to_geojson_aggregates_polygon_time_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raster_path = Path(tmpdir) / "mask.tif"
            time_path = Path(tmpdir) / "time.tif"
            transform = from_origin(0, 4, 1, 1)
            profile = {
                "driver": "GTiff",
                "height": 4,
                "width": 4,
                "count": 1,
                "dtype": "int16",
                "crs": "EPSG:3857",
                "transform": transform,
                "nodata": -1,
            }
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[1:3, 1:3] = 1
            with rasterio.open(raster_path, "w", **{**profile, "dtype": "uint8", "nodata": 0}) as dst:
                dst.write(mask, 1)

            time_bins = np.full((4, 4), -1, dtype=np.int16)
            time_bins[1:3, 1:3] = np.array([[1, 1], [2, 1]], dtype=np.int16)
            with rasterio.open(time_path, "w", **profile) as dst:
                dst.write(time_bins, 1)

            geojson = raster_to_geojson(
                raster_path,
                min_area_ha=0.0,
                time_raster_path=time_path,
                time_labels=["2020", "2021", "2022"],
            )
            self.assertEqual(geojson["features"][0]["properties"]["time_step"], "2021")


if __name__ == "__main__":
    unittest.main()
