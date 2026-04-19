"""Normalization helpers for engineered feature cubes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class RunningChannelStats:
    """Streaming per-channel mean and standard deviation."""

    num_channels: int
    sum: np.ndarray = field(init=False)
    sum_sq: np.ndarray = field(init=False)
    count: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.sum = np.zeros(self.num_channels, dtype=np.float64)
        self.sum_sq = np.zeros(self.num_channels, dtype=np.float64)
        self.count = np.zeros(self.num_channels, dtype=np.float64)

    def update(self, array: np.ndarray, valid_mask: np.ndarray | None = None) -> None:
        """Update streaming statistics from a channel-first array."""

        if array.ndim != 3:
            raise ValueError(f"Expected a [C, H, W] array, got {array.shape}")
        if valid_mask is None:
            valid_mask = np.ones(array.shape[1:], dtype=bool)

        for channel_index in range(array.shape[0]):
            values = array[channel_index][valid_mask]
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            self.sum[channel_index] += values.sum(dtype=np.float64)
            self.sum_sq[channel_index] += np.square(values, dtype=np.float64).sum(dtype=np.float64)
            self.count[channel_index] += values.size

    def finalize(
        self,
        feature_names: list[str],
        clip_lower: list[float],
        clip_upper: list[float],
        *,
        min_std: float = 1e-2,
    ) -> dict[str, list[float]]:
        """Return a serializable normalization artifact."""

        safe_count = np.maximum(self.count, 1.0)
        mean = self.sum / safe_count
        variance = np.maximum(self.sum_sq / safe_count - np.square(mean), 0.0)
        std = np.sqrt(variance)
        missing = self.count <= 0
        mean[missing] = 0.0
        std[missing] = 1.0
        std = np.maximum(std, float(min_std))
        return {
            "feature_names": feature_names,
            "mean": mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "fill_value": mean.astype(np.float32).tolist(),
            "clip_lower": [float(value) for value in clip_lower],
            "clip_upper": [float(value) for value in clip_upper],
        }


@dataclass(slots=True)
class ReservoirPercentileEstimator:
    """Approximate per-channel percentiles via bounded random sampling."""

    num_channels: int
    max_samples_per_channel: int
    seed: int = 42
    samples: list[np.ndarray] = field(init=False)
    counts: np.ndarray = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.samples = [np.empty((0,), dtype=np.float32) for _ in range(self.num_channels)]
        self.counts = np.zeros(self.num_channels, dtype=np.int64)
        self.rng = np.random.default_rng(self.seed)

    def update(self, array: np.ndarray, valid_mask: np.ndarray | None = None) -> None:
        """Sample finite values from a [C, H, W] feature tensor."""

        if valid_mask is None:
            valid_mask = np.ones(array.shape[1:], dtype=bool)
        for channel_index in range(array.shape[0]):
            values = array[channel_index][valid_mask]
            values = values[np.isfinite(values)].astype(np.float32, copy=False)
            if values.size == 0:
                continue

            self.counts[channel_index] += values.size
            if values.size > self.max_samples_per_channel:
                values = self.rng.choice(values, size=self.max_samples_per_channel, replace=False)

            current = self.samples[channel_index]
            merged = np.concatenate([current, values], axis=0)
            if merged.size > self.max_samples_per_channel:
                merged = self.rng.choice(merged, size=self.max_samples_per_channel, replace=False)
            self.samples[channel_index] = merged

    def finalize(self, lower_percentile: float, upper_percentile: float) -> tuple[list[float], list[float]]:
        """Return approximate lower and upper clip percentiles."""

        lowers: list[float] = []
        uppers: list[float] = []
        for values in self.samples:
            if values.size == 0:
                lowers.append(0.0)
                uppers.append(1.0)
                continue
            lowers.append(float(np.percentile(values, lower_percentile)))
            uppers.append(float(np.percentile(values, upper_percentile)))
        return lowers, uppers


def clip_array(array: np.ndarray, clip_lower: list[float], clip_upper: list[float]) -> np.ndarray:
    """Clip each feature channel in-place to train-fitted bounds."""

    lower = np.asarray(clip_lower, dtype=np.float32)[:, None, None]
    upper = np.asarray(clip_upper, dtype=np.float32)[:, None, None]
    clipped = array.astype(np.float32, copy=True)
    finite = np.isfinite(clipped)
    clipped = np.where(finite, np.minimum(np.maximum(clipped, lower), upper), clipped)
    return clipped


def normalize_array(array: np.ndarray, stats: dict[str, list[float] | list[str]], normalization: str = "zscore") -> np.ndarray:
    """Apply clipping, train-mean fill, and z-score normalization."""

    prepared = clip_array(
        array,
        clip_lower=list(stats["clip_lower"]),
        clip_upper=list(stats["clip_upper"]),
    )
    fill_value = np.asarray(stats["fill_value"], dtype=np.float32)[:, None, None]
    prepared = np.where(np.isfinite(prepared), prepared, fill_value)

    if normalization == "none":
        return prepared.astype(np.float32)
    if normalization != "zscore":
        raise ValueError(f"Unsupported normalization: {normalization}")

    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
    normalized = ((prepared - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
