"""Temporal label binning and weak-date aggregation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _parse_month_token(token: str) -> tuple[int, int]:
    year_raw, month_raw = token.split("-", 1)
    return int(year_raw), int(month_raw)


def month_ordinal(year: int, month: int) -> int:
    return (year - 1970) * 12 + (month - 1)


def iter_months(start: str, end: str, *, step_months: int = 1) -> list[tuple[int, int]]:
    """Return inclusive ``(year, month)`` steps between two ``YYYY-MM`` tokens."""

    if step_months < 1:
        raise ValueError(f"step_months must be >= 1, got {step_months}")

    start_year, start_month = _parse_month_token(start)
    end_year, end_month = _parse_month_token(end)
    current = month_ordinal(start_year, start_month)
    stop = month_ordinal(end_year, end_month)
    if current > stop:
        raise ValueError(f"time_start must be <= time_end, got {start} > {end}")
    months: list[tuple[int, int]] = []
    while current <= stop:
        year = 1970 + (current // 12)
        month = (current % 12) + 1
        months.append((year, month))
        current += step_months
    return months


@dataclass(slots=True)
class TimeBinSpec:
    """Resolved mapping from dates to temporal class indices."""

    mode: str
    labels: list[str]
    ignore_index: int
    start_year: int
    end_year: int
    start_month_ordinal: int
    end_month_ordinal: int

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    def to_payload(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "labels": self.labels,
            "ignore_index": self.ignore_index,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "start_month_ordinal": self.start_month_ordinal,
            "end_month_ordinal": self.end_month_ordinal,
            "num_classes": self.num_classes,
        }


def build_time_bin_spec(mode: str, *, start: str, end: str, ignore_index: int = -1) -> TimeBinSpec:
    """Build a stable time-bin specification from a calendar range."""

    months = iter_months(start, end)
    start_year, _ = months[0]
    end_year, _ = months[-1]
    start_ordinal = month_ordinal(*months[0])
    end_ordinal = month_ordinal(*months[-1])
    mode = mode.lower()
    if mode == "year":
        labels = [str(year) for year in range(start_year, end_year + 1)]
    elif mode == "month":
        labels = [f"{year:04d}-{month:02d}" for year, month in months]
    elif mode == "quarter":
        labels = []
        for year in range(start_year, end_year + 1):
            for quarter in range(1, 5):
                quarter_start = month_ordinal(year, (quarter - 1) * 3 + 1)
                quarter_end = month_ordinal(year, (quarter - 1) * 3 + 3)
                if quarter_end < start_ordinal or quarter_start > end_ordinal:
                    continue
                labels.append(f"{year:04d}-Q{quarter}")
    else:
        raise ValueError(f"Unsupported time_bin_mode: {mode}")
    return TimeBinSpec(
        mode=mode,
        labels=labels,
        ignore_index=ignore_index,
        start_year=start_year,
        end_year=end_year,
        start_month_ordinal=start_ordinal,
        end_month_ordinal=end_ordinal,
    )


def filter_dates_to_range(dates: np.ndarray, spec: TimeBinSpec) -> np.ndarray:
    """Keep only dates that fall inside the configured bin range."""

    filtered = np.full(dates.shape, np.datetime64("NaT"), dtype="datetime64[D]")
    valid = ~np.isnat(dates)
    if not valid.any():
        return filtered
    months = dates[valid].astype("datetime64[M]").astype(np.int64)
    in_range = (months >= spec.start_month_ordinal) & (months <= spec.end_month_ordinal)
    filtered_indices = np.flatnonzero(valid.reshape(-1))[in_range]
    filtered_flat = filtered.reshape(-1)
    dates_flat = dates.reshape(-1)
    filtered_flat[filtered_indices] = dates_flat[filtered_indices]
    return filtered


def merge_event_dates(
    candidate_dates: list[np.ndarray],
    candidate_confidences: list[np.ndarray],
    candidate_validity: list[np.ndarray],
    *,
    strategy: str,
) -> np.ndarray:
    """Merge multiple weak event-date candidates into a single calendar date raster."""

    if not candidate_dates:
        raise ValueError("candidate_dates must not be empty")

    shape = candidate_dates[0].shape
    merged = np.full(shape, np.datetime64("NaT"), dtype="datetime64[D]")
    strategy = strategy.lower()

    if strategy == "highest_confidence":
        best_confidence = np.full(shape, -1.0, dtype=np.float32)
        best_days = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
        for dates, confidences, valid in zip(candidate_dates, candidate_confidences, candidate_validity, strict=True):
            usable = valid & ~np.isnat(dates)
            if not usable.any():
                continue
            days = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
            days[usable] = dates[usable].astype("datetime64[D]").astype(np.int64)
            replace = usable & (
                (confidences > best_confidence)
                | ((confidences == best_confidence) & (days < best_days))
            )
            best_confidence[replace] = confidences[replace]
            best_days[replace] = days[replace]
        valid = best_confidence >= 0
        merged[valid] = best_days[valid].astype("datetime64[D]")
        return merged

    if strategy == "earliest":
        best_days = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
        for dates, valid in zip(candidate_dates, candidate_validity, strict=True):
            usable = valid & ~np.isnat(dates)
            if not usable.any():
                continue
            days = np.full(shape, np.iinfo(np.int64).max, dtype=np.int64)
            days[usable] = dates[usable].astype("datetime64[D]").astype(np.int64)
            replace = usable & (days < best_days)
            best_days[replace] = days[replace]
        valid = best_days < np.iinfo(np.int64).max
        merged[valid] = best_days[valid].astype("datetime64[D]")
        return merged

    if strategy == "median":
        stacked: list[np.ndarray] = []
        for dates, valid in zip(candidate_dates, candidate_validity, strict=True):
            values = np.full(shape, np.nan, dtype=np.float64)
            usable = valid & ~np.isnat(dates)
            if usable.any():
                values[usable] = dates[usable].astype("datetime64[D]").astype(np.int64).astype(np.float64)
            stacked.append(values)
        with np.errstate(all="ignore"):
            median_days = np.nanmedian(np.stack(stacked, axis=0), axis=0)
        valid = np.isfinite(median_days)
        merged[valid] = np.rint(median_days[valid]).astype(np.int64).astype("datetime64[D]")
        return merged

    raise ValueError(f"Unsupported time merge strategy: {strategy}")


def dates_to_bin_indices(dates: np.ndarray, spec: TimeBinSpec) -> np.ndarray:
    """Convert decoded event dates into class indices for the configured bins."""

    indices = np.full(dates.shape, spec.ignore_index, dtype=np.int64)
    valid = ~np.isnat(dates)
    if not valid.any():
        return indices

    years = dates[valid].astype("datetime64[Y]").astype(np.int64) + 1970
    months = dates[valid].astype("datetime64[M]").astype(np.int64)

    if spec.mode == "year":
        mapped = years - spec.start_year
    elif spec.mode == "month":
        mapped = months - spec.start_month_ordinal
    elif spec.mode == "quarter":
        mapped = ((months - spec.start_month_ordinal) // 3).astype(np.int64)
    else:
        raise ValueError(f"Unsupported mode: {spec.mode}")

    flat = indices.reshape(-1)
    valid_flat = valid.reshape(-1)
    mapped = mapped.astype(np.int64, copy=False)
    mapped_valid = (mapped >= 0) & (mapped < spec.num_classes)
    target_positions = np.argwhere(valid_flat).ravel()[mapped_valid]
    flat[target_positions] = mapped[mapped_valid]
    return indices
