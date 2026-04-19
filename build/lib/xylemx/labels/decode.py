"""Decoders for the weak-label raster formats used in the challenge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DecodedLabel:
    """Structured, pixel-wise weak-label representation."""

    is_positive: np.ndarray
    confidence_score: np.ndarray
    event_date: np.ndarray
    raw_class: np.ndarray
    is_uncertain: np.ndarray
    valid_mask: np.ndarray


def _empty_dates(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.datetime64("NaT"), dtype="datetime64[D]")


def _date_from_offsets(offsets: np.ndarray, base_date: str) -> np.ndarray:
    dates = _empty_dates(offsets.shape)
    positive = offsets > 0
    if positive.any():
        dates[positive] = np.datetime64(base_date) + offsets[positive].astype("timedelta64[D]")
    return dates


def decode_radd_array(raw: np.ndarray) -> DecodedLabel:
    """Decode a RADD label raster."""

    raw = raw.astype(np.uint16, copy=False)
    leading = raw // 10000
    offset_days = raw % 10000
    is_positive = np.isin(leading, [2, 3]) & (offset_days > 0)
    is_uncertain = np.zeros(raw.shape, dtype=bool)
    confidence = np.zeros(raw.shape, dtype=np.float32)
    confidence[leading == 2] = 0.6
    confidence[leading == 3] = 1.0
    event_date = _date_from_offsets(offset_days.astype(np.int32), "2014-12-31")
    valid_mask = np.ones(raw.shape, dtype=bool)
    return DecodedLabel(
        is_positive=is_positive,
        confidence_score=confidence,
        event_date=event_date,
        raw_class=leading.astype(np.uint8),
        is_uncertain=is_uncertain,
        valid_mask=valid_mask,
    )


def decode_glads2_array(alert: np.ndarray, alert_date: np.ndarray) -> DecodedLabel:
    """Decode GLAD-S2 alert and alertDate rasters."""

    alert = alert.astype(np.uint8, copy=False)
    alert_date = alert_date.astype(np.uint16, copy=False)
    is_positive = (alert >= 2) & (alert_date > 0)
    is_uncertain = alert == 1
    confidence = np.zeros(alert.shape, dtype=np.float32)
    confidence[alert == 1] = 0.2
    confidence[alert == 2] = 0.45
    confidence[alert == 3] = 0.7
    confidence[alert == 4] = 1.0
    event_date = _date_from_offsets(alert_date.astype(np.int32), "2019-01-01")
    valid_mask = np.ones(alert.shape, dtype=bool)
    return DecodedLabel(
        is_positive=is_positive,
        confidence_score=confidence,
        event_date=event_date,
        raw_class=alert,
        is_uncertain=is_uncertain,
        valid_mask=valid_mask,
    )


def decode_gladl_year(alert: np.ndarray, alert_date: np.ndarray, year: int) -> DecodedLabel:
    """Decode one GLAD-L year pair."""

    alert = alert.astype(np.uint8, copy=False)
    alert_date = alert_date.astype(np.uint16, copy=False)
    is_positive = np.isin(alert, [2, 3]) & (alert_date > 0)
    is_uncertain = np.zeros(alert.shape, dtype=bool)
    confidence = np.zeros(alert.shape, dtype=np.float32)
    confidence[alert == 2] = 0.6
    confidence[alert == 3] = 1.0
    event_date = _empty_dates(alert.shape)
    positive = alert_date > 0
    if positive.any():
        # GLAD-L dates are day-of-year in the corresponding calendar year.
        event_date[positive] = np.datetime64(f"{year}-01-01") + (alert_date[positive] - 1).astype("timedelta64[D]")
    valid_mask = np.ones(alert.shape, dtype=bool)
    return DecodedLabel(
        is_positive=is_positive,
        confidence_score=confidence,
        event_date=event_date,
        raw_class=alert,
        is_uncertain=is_uncertain,
        valid_mask=valid_mask,
    )


def combine_decoded_labels(labels: list[DecodedLabel]) -> DecodedLabel:
    """Collapse multiple decoded rasters from the same source into one."""

    if not labels:
        raise ValueError("No decoded labels were provided")

    shape = labels[0].is_positive.shape
    is_positive = np.zeros(shape, dtype=bool)
    is_uncertain = np.zeros(shape, dtype=bool)
    confidence = np.zeros(shape, dtype=np.float32)
    raw_class = np.zeros(shape, dtype=np.uint8)
    valid_mask = np.zeros(shape, dtype=bool)
    event_date = _empty_dates(shape)

    for label in labels:
        is_positive |= label.is_positive
        is_uncertain |= label.is_uncertain
        confidence = np.maximum(confidence, label.confidence_score)
        raw_class = np.maximum(raw_class, label.raw_class.astype(np.uint8))
        valid_mask |= label.valid_mask

        candidate = label.event_date
        replace = ~np.isnat(candidate) & (np.isnat(event_date) | (candidate < event_date))
        event_date[replace] = candidate[replace]

    return DecodedLabel(
        is_positive=is_positive,
        confidence_score=confidence,
        event_date=event_date,
        raw_class=raw_class,
        is_uncertain=is_uncertain,
        valid_mask=valid_mask,
    )
