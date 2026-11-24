"""Small statistics helpers for time-series telemetry (no numpy needed at this size)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from fleet_copilot.utils.time_utils import ensure_utc


def parse_iso(value: str) -> datetime:
    return ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def days_between(start_iso: str, end_iso: str) -> float:
    return (parse_iso(end_iso) - parse_iso(start_iso)).total_seconds() / 86_400


def linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Least-squares slope of ys over xs; None with fewer than two distinct x values."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    return numerator / denominator


def daily_slope(timestamps: Sequence[str], values: Sequence[float]) -> float | None:
    """Slope in units per day, measured from the first timestamp."""
    if not timestamps:
        return None
    xs = [days_between(timestamps[0], ts) for ts in timestamps]
    return linear_slope(xs, values)


def share(flags: Sequence[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0
