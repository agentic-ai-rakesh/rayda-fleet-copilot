"""Unit conversions for telemetry values."""

from __future__ import annotations

BYTES_PER_GB = 1024**3


def bytes_to_gb(value: int | None, ndigits: int = 1) -> float | None:
    if value is None:
        return None
    return round(value / BYTES_PER_GB, ndigits)


def percent(part: float | None, whole: float | None, ndigits: int = 2) -> float | None:
    """`part` as a percentage of `whole`; None when it cannot be computed."""
    if part is None or whole is None or whole <= 0:
        return None
    return round(100.0 * part / whole, ndigits)
