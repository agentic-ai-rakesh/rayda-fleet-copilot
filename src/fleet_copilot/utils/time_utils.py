"""UTC timestamp helpers. All timestamps in the app are UTC, ISO 8601, with a `Z` suffix."""

from __future__ import annotations

from datetime import UTC, datetime


def ensure_utc(value: datetime) -> datetime:
    """Treat naive datetimes as UTC and convert aware ones to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_iso_utc(value: datetime) -> str:
    """Format as `2026-05-14T09:54:00Z`."""
    return ensure_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_compact_utc(value: datetime) -> str:
    """Format as `20260514T0954Z`, used inside evidence ids."""
    return ensure_utc(value).strftime("%Y%m%dT%H%MZ")


def utc_now_iso() -> str:
    return to_iso_utc(datetime.now(UTC))
