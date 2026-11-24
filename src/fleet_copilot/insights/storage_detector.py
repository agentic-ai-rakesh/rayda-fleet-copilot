"""Devices low on, or running out of, system disk space.

A device is reported when its latest free space is below the low threshold,
or it was low in at least `intermittent_share` of snapshots. The daily trend
gives a projection of days until the critical threshold.
"""

from __future__ import annotations

from typing import Any

from fleet_copilot.config.settings import InsightSettings, ThresholdSettings
from fleet_copilot.evidence import evidence_from_row, merge_evidence
from fleet_copilot.insights.models import Insight, group_by_device, sort_insights
from fleet_copilot.utils.stats_utils import daily_slope, share

INSIGHT_TYPE = "storage_pressure"
_FIELDS = ("disk_free_pct", "disk_available_bytes")


def detect_storage_pressure(
    rows: list[dict[str, Any]], thresholds: ThresholdSettings, settings: InsightSettings
) -> tuple[list[Insight], dict[str, Any]]:
    low, critical = thresholds.low_disk_free_pct, thresholds.critical_disk_free_pct
    insights: list[Insight] = []
    checked, skipped = 0, []
    for device_id, series in group_by_device(rows).items():
        series = [r for r in series if r.get("disk_free_pct") is not None]
        if len(series) < settings.min_snapshots:
            skipped.append(device_id)
            continue
        checked += 1
        values = [r["disk_free_pct"] for r in series]
        latest = series[-1]
        low_share = share([v < low for v in values])
        currently_low = latest["disk_free_pct"] < low
        if not currently_low and low_share < settings.intermittent_share:
            continue

        slope = daily_slope([r["collected_at"] for r in series], values)
        days_to_critical = _days_until(latest["disk_free_pct"], critical, slope)
        pattern = (
            "consistent"
            if low_share >= settings.consistent_share
            else "intermittent"
            if low_share >= settings.intermittent_share
            else "recent"
        )
        first_low = next((r for r in series if r["disk_free_pct"] < low), None)
        trend_text = (
            f"falling {abs(slope):.2f} points/day"
            if slope is not None and slope < 0
            else "not falling"
        )
        at_critical = latest["disk_free_pct"] <= critical
        critical_within_week = days_to_critical is not None and days_to_critical <= 7
        insights.append(
            Insight(
                insight_type=INSIGHT_TYPE,
                device_id=device_id,
                severity="high" if at_critical or critical_within_week else "medium",
                finding=(
                    f"{device_id} has {latest['disk_free_pct']:.1f}% system disk free "
                    f"(was {series[0]['disk_free_pct']:.1f}%), below {low:g}% in "
                    f"{low_share:.0%} of {len(series)} snapshots; {trend_text}."
                ),
                explanation=(
                    f"Below {low:g}% free, updates and apps start failing; at {critical:g}% the "
                    "system is close to full. "
                    + (
                        "Already at or below the critical level."
                        if at_critical
                        else f"At the current rate it reaches {critical:g}% in about "
                        f"{days_to_critical:.0f} days."
                        if days_to_critical is not None
                        else "No downward trend to project."
                    )
                ),
                metrics={
                    "pattern": pattern,
                    "latest_free_pct": latest["disk_free_pct"],
                    "first_free_pct": series[0]["disk_free_pct"],
                    "min_free_pct": min(values),
                    "share_of_snapshots_low": round(low_share, 2),
                    "free_pct_change_per_day": None if slope is None else round(slope, 3),
                    "days_until_critical": days_to_critical,
                    "low_threshold_pct": low,
                    "critical_threshold_pct": critical,
                    "snapshots": len(series),
                },
                evidence=merge_evidence(
                    evidence_from_row(r, _FIELDS)
                    for r in (series[0], first_low, latest)
                    if r is not None
                ),
            )
        )
    return sort_insights(insights), {"devices_checked": checked, "devices_skipped": skipped}


def _days_until(current: float, target: float, slope_per_day: float | None) -> float | None:
    if current <= target:
        return 0.0
    if slope_per_day is None or slope_per_day >= 0:
        return None
    return round((current - target) / -slope_per_day, 1)
