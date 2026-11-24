"""Batteries at or approaching end of life.

The dataset has no design capacity, so health is judged from three signals:
- condition reported by the OS (e.g. "Service Recommended")   -> strong
- cycle count at or above the rated limit                       -> strong
- cycle count above the warning share of the limit, or
  full-charge capacity falling by more than the threshold       -> early warning
Snapshots without battery data are skipped, never treated as zero.
"""

from __future__ import annotations

from typing import Any

from fleet_copilot.config.settings import InsightSettings, ThresholdSettings
from fleet_copilot.evidence import evidence_from_row, merge_evidence
from fleet_copilot.insights.models import Insight, group_by_device, sort_insights
from fleet_copilot.utils.stats_utils import daily_slope, days_between

INSIGHT_TYPE = "battery_end_of_life"
_FIELDS = ("battery_cycle_count", "battery_condition", "battery_full_charge_capacity")


def detect_battery_end_of_life(
    rows: list[dict[str, Any]], thresholds: ThresholdSettings, settings: InsightSettings
) -> tuple[list[Insight], dict[str, Any]]:
    """Return insights plus coverage (devices checked / skipped)."""
    insights: list[Insight] = []
    checked, skipped = 0, []
    for device_id, series in group_by_device(rows).items():
        readings = [r for r in series if r["battery_reported"]]
        if len(readings) < settings.min_snapshots:
            skipped.append(device_id)
            continue
        checked += 1
        insight = _assess(device_id, readings, len(series), thresholds)
        if insight:
            insights.append(insight)
    coverage = {"devices_checked": checked, "devices_skipped_insufficient_battery_data": skipped}
    return sort_insights(insights), coverage


def _assess(
    device_id: str,
    readings: list[dict[str, Any]],
    total_snapshots: int,
    thresholds: ThresholdSettings,
) -> Insight | None:
    first, latest = readings[0], readings[-1]
    cycles = latest["battery_cycle_count"]
    condition = latest["battery_condition"]
    limit = thresholds.battery_cycle_limit
    service_conditions = set(thresholds.battery_service_conditions)

    capacity_first = first["battery_full_charge_capacity"]
    capacity_latest = latest["battery_full_charge_capacity"]
    capacity_drop_pct = (
        round(100 * (capacity_first - capacity_latest) / capacity_first, 1)
        if capacity_first and capacity_latest is not None
        else None
    )
    slope = daily_slope(
        [r["collected_at"] for r in readings],
        [r["battery_full_charge_capacity"] for r in readings],
    )

    needs_service = condition in service_conditions
    over_limit = cycles is not None and cycles >= limit
    near_limit = cycles is not None and cycles >= thresholds.battery_cycle_warning_ratio * limit
    declining = (
        capacity_drop_pct is not None and capacity_drop_pct >= thresholds.battery_capacity_drop_pct
    )

    reasons = []
    if needs_service:
        reasons.append(f"the OS reports condition '{condition}'")
    if over_limit:
        reasons.append(f"{cycles} charge cycles is at or past the rated {limit}")
    elif near_limit:
        reasons.append(
            f"{cycles} charge cycles is over "
            f"{thresholds.battery_cycle_warning_ratio:.0%} of the rated {limit}"
        )
    if declining:
        reasons.append(
            f"full-charge capacity fell {capacity_drop_pct}% "
            f"({capacity_first} → {capacity_latest} mAh)"
        )
    if not reasons:
        return None

    end_of_life = needs_service or over_limit
    window_days = round(days_between(first["collected_at"], latest["collected_at"]))
    first_service = next(
        (r for r in readings if r["battery_condition"] in service_conditions), None
    )
    evidence = merge_evidence(
        evidence_from_row(r, _FIELDS) for r in (first, first_service, latest) if r is not None
    )
    status = "at end of life" if end_of_life else "approaching end of life"
    return Insight(
        insight_type=INSIGHT_TYPE,
        device_id=device_id,
        severity="high" if end_of_life else "medium",
        finding=f"Battery on {device_id} is {status}: " + "; ".join(reasons) + ".",
        explanation=(
            "Batteries past their rated cycle life or flagged for service hold less charge "
            f"and can fail unexpectedly. Judged over {len(readings)} readings across "
            f"{window_days} days."
        ),
        metrics={
            "status": "end_of_life" if end_of_life else "approaching_end_of_life",
            "cycle_count": cycles,
            "cycle_limit": limit,
            "condition": condition,
            "capacity_first_mah": capacity_first,
            "capacity_latest_mah": capacity_latest,
            "capacity_drop_pct": capacity_drop_pct,
            "capacity_change_mah_per_day": None if slope is None else round(slope, 2),
            "first_service_condition_at": first_service["collected_at"] if first_service else None,
            "readings_used": len(readings),
            "readings_missing": total_snapshots - len(readings),
        },
        evidence=evidence,
    )
