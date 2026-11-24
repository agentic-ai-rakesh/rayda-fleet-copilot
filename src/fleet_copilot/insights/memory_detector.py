"""Devices constrained by RAM.

"Consistent" means memory use is at or above the high threshold in at least
`consistent_share` of snapshots; "intermittent" means at least
`intermittent_share`. A single spike is not reported.
"""

from __future__ import annotations

from typing import Any

from fleet_copilot.config.settings import InsightSettings, ThresholdSettings
from fleet_copilot.evidence import evidence_from_row, merge_evidence
from fleet_copilot.insights.models import Insight, group_by_device, sort_insights
from fleet_copilot.utils.stats_utils import share
from fleet_copilot.utils.unit_utils import bytes_to_gb

INSIGHT_TYPE = "memory_pressure"
_FIELDS = ("memory_used_pct",)


def detect_memory_pressure(
    rows: list[dict[str, Any]], thresholds: ThresholdSettings, settings: InsightSettings
) -> tuple[list[Insight], dict[str, Any]]:
    limit = thresholds.high_memory_used_pct
    insights: list[Insight] = []
    checked, skipped = 0, []
    for device_id, series in group_by_device(rows).items():
        if len(series) < settings.min_snapshots:
            skipped.append(device_id)
            continue
        checked += 1
        values = [r["memory_used_pct"] for r in series]
        high_share = share([v >= limit for v in values])
        if high_share < settings.intermittent_share:
            continue

        consistent = high_share >= settings.consistent_share
        peak = max(series, key=lambda r: r["memory_used_pct"])
        average = sum(values) / len(values)
        first_high = next(r for r in series if r["memory_used_pct"] >= limit)
        latest = series[-1]
        ram_gb = bytes_to_gb(latest["memory_total_bytes"])
        pattern = "consistently" if consistent else "intermittently"
        insights.append(
            Insight(
                insight_type=INSIGHT_TYPE,
                device_id=device_id,
                severity="high" if consistent else "medium",
                finding=(
                    f"{device_id} ({ram_gb:g} GB RAM) is {pattern} memory-constrained: "
                    f"use was ≥{limit:g}% in {high_share:.0%} of {len(series)} snapshots "
                    f"(average {average:.1f}%, peak {peak['memory_used_pct']:.1f}%)."
                ),
                explanation=(
                    f"Sustained memory use above {limit:g}% causes swapping and slowdowns. "
                    f"'Consistent' means ≥{settings.consistent_share:.0%} of snapshots, "
                    f"'intermittent' ≥{settings.intermittent_share:.0%}."
                ),
                metrics={
                    "pattern": "consistent" if consistent else "intermittent",
                    "ram_gb": ram_gb,
                    "share_of_snapshots_high": round(high_share, 2),
                    "average_used_pct": round(average, 1),
                    "peak_used_pct": peak["memory_used_pct"],
                    "latest_used_pct": latest["memory_used_pct"],
                    "threshold_pct": limit,
                    "snapshots": len(series),
                },
                evidence=merge_evidence(
                    evidence_from_row(r, _FIELDS) for r in (first_high, peak, latest)
                ),
            )
        )
    return sort_insights(insights), {"devices_checked": checked, "devices_skipped": skipped}
