"""Insight tools: load the tenant's time series and run a detector over it."""

from __future__ import annotations

from typing import Any

from fleet_copilot.evidence import evidence_from_row, merge_evidence
from fleet_copilot.insights.battery_detector import detect_battery_end_of_life
from fleet_copilot.insights.compliance_detector import detect_compliance_drift
from fleet_copilot.insights.memory_detector import detect_memory_pressure
from fleet_copilot.insights.models import Insight
from fleet_copilot.insights.storage_detector import detect_storage_pressure
from fleet_copilot.tools.context import ToolContext
from fleet_copilot.tools.results import ToolOutput
from fleet_copilot.tools.schemas import (
    DetectBatteryEndOfLifeInput,
    DetectComplianceDriftInput,
    DetectMemoryPressureInput,
    DetectStoragePressureInput,
)


def detect_battery_end_of_life_tool(
    ctx: ToolContext, args: DetectBatteryEndOfLifeInput
) -> ToolOutput:
    since = ctx.window_start(args.days)
    rows = ctx.repo.device_state_history(since=since, device_type="laptop")
    insights, coverage = detect_battery_end_of_life(rows, ctx.thresholds, ctx.insights)
    t = ctx.thresholds
    return _output(
        ctx,
        since,
        insights,
        coverage,
        noun="laptop battery",
        checked=(
            rows,
            ("battery_cycle_count", "battery_condition", "battery_full_charge_capacity"),
        ),
        assumptions=[
            f"End of life: condition in {t.battery_service_conditions} or cycle count ≥ "
            f"{t.battery_cycle_limit}.",
            f"Approaching: cycle count ≥ {t.battery_cycle_warning_ratio:.0%} of the limit, or "
            f"full-charge capacity down ≥ {t.battery_capacity_drop_pct:g}% over the window.",
            "No design capacity is reported, so health is not a percentage of original capacity.",
            "Snapshots without battery data are skipped.",
        ],
    )


def detect_memory_pressure_tool(ctx: ToolContext, args: DetectMemoryPressureInput) -> ToolOutput:
    since = ctx.window_start(args.days)
    rows = ctx.repo.device_state_history(since=since)
    insights, coverage = detect_memory_pressure(rows, ctx.thresholds, ctx.insights)
    s = ctx.insights
    return _output(
        ctx,
        since,
        insights,
        coverage,
        noun="memory-constrained device",
        checked=(rows, ("memory_used_pct",)),
        assumptions=[
            f"High memory use: ≥ {ctx.thresholds.high_memory_used_pct:g}% of RAM.",
            f"Consistent: high in ≥ {s.consistent_share:.0%} of snapshots; "
            f"intermittent: ≥ {s.intermittent_share:.0%}.",
        ],
    )


def detect_storage_pressure_tool(ctx: ToolContext, args: DetectStoragePressureInput) -> ToolOutput:
    since = ctx.window_start(args.days)
    rows = ctx.repo.device_state_history(since=since)
    insights, coverage = detect_storage_pressure(rows, ctx.thresholds, ctx.insights)
    t = ctx.thresholds
    return _output(
        ctx,
        since,
        insights,
        coverage,
        noun="device low on storage",
        checked=(rows, ("disk_free_pct",)),
        assumptions=[
            f"Low disk: system volume free space < {t.low_disk_free_pct:g}%; "
            f"critical: ≤ {t.critical_disk_free_pct:g}%.",
            "Days until critical is a straight-line projection of the daily trend.",
        ],
    )


def detect_compliance_drift_tool(ctx: ToolContext, args: DetectComplianceDriftInput) -> ToolOutput:
    since = ctx.window_start(args.days)
    rows = ctx.repo.compliance_history(since=since, check_id=args.check_id)
    insights, coverage = detect_compliance_drift(rows, ctx.insights)
    output = _output(
        ctx,
        since,
        insights,
        coverage,
        noun="compliance change",
        checked=(rows, ("check_id", "severity", "status")),
        assumptions=[
            "Drift = passed at the start of the window and failing at the end.",
            "Devices failing in every snapshot are counted as persistent failures, not drift.",
        ],
    )
    trend = coverage["fleet_trend_by_check"]
    if trend:
        any_check = next(iter(trend.values()))
        changes = "; ".join(
            f"{check} {t['failing_devices_first_day']} → {t['failing_devices_last_day']}"
            for check, t in trend.items()
        )
        output.summary += (
            f" Failing devices per check, {any_check['first_day']} → "
            f"{any_check['last_day']}: {changes}."
        )
    return output


def _output(
    ctx: ToolContext,
    since: str,
    insights: list[Insight],
    coverage: dict[str, Any],
    *,
    noun: str,
    assumptions: list[str],
    checked: tuple[list[dict[str, Any]], tuple[str, ...]],
) -> ToolOutput:
    by_severity: dict[str, int] = {}
    for insight in insights:
        by_severity[insight.severity] = by_severity.get(insight.severity, 0) + 1
    breakdown = ", ".join(f"{count} {sev}" for sev, count in by_severity.items())
    summary = (
        f"Found {len(insights)} {noun} finding(s)"
        + (f" ({breakdown})" if breakdown else "")
        + f" between {since} and {ctx.as_of}."
    )
    evidence = merge_evidence(e for insight in insights for e in insight.evidence)
    if not insights:
        # "Nothing found" is a finding too: cite the latest snapshot of every
        # device that was checked, so the answer "none" is grounded.
        rows, fields = checked
        latest = {row["device_id"]: row for row in rows}  # rows are oldest-first
        evidence = [evidence_from_row(row, fields) for _, row in sorted(latest.items())]
        summary += f" Checked {len(latest)} device(s); their latest snapshots are cited."
    return ToolOutput(
        summary=summary,
        items=[insight.to_item() for insight in insights],
        evidence=evidence,
        assumptions=assumptions,
        coverage={**coverage, "window_start": since, "window_end": ctx.as_of},
    )
