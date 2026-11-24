"""Compliance drift: checks that changed state over the window.

Per device and check, the status series is classified as:
- new_failure        passed at the start, failing now      -> reported (drift)
- resolved           failed at the start, passing now      -> reported (low)
- persistent_failure failing in every snapshot             -> counted, not drift
- stable_pass / other                                      -> counted only
The fleet view compares how many devices failed each check on the first and
last day of the window.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fleet_copilot.config.settings import InsightSettings
from fleet_copilot.evidence import evidence_from_row, merge_evidence
from fleet_copilot.insights.models import Insight, sort_insights

INSIGHT_TYPE = "compliance_drift"
_FIELDS = ("check_id", "severity", "status")
_SEVERITY_MAP = {"critical": "high", "high": "high", "medium": "medium", "low": "low"}


def detect_compliance_drift(
    rows: list[dict[str, Any]], settings: InsightSettings
) -> tuple[list[Insight], dict[str, Any]]:
    series_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        series_by_key[(row["check_id"], row["device_id"])].append(row)

    insights: list[Insight] = []
    classes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (check_id, device_id), series in sorted(series_by_key.items()):
        series.sort(key=lambda r: r["collected_at"])
        if len(series) < settings.min_snapshots:
            classes[check_id]["insufficient_data"] += 1
            continue
        label = _classify(series)
        classes[check_id][label] += 1
        if label in ("new_failure", "resolved"):
            insights.append(_build_insight(check_id, device_id, series, label))

    return sort_insights(insights), {
        "status_by_check": {check: dict(counts) for check, counts in classes.items()},
        "fleet_trend_by_check": _fleet_trend(rows),
    }


def _classify(series: list[dict[str, Any]]) -> str:
    statuses = [r["status"] for r in series]
    if all(s == "fail" for s in statuses):
        return "persistent_failure"
    if all(s == "pass" for s in statuses):
        return "stable_pass"
    if statuses[0] == "pass" and statuses[-1] == "fail":
        return "new_failure"
    if statuses[0] == "fail" and statuses[-1] == "pass":
        return "resolved"
    return "fluctuating"


def _build_insight(
    check_id: str, device_id: str, series: list[dict[str, Any]], label: str
) -> Insight:
    latest = series[-1]
    target = "fail" if label == "new_failure" else "pass"
    # The change point: last snapshot before the final run of `target`, and the first in it.
    start_of_run = len(series) - 1
    while start_of_run > 0 and series[start_of_run - 1]["status"] == target:
        start_of_run -= 1
    changed_at = series[start_of_run]
    before = series[start_of_run - 1] if start_of_run > 0 else None
    transitions = sum(
        1 for a, b in zip(series, series[1:], strict=False) if a["status"] != b["status"]
    )
    run_length = len(series) - start_of_run

    if label == "new_failure":
        severity = _SEVERITY_MAP.get(latest["severity"], "medium")
        finding = (
            f"{device_id} started failing '{check_id}' ({latest['severity']} severity) on "
            f"{changed_at['collected_date']} and has failed in all {run_length} snapshots since."
        )
        explanation = (
            "The device was compliant at the start of the window and has drifted out of "
            "compliance; it has not recovered."
        )
    else:
        severity = "low"
        finding = (
            f"{device_id} has passed '{check_id}' since {changed_at['collected_date']} "
            f"after failing at the start of the window."
        )
        explanation = "The device recovered; no action is needed unless it fails again."

    return Insight(
        insight_type=INSIGHT_TYPE,
        device_id=device_id,
        severity=severity,
        finding=finding,
        explanation=explanation,
        metrics={
            "check_id": check_id,
            "check_severity": latest["severity"],
            "change": label,
            "changed_at": changed_at["collected_at"],
            "snapshots_in_current_state": run_length,
            "transitions": transitions,
            "snapshots": len(series),
        },
        evidence=merge_evidence(
            evidence_from_row(r, _FIELDS) for r in (before, changed_at, latest) if r is not None
        ),
    )


def _fleet_trend(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Devices failing each check on the first and last day of the window."""
    by_check_day: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    days: set[str] = set()
    for row in rows:
        days.add(row["collected_date"])
        if row["status"] == "fail":
            by_check_day[row["check_id"]][row["collected_date"]].add(row["device_id"])
        else:
            by_check_day[row["check_id"]].setdefault(row["collected_date"], set())
    if not days:
        return {}
    first_day, last_day = min(days), max(days)
    return {
        check: {
            "first_day": first_day,
            "failing_devices_first_day": len(per_day.get(first_day, set())),
            "last_day": last_day,
            "failing_devices_last_day": len(per_day.get(last_day, set())),
        }
        for check, per_day in sorted(by_check_day.items())
    }
