"""Read tools: answer "what is true now" questions from each device's latest snapshot.

Each handler returns items that carry `evidence_ids`, evidence with the exact
values used, the assumptions applied, and coverage (how many devices were
checked), so an empty answer is still a grounded answer.
"""

from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Callable
from typing import Any

from fleet_copilot.evidence import Evidence, evidence_from_row
from fleet_copilot.tools.context import DeviceNotInScopeError, ToolContext, ToolInputError
from fleet_copilot.tools.results import ToolOutput
from fleet_copilot.tools.schemas import (
    Comparison,
    DeviceMetric,
    FindDevicesByMetricInput,
    FindDevicesByOsVersionInput,
    GetComplianceStatusInput,
    GetDeviceDetailsInput,
    GetDeviceHistoryInput,
    GetFleetOverviewInput,
    ListDevicesInput,
    OsName,
    SearchInstalledSoftwareInput,
)
from fleet_copilot.utils.unit_utils import bytes_to_gb
from fleet_copilot.utils.version_utils import parse_os_version

COMPARATORS: dict[Comparison, Callable[[Any, Any], bool]] = {
    Comparison.LT: operator.lt,
    Comparison.LTE: operator.le,
    Comparison.GT: operator.gt,
    Comparison.GTE: operator.ge,
    Comparison.EQ: operator.eq,
}
COMPARISON_WORDS = {
    Comparison.LT: "below",
    Comparison.LTE: "at or below",
    Comparison.GT: "above",
    Comparison.GTE: "at or above",
    Comparison.EQ: "equal to",
}
METRIC_LABELS = {
    DeviceMetric.DISK_FREE_PCT: "system disk free %",
    DeviceMetric.MEMORY_USED_PCT: "memory used %",
    DeviceMetric.BATTERY_CYCLE_COUNT: "battery cycle count",
    DeviceMetric.BATTERY_FULL_CHARGE_CAPACITY: "battery full-charge capacity (mAh)",
}
BATTERY_METRICS = {DeviceMetric.BATTERY_CYCLE_COUNT, DeviceMetric.BATTERY_FULL_CHARGE_CAPACITY}
DEVICE_FIELDS = ("device_id", "device_type", "model_name", "employee_id")


# ---------------------------------------------------------------------- tools
def get_fleet_overview(ctx: ToolContext, _: GetFleetOverviewInput) -> ToolOutput:
    states = ctx.repo.latest_device_states()
    failing = Counter(row["check_id"] for row in ctx.repo.latest_compliance(status="fail"))
    coverage = {
        "devices": len(states),
        "by_device_type": dict(Counter(s["device_type"] for s in states)),
        "by_os": dict(Counter(f"{s['os_name']} {s['os_version']}" for s in states)),
        "by_model": dict(Counter(s["model_name"] for s in states)),
        "failing_devices_by_check": dict(failing),
    }
    items = [_device_item(s, ("os_name", "os_version")) for s in states]
    evidence = [evidence_from_row(s, ("device_type", "os_name", "os_version")) for s in states]
    tenant = ctx.repo.tenant.company_name
    return ToolOutput(
        summary=f"{tenant} has {len(states)} devices: "
        f"{_format_counts(coverage['by_device_type'])}.",
        items=items,
        evidence=evidence,
        assumptions=[_latest_assumption(ctx)],
        coverage=coverage,
    )


def list_devices(ctx: ToolContext, args: ListDevicesInput) -> ToolOutput:
    states = ctx.repo.latest_device_states(device_type=_value(args.device_type))
    total = len(states)
    if args.os_name:
        states = [s for s in states if s["os_name"] == args.os_name.value]
    if args.model_name:
        states = [s for s in states if args.model_name.lower() in s["model_name"].lower()]
    if args.employee_id:
        states = [s for s in states if s["employee_id"] == args.employee_id]
    fields = ("os_name", "os_version", "serial_number")
    return ToolOutput(
        summary=f"{len(states)} of {total} devices match the filters.",
        items=[_device_item(s, fields) for s in states],
        evidence=[
            evidence_from_row(s, ("device_type", "model_name", "os_version")) for s in states
        ],
        assumptions=[_latest_assumption(ctx)],
        coverage={"devices_checked": total},
    )


def get_device_details(ctx: ToolContext, args: GetDeviceDetailsInput) -> ToolOutput:
    state = ctx.repo.latest_device_state(args.device_id)
    if state is None:
        raise DeviceNotInScopeError(args.device_id)
    compliance = ctx.repo.latest_compliance(device_id=args.device_id)
    software = ctx.repo.latest_software(device_id=args.device_id)

    detail_fields = (
        "serial_number",
        "os_name",
        "os_version",
        "memory_used_pct",
        "disk_free_pct",
        "battery_reported",
        "battery_condition",
        "battery_cycle_count",
        "battery_full_charge_capacity",
        "battery_percentage",
    )
    item = _device_item(state, detail_fields)
    item["ram_gb"] = bytes_to_gb(state["memory_total_bytes"])
    item["disk_free_gb"] = bytes_to_gb(state["disk_available_bytes"])
    item["disk_size_gb"] = bytes_to_gb(state["disk_size_bytes"])
    item["compliance"] = [
        {"check_id": c["check_id"], "severity": c["severity"], "status": c["status"]}
        for c in compliance
    ]
    item["installed_software"] = [{"name": s["name"], "version": s["version"]} for s in software]
    values = {field: state.get(field) for field in detail_fields}
    values["compliance"] = {c["check_id"]: c["status"] for c in compliance}
    evidence = [
        Evidence(
            snapshot_id=state["snapshot_id"],
            device_id=state["device_id"],
            collected_at=state["collected_at"],
            values=values,
        )
    ]
    return ToolOutput(
        summary=f"Latest snapshot of {args.device_id} ({state['model_name']}) "
        f"at {state['collected_at']}.",
        items=[item],
        evidence=evidence,
        assumptions=[_latest_assumption(ctx)],
    )


def find_devices_by_metric(ctx: ToolContext, args: FindDevicesByMetricInput) -> ToolOutput:
    metric = args.metric.value
    states = ctx.repo.latest_device_states(device_type=_value(args.device_type))
    with_data = [s for s in states if s.get(metric) is not None]
    compare = COMPARATORS[args.comparison]
    matches = [s for s in with_data if compare(s[metric], args.value)]
    descending = args.comparison in (Comparison.GT, Comparison.GTE)
    matches.sort(key=lambda s: s[metric], reverse=descending)

    label = METRIC_LABELS[args.metric]
    assumptions = [_latest_assumption(ctx)]
    if args.metric is DeviceMetric.DISK_FREE_PCT:
        assumptions.append("Disk means the system volume ('/' on macOS, 'C:\\' on Windows).")
    skipped = len(states) - len(with_data)
    if skipped and args.metric in BATTERY_METRICS:
        assumptions.append(f"{skipped} device(s) skipped: no battery data in the latest snapshot.")

    return ToolOutput(
        summary=f"{len(matches)} of {len(with_data)} devices have {label} "
        f"{COMPARISON_WORDS[args.comparison]} {args.value:g}.",
        items=[_device_item(s, (metric,)) for s in matches],
        evidence=[evidence_from_row(s, (metric,)) for s in matches],
        assumptions=assumptions,
        coverage={"devices_checked": len(with_data), "devices_without_data": skipped},
    )


def find_devices_by_os_version(ctx: ToolContext, args: FindDevicesByOsVersionInput) -> ToolOutput:
    target = _version_key(args.os_name, args.version)
    if target is None:
        raise ToolInputError(f"cannot read version '{args.version}' for {args.os_name.value}.")

    states = ctx.repo.latest_device_states(device_type=_value(args.device_type))
    on_os = [s for s in states if s["os_name"] == args.os_name.value]
    compare = COMPARATORS[args.comparison]
    matches, unparsed = [], 0
    for state in on_os:
        device_key = _device_version_key(args.os_name, state)
        if device_key is None:
            unparsed += 1
            continue
        if compare(device_key[: len(target)], target):
            matches.append(state)

    fields = ("os_name", "os_version")
    assumptions = [_latest_assumption(ctx)]
    if unparsed:
        assumptions.append(f"{unparsed} device(s) skipped: OS version could not be read.")
    other_os = len(states) - len(on_os)
    return ToolOutput(
        summary=f"{len(matches)} of {len(on_os)} {args.os_name.value} devices run a version "
        f"{COMPARISON_WORDS[args.comparison]} {args.version} "
        f"({other_os} device(s) on other operating systems not counted).",
        items=[_device_item(s, fields) for s in matches],
        evidence=[evidence_from_row(s, fields) for s in matches],
        assumptions=assumptions,
        coverage={"devices_on_os": len(on_os), "devices_on_other_os": other_os},
    )


def get_compliance_status(ctx: ToolContext, args: GetComplianceStatusInput) -> ToolOutput:
    device_type = _value(args.device_type)
    all_latest = [
        row
        for row in ctx.repo.latest_compliance()
        if device_type is None or row["device_type"] == device_type
    ]
    known_checks = sorted({row["check_id"] for row in all_latest})
    in_scope = [
        row
        for row in all_latest
        if (args.check_id is None or row["check_id"] == args.check_id)
        and (args.severity is None or row["severity"] == args.severity.value)
    ]
    matches = [row for row in in_scope if args.status is None or row["status"] == args.status.value]

    devices_checked = len({row["device_id"] for row in in_scope})
    devices_matching = len({row["device_id"] for row in matches})
    fields = ("check_id", "severity", "status")
    # An empty answer is grounded by the snapshots that were checked and passed.
    evidence_rows = matches if matches else in_scope

    description = _describe_compliance_filter(args)
    summary = f"{devices_matching} of {devices_checked} devices {description}."
    if args.check_id and args.check_id not in known_checks:
        summary = f"Unknown check '{args.check_id}'. Known checks: {', '.join(known_checks)}."

    return ToolOutput(
        summary=summary,
        items=[
            {
                **{k: row[k] for k in DEVICE_FIELDS},
                **{k: row[k] for k in fields},
                "evidence_ids": [row["snapshot_id"]],
            }
            for row in matches
        ],
        evidence=[evidence_from_row(row, fields) for row in evidence_rows],
        assumptions=[_latest_assumption(ctx)],
        coverage={
            "devices_checked": devices_checked,
            "devices_matching": devices_matching,
            "known_checks": known_checks,
        },
    )


def search_installed_software(ctx: ToolContext, args: SearchInstalledSoftwareInput) -> ToolOutput:
    device_type = _value(args.device_type)
    states = ctx.repo.latest_device_states(device_type=device_type)
    rows = [
        row
        for row in ctx.repo.latest_software(name_contains=args.name)
        if device_type is None or row["device_type"] == device_type
    ]
    fields = ("name", "version", "publisher")
    devices = len({row["device_id"] for row in rows})
    return ToolOutput(
        summary=f"'{args.name}' is installed on {devices} of {len(states)} devices.",
        items=[
            {
                "device_id": row["device_id"],
                "device_type": row["device_type"],
                "employee_id": row["employee_id"],
                **{k: row[k] for k in fields},
                "evidence_ids": [row["snapshot_id"]],
            }
            for row in rows
        ],
        evidence=[evidence_from_row(row, fields) for row in rows],
        assumptions=[_latest_assumption(ctx)],
        coverage={"devices_checked": len(states), "devices_matching": devices},
    )


def get_device_history(ctx: ToolContext, args: GetDeviceHistoryInput) -> ToolOutput:
    if ctx.repo.latest_device_state(args.device_id) is None:
        raise DeviceNotInScopeError(args.device_id)
    since = ctx.window_start(args.days)
    rows = ctx.repo.device_state_history(since=since, device_id=args.device_id)
    metrics = [m.value for m in args.metrics]

    items = [
        {
            "collected_at": r["collected_at"],
            **{m: r.get(m) for m in metrics},
            "evidence_ids": [r["snapshot_id"]],
        }
        for r in rows
    ]
    changes = []
    for metric in metrics:
        series = [r[metric] for r in rows if r.get(metric) is not None]
        if series:
            changes.append(f"{metric} {series[0]} → {series[-1]}")
        else:
            changes.append(f"{metric}: no data")
    return ToolOutput(
        summary=f"{len(rows)} snapshots of {args.device_id} since {since}: " + "; ".join(changes),
        items=items,
        evidence=[evidence_from_row(r, metrics) for r in rows],
        assumptions=[f"Window ends at the dataset's as-of time {ctx.as_of}."],
        coverage={"snapshots": len(rows), "window_start": since},
    )


# -------------------------------------------------------------------- helpers
def _device_item(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    item = {key: row.get(key) for key in DEVICE_FIELDS}
    item.update({key: row.get(key) for key in fields})
    item["evidence_ids"] = [row["snapshot_id"]]
    return item


def _latest_assumption(ctx: ToolContext) -> str:
    return f"Values come from each device's latest snapshot (dataset as of {ctx.as_of})."


def _value(enum_value: Any) -> Any:
    return None if enum_value is None else enum_value.value


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))


def _describe_compliance_filter(args: GetComplianceStatusInput) -> str:
    status = args.status.value if args.status else "have a result for"
    verb = {"fail": "fail", "pass": "pass"}.get(status, status)
    parts = [verb]
    if args.severity:
        parts.append(f"{args.severity.value}-severity")
    parts.append(f"check '{args.check_id}'" if args.check_id else "checks")
    if args.device_type:
        parts.append(f"({args.device_type.value}s only)")
    return " ".join(parts)


def _version_key(os_name: OsName, version: str) -> tuple[Any, ...] | None:
    """Comparable key for a requested version, only as precise as the request."""
    parsed = parse_os_version(version)
    if not parsed.is_parsed:
        return None
    if os_name is OsName.WINDOWS:
        return (parsed.major,) if parsed.release is None else (parsed.major, parsed.release)
    depth = len(version.strip().split("."))
    return (parsed.major, parsed.minor, parsed.patch)[:depth]


def _device_version_key(os_name: OsName, state: dict[str, Any]) -> tuple[Any, ...] | None:
    if state["os_version_major"] is None:
        return None
    if os_name is OsName.WINDOWS:
        return (state["os_version_major"], state["os_release"] or "")
    return (
        state["os_version_major"],
        state["os_version_minor"] or 0,
        state["os_version_patch"] or 0,
    )
