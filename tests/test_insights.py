"""Detector logic on small synthetic series."""

from datetime import UTC, datetime, timedelta

import pytest

from fleet_copilot.config.settings import InsightSettings, ThresholdSettings
from fleet_copilot.insights.battery_detector import detect_battery_end_of_life
from fleet_copilot.insights.compliance_detector import detect_compliance_drift
from fleet_copilot.insights.memory_detector import detect_memory_pressure
from fleet_copilot.insights.storage_detector import detect_storage_pressure
from fleet_copilot.utils.time_utils import to_compact_utc, to_iso_utc

THRESHOLDS = ThresholdSettings()
SETTINGS = InsightSettings(min_snapshots=3)
START = datetime(2026, 5, 1, 9, tzinfo=UTC)


def series(device_id: str, values: list[dict], **constant) -> list[dict]:
    rows = []
    for day, value in enumerate(values):
        moment = START + timedelta(days=day)
        rows.append(
            {
                "device_id": device_id,
                "snapshot_id": f"{device_id}@{to_compact_utc(moment)}",
                "collected_at": to_iso_utc(moment),
                "collected_date": moment.date().isoformat(),
                **constant,
                **value,
            }
        )
    return rows


def battery(device_id, cycles, capacity, condition="Normal", reported=1):
    return series(
        device_id,
        [
            {
                "battery_cycle_count": c,
                "battery_full_charge_capacity": cap,
                "battery_condition": condition,
                "battery_reported": reported,
            }
            for c, cap in zip(cycles, capacity, strict=True)
        ],
    )


# -------------------------------------------------------------------- battery
def test_battery_over_cycle_limit_is_end_of_life():
    rows = battery("A", [998, 1000, 1002], [4000, 3990, 3980])
    (insight,), coverage = detect_battery_end_of_life(rows, THRESHOLDS, SETTINGS)
    assert insight.severity == "high"
    assert insight.metrics["status"] == "end_of_life"
    assert coverage["devices_checked"] == 1


def test_service_condition_is_end_of_life_even_below_limit():
    rows = battery("A", [500, 501, 502], [4000, 4000, 4000], condition="Service Recommended")
    (insight,), _ = detect_battery_end_of_life(rows, THRESHOLDS, SETTINGS)
    assert insight.metrics["status"] == "end_of_life"
    assert "Service Recommended" in insight.finding


def test_capacity_decline_is_approaching_end_of_life():
    rows = battery("A", [300, 301, 302], [4000, 3900, 3800])  # 5% drop
    (insight,), _ = detect_battery_end_of_life(rows, THRESHOLDS, SETTINGS)
    assert (insight.severity, insight.metrics["status"]) == ("medium", "approaching_end_of_life")
    assert insight.metrics["capacity_drop_pct"] == 5.0
    assert insight.metrics["capacity_change_mah_per_day"] == -100.0


def test_healthy_battery_is_not_reported():
    rows = battery("A", [100, 101, 102], [5000, 5000, 4999])
    assert detect_battery_end_of_life(rows, THRESHOLDS, SETTINGS)[0] == []


def test_missing_battery_readings_are_skipped_not_zeroed():
    rows = battery("A", [1200, 1201, 1202], [4000, 4000, 4000])
    rows[1].update(battery_reported=0, battery_cycle_count=None, battery_full_charge_capacity=None)
    insights, coverage = detect_battery_end_of_life(rows, THRESHOLDS, SETTINGS)
    assert coverage["devices_skipped_insufficient_battery_data"] == ["A"]  # 2 readings < 3
    assert insights == []


# --------------------------------------------------------------------- memory
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([90, 91, 92, 93], "consistent"),
        ([90, 50, 92, 50], "intermittent"),  # 50% of snapshots high
        ([90, 50, 50, 50], None),  # one spike: not reported
    ],
)
def test_memory_pattern(values, expected):
    rows = series("M", [{"memory_used_pct": v} for v in values], memory_total_bytes=8 * 1024**3)
    insights, _ = detect_memory_pressure(rows, THRESHOLDS, SETTINGS)
    if expected is None:
        assert insights == []
    else:
        assert insights[0].metrics["pattern"] == expected
        assert insights[0].metrics["ram_gb"] == 8.0


# -------------------------------------------------------------------- storage
def test_storage_projection_until_critical():
    rows = series("S", [{"disk_free_pct": v, "disk_available_bytes": 1} for v in (14, 12, 10)])
    (insight,), _ = detect_storage_pressure(rows, THRESHOLDS, SETTINGS)
    assert insight.metrics["free_pct_change_per_day"] == -2.0
    assert insight.metrics["days_until_critical"] == 2.5  # (10 - 5) / 2
    assert insight.severity == "high"  # critical within a week


def test_storage_already_critical():
    rows = series("S", [{"disk_free_pct": v, "disk_available_bytes": 1} for v in (6, 5, 4)])
    (insight,), _ = detect_storage_pressure(rows, THRESHOLDS, SETTINGS)
    assert insight.metrics["days_until_critical"] == 0.0


def test_healthy_storage_is_not_reported():
    rows = series("S", [{"disk_free_pct": v, "disk_available_bytes": 1} for v in (40, 39, 41)])
    assert detect_storage_pressure(rows, THRESHOLDS, SETTINGS)[0] == []


# ----------------------------------------------------------------- compliance
def compliance(device_id, statuses, check="screen_lock", severity="medium"):
    return series(
        device_id,
        [{"check_id": check, "severity": severity, "status": s} for s in statuses],
    )


def test_compliance_classification_and_fleet_trend():
    rows = (
        compliance("NEW", ["pass", "pass", "fail", "fail"])
        + compliance("FIXED", ["fail", "pass", "pass", "pass"])
        + compliance("ALWAYS", ["fail", "fail", "fail", "fail"])
        + compliance("OK", ["pass", "pass", "pass", "pass"])
    )
    insights, coverage = detect_compliance_drift(rows, SETTINGS)
    by_device = {i.device_id: i for i in insights}

    assert set(by_device) == {"NEW", "FIXED"}
    new = by_device["NEW"]
    assert new.metrics["change"] == "new_failure"
    assert new.metrics["changed_at"] == "2026-05-03T09:00:00Z"
    assert [e.values["status"] for e in new.evidence] == ["pass", "fail", "fail"]
    assert by_device["FIXED"].severity == "low"

    counts = coverage["status_by_check"]["screen_lock"]
    assert counts == {"new_failure": 1, "resolved": 1, "persistent_failure": 1, "stable_pass": 1}
    trend = coverage["fleet_trend_by_check"]["screen_lock"]
    assert (trend["failing_devices_first_day"], trend["failing_devices_last_day"]) == (2, 2)


def test_high_severity_drift_is_high():
    rows = compliance("X", ["pass", "fail", "fail"], check="disk_encryption", severity="high")
    (insight,), _ = detect_compliance_drift(rows, SETTINGS)
    assert insight.severity == "high"
