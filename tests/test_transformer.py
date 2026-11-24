import pytest

from fleet_copilot.exceptions import IngestionError
from fleet_copilot.ingestion.raw_models import TelemetryRecord
from fleet_copilot.ingestion.transformer import (
    build_dimensions,
    classify_device_type,
    company_display_name,
    to_snapshot_rows,
)

DESKTOPS = ["Mac mini"]


def _record(make_record, **overrides) -> TelemetryRecord:
    return TelemetryRecord.model_validate(make_record(**overrides))


def test_snapshot_rows_are_normalised(make_record):
    rows = to_snapshot_rows(_record(make_record))
    snap = rows.snapshot
    assert snap.snapshot_id == "DEV001@20260514T0954Z"
    assert snap.collected_at == "2026-05-14T09:54:00Z"
    assert snap.collected_date == "2026-05-14"
    assert (snap.os_version_major, snap.os_version_minor) == (14, 5)
    assert snap.memory_used_pct == 60.0
    assert snap.battery_reported == 1
    assert rows.disk_volumes[0].free_pct == 10.0
    assert rows.disk_volumes[0].is_system_volume == 1
    assert {c.check_id for c in rows.compliance_results} == {"disk_encryption", "os_up_to_date"}
    assert all(r.snapshot_id == snap.snapshot_id for r in rows.installed_software)


def test_missing_battery_is_recorded_as_not_reported(make_record):
    raw = make_record()
    raw.pop("battery")
    snap = to_snapshot_rows(TelemetryRecord.model_validate(raw)).snapshot
    assert snap.battery_reported == 0
    assert snap.battery_cycle_count is None


def test_windows_version_keeps_release(make_record):
    os_info = {"platform": "win32", "product_name": "Windows", "product_version": "11 23H2"}
    snap = to_snapshot_rows(_record(make_record, os=os_info)).snapshot
    assert (snap.os_version_major, snap.os_release) == (11, "23H2")


def test_unknown_version_does_not_fail(make_record):
    os_info = {"platform": "darwin", "product_name": "macOS", "product_version": "beta"}
    snap = to_snapshot_rows(_record(make_record, os=os_info)).snapshot
    assert snap.os_version_major is None
    assert snap.os_version == "beta"


def test_dimensions_use_latest_snapshot(make_record):
    older = _record(make_record, collected_at="2026-05-14T09:00:00Z")
    newer = _record(make_record, collected_at="2026-05-20T09:00:00Z", employee_id="emp-acme-2")
    dims = build_dimensions([newer, older], DESKTOPS)

    (device,) = dims.devices
    assert device.employee_id == "emp-acme-2"
    assert device.first_seen_at == "2026-05-14T09:00:00Z"
    assert device.last_seen_at == "2026-05-20T09:00:00Z"
    assert device.snapshot_count == 2
    assert [c.company_name for c in dims.companies] == ["Acme"]
    assert {e.employee_id for e in dims.employees} == {"emp-acme-1", "emp-acme-2"}


def test_device_owned_by_two_companies_stops_ingestion(make_record):
    first = _record(make_record)
    second = _record(make_record, company_id="globex-002", collected_at="2026-05-15T09:00:00Z")
    with pytest.raises(IngestionError, match="Tenant integrity"):
        build_dimensions([first, second], DESKTOPS)


def test_employee_owned_by_two_companies_stops_ingestion(make_record):
    first = _record(make_record)
    second = _record(make_record, device_id="DEV002", company_id="globex-002")
    with pytest.raises(IngestionError, match="employee_id"):
        build_dimensions([first, second], DESKTOPS)


@pytest.mark.parametrize(
    ("model", "expected"),
    [("Mac mini", "desktop"), ("mac MINI", "desktop"), ("MacBook Air", "laptop")],
)
def test_classify_device_type(model, expected):
    assert classify_device_type(model, DESKTOPS) == expected


def test_company_display_name():
    assert company_display_name("initech-003") == "Initech"
