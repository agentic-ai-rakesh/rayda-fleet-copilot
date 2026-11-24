"""Tool answers on the real dataset, plus executor fallbacks and audit behaviour.

Expected values were checked independently against the raw data during
profiling; they double as early ground truth for the eval suite.
"""

import pytest

from fleet_copilot.audit import AuditLogger
from fleet_copilot.exceptions import AuditError
from fleet_copilot.tools import TOOL_REGISTRY, ToolExecutor
from fleet_copilot.tools.registry import ToolSpec
from fleet_copilot.tools.results import ToolOutput
from tests.conftest import ACME, GLOBEX, INITECH, tenant


@pytest.fixture
def executor_for(fleet_settings):
    def build(company=ACME, settings=None) -> ToolExecutor:
        chosen = settings or fleet_settings
        return ToolExecutor(chosen, tenant(*company), AuditLogger(chosen.paths.audit_db))

    return build


def _ids(result) -> set[str]:
    return {item["device_id"] for item in result.items}


# ------------------------------------------------------------ grounded answers
def test_low_disk_at_acme(executor_for):
    result = executor_for(ACME).execute(
        "find_devices_by_metric", {"metric": "disk_free_pct", "comparison": "lt", "value": 15}
    )
    assert _ids(result) == {"MT7PJB7N5LRE", "8NM23J95R5I6"}
    assert result.coverage["devices_checked"] == 10
    assert all(e.values["disk_free_pct"] < 15 for e in result.evidence)


def test_no_laptop_fails_a_high_severity_check(executor_for):
    for company in (ACME, GLOBEX, INITECH):
        result = executor_for(company).execute(
            "get_compliance_status", {"severity": "high", "device_type": "laptop"}
        )
        assert result.items == []
        assert result.summary.startswith("0 of")
        # The empty answer is grounded by the passing snapshots that were checked.
        assert result.evidence and all(e.values["status"] == "pass" for e in result.evidence)


@pytest.mark.parametrize(("company", "expected"), [(ACME, 4), (GLOBEX, 2), (INITECH, 4)])
def test_macos_older_than_15(executor_for, company, expected):
    result = executor_for(company).execute(
        "find_devices_by_os_version", {"os_name": "macOS", "comparison": "lt", "version": "15"}
    )
    assert len(result.items) == expected
    assert all(item["os_version"].split(".")[0] in {"13", "14"} for item in result.items)


def test_os_version_compares_at_requested_precision(executor_for):
    run = executor_for(ACME).execute
    below_14_5 = run(
        "find_devices_by_os_version", {"os_name": "macOS", "comparison": "lt", "version": "14.5"}
    )
    at_14 = run(
        "find_devices_by_os_version", {"os_name": "macOS", "comparison": "eq", "version": "14"}
    )
    assert {i["os_version"] for i in below_14_5.items} == {"13.6"}
    assert {i["os_version"] for i in at_14.items} == {"14.5"}


def test_windows_versions_compare_by_major_and_release(executor_for):
    result = executor_for(INITECH).execute(
        "find_devices_by_os_version",
        {"os_name": "Windows", "comparison": "lt", "version": "11 23H2"},
    )
    assert all(i["os_version"] in {"10 22H2", "11 22H2"} for i in result.items)


def test_unreadable_version_is_an_error(executor_for):
    result = executor_for().execute(
        "find_devices_by_os_version", {"os_name": "macOS", "comparison": "lt", "version": "Sequoia"}
    )
    assert result.status == "error"


def test_battery_end_of_life_finds_the_three_worn_batteries(executor_for):
    result = executor_for(ACME).execute("detect_battery_end_of_life", {})
    assert _ids(result) == {"7VP16KHV88LM", "8NM23J95R5I6", "M4XVHUV1MEPZ"}
    assert all(item["severity"] == "high" for item in result.items)
    for item in result.items:
        assert set(item["evidence_ids"]) <= set(result.evidence_ids)
        assert item["metrics"]["condition"] == "Service Recommended"


def test_memory_pressure_separates_consistent_from_intermittent(executor_for):
    globex = executor_for(GLOBEX).execute("detect_memory_pressure", {})
    patterns = {i["device_id"]: i["metrics"]["pattern"] for i in globex.items}
    assert patterns == {"JRZSGXVMKE6M": "consistent", "CUXW0BQSV1B7": "intermittent"}


def test_storage_pressure_projects_days_until_critical(executor_for):
    result = executor_for(INITECH).execute("detect_storage_pressure", {})
    assert _ids(result) == {"DARFCP0BQM2G", "RLD7WV7J82SH"}
    assert all(i["metrics"]["latest_free_pct"] <= 2.0 for i in result.items)
    assert all(i["metrics"]["days_until_critical"] == 0.0 for i in result.items)


def test_compliance_drift_reports_screen_lock_spread(executor_for):
    result = executor_for(ACME).execute("detect_compliance_drift", {"check_id": "screen_lock"})
    trend = result.coverage["fleet_trend_by_check"]["screen_lock"]
    assert (trend["failing_devices_first_day"], trend["failing_devices_last_day"]) == (0, 5)
    assert {i["metrics"]["change"] for i in result.items} == {"new_failure"}
    assert len(result.items) == 5


def test_device_history_respects_window(executor_for):
    result = executor_for(ACME).execute(
        "get_device_history",
        {"device_id": "1LYSSFD074BB", "metrics": ["disk_free_pct"], "days": 3},
    )
    assert len(result.items) == 3
    assert result.items[-1]["collected_at"] <= result.as_of


EVERY_TOOL_ARGS = {
    "get_fleet_overview": {},
    "list_devices": {},
    "get_device_details": {"device_id": "1LYSSFD074BB"},
    "find_devices_by_metric": {"metric": "memory_used_pct", "comparison": "gte", "value": 50},
    "find_devices_by_os_version": {"os_name": "macOS", "comparison": "gte", "version": "13"},
    "get_compliance_status": {},
    "search_installed_software": {"name": "zoom"},
    "get_device_history": {"device_id": "1LYSSFD074BB", "metrics": ["memory_used_pct"]},
    "detect_battery_end_of_life": {},
    "detect_memory_pressure": {},
    "detect_storage_pressure": {},
    "detect_compliance_drift": {},
}


def test_args_table_covers_every_tool():
    assert set(EVERY_TOOL_ARGS) == set(TOOL_REGISTRY)


@pytest.mark.parametrize("tool_name", sorted(EVERY_TOOL_ARGS))
def test_every_item_cites_evidence_present_in_the_result(executor_for, tool_name):
    result = executor_for(ACME).execute(tool_name, EVERY_TOOL_ARGS[tool_name])
    assert result.status == "ok", result.summary
    assert result.evidence, "every result must cite evidence, even an empty one"
    for item in result.items:
        assert item["evidence_ids"]
        assert set(item["evidence_ids"]) <= set(result.evidence_ids)


# --------------------------------------------------------- fallbacks & audit
def test_unknown_tool_returns_error_and_is_audited(executor_for, fleet_settings):
    result = executor_for().execute("delete_everything", {})
    assert result.status == "error"
    assert "Available tools" in result.summary
    event = AuditLogger(fleet_settings.paths.audit_db).list_events(limit=1)[0]
    assert (event.action, event.status) == ("delete_everything", "failed")


def test_invalid_arguments_never_reach_the_database(executor_for):
    result = executor_for().execute("find_devices_by_metric", {"metric": "cpu_temp"})
    assert result.status == "error"
    assert "metric" in result.error


def test_results_are_truncated_with_evidence_kept_in_step(fleet_settings, executor_for):
    small = fleet_settings.model_copy(deep=True)
    small.tools.max_rows = 2
    result = executor_for(ACME, small).execute("list_devices", {})
    assert result.truncated and len(result.items) == 2
    assert result.coverage["items_total"] == 10
    assert set(result.evidence_ids) == {i["evidence_ids"][0] for i in result.items}


def test_unexpected_handler_error_becomes_error_result(fleet_settings):
    def broken(_ctx, _args) -> ToolOutput:
        raise RuntimeError("boom")

    registry = {
        "broken": ToolSpec(
            "broken", "read", "test", TOOL_REGISTRY["list_devices"].input_model, broken
        )
    }
    audit = AuditLogger(fleet_settings.paths.audit_db)
    result = ToolExecutor(fleet_settings, tenant(*ACME), audit, registry).execute("broken", {})
    assert result.status == "error"
    assert "boom" not in result.summary  # internals stay in the log
    assert audit.list_events(limit=1)[0].status == "failed"


def test_missing_database_is_reported_not_raised(fleet_settings, tmp_path, executor_for):
    missing = fleet_settings.model_copy(deep=True)
    missing.paths.fleet_db = tmp_path / "absent.db"
    missing.paths.audit_db = tmp_path / "audit.db"
    result = executor_for(ACME, missing).execute("list_devices", {})
    assert result.status == "error"
    assert "ingest" in result.summary


def test_audit_failure_stops_the_call(fleet_settings, monkeypatch):
    audit = AuditLogger(fleet_settings.paths.audit_db)

    def fail(**_kwargs):
        raise AuditError("disk full")

    monkeypatch.setattr(audit, "record", fail)
    with pytest.raises(AuditError):
        ToolExecutor(fleet_settings, tenant(*ACME), audit).execute("list_devices", {})


def test_every_call_is_audited_with_tenant_actor_and_evidence(fleet_settings):
    audit = AuditLogger(fleet_settings.paths.audit_db)
    executor = ToolExecutor(fleet_settings, tenant(*GLOBEX, actor="bob"), audit)
    result = executor.execute("detect_memory_pressure", {})
    event = audit.list_events(limit=1)[0]
    assert (event.event_type, event.action, event.status) == (
        "tool_call",
        "detect_memory_pressure",
        "succeeded",
    )
    assert (event.actor, event.tenant_id) == ("bob", GLOBEX[0])
    assert event.details["evidence_ids"] == result.evidence_ids
    assert audit.verify_integrity() > 0
