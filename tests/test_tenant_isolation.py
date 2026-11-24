"""Tenant isolation: no repository query or tool may return another company's data."""

from contextlib import closing

import pytest

from fleet_copilot.audit import AuditLogger
from fleet_copilot.db.connection import connect
from fleet_copilot.db.fleet_repository import FleetRepository
from fleet_copilot.exceptions import TenantNotFoundError
from fleet_copilot.guardrails.tenant_context import resolve_tenant
from fleet_copilot.tools import TOOL_REGISTRY, ToolExecutor
from tests.conftest import ACME, GLOBEX, INITECH, tenant

TENANTS = [ACME, GLOBEX, INITECH]
SINCE = "2000-01-01T00:00:00Z"

# Arguments that make each tool return as much data as possible.
TOOL_ARGS = {
    "get_fleet_overview": {},
    "list_devices": {},
    "find_devices_by_metric": {"metric": "memory_used_pct", "comparison": "gte", "value": 0},
    "find_devices_by_os_version": {"os_name": "macOS", "comparison": "gte", "version": "1"},
    "get_compliance_status": {"status": None},
    "search_installed_software": {"name": "Slack"},
    "detect_battery_end_of_life": {},
    "detect_memory_pressure": {},
    "detect_storage_pressure": {},
    "detect_compliance_drift": {},
}


def _device_owners(settings) -> dict[str, str]:
    with closing(connect(settings.paths.fleet_db, read_only=True)) as conn:
        return dict(conn.execute("SELECT device_id, company_id FROM devices").fetchall())


@pytest.mark.parametrize(("company_id", "name"), TENANTS)
def test_repository_returns_only_own_devices(fleet_settings, company_id, name):
    owners = _device_owners(fleet_settings)
    with closing(connect(fleet_settings.paths.fleet_db, read_only=True)) as conn:
        repo = FleetRepository(conn, tenant(company_id, name))
        results = {
            "latest_device_states": repo.latest_device_states(),
            "latest_compliance": repo.latest_compliance(),
            "latest_software": repo.latest_software(),
            "device_state_history": repo.device_state_history(since=SINCE),
            "compliance_history": repo.compliance_history(since=SINCE),
        }
    for method, rows in results.items():
        assert rows, f"{method} returned nothing"
        foreign = {r["device_id"] for r in rows if owners[r["device_id"]] != company_id}
        assert not foreign, f"{method} leaked devices {foreign}"


def test_repository_cannot_read_another_tenants_device(fleet_settings):
    owners = _device_owners(fleet_settings)
    globex_device = next(d for d, c in owners.items() if c == GLOBEX[0])
    with closing(connect(fleet_settings.paths.fleet_db, read_only=True)) as conn:
        repo = FleetRepository(conn, tenant(*ACME))
        assert repo.latest_device_state(globex_device) is None
        assert repo.device_state_history(since=SINCE, device_id=globex_device) == []
        assert repo.latest_compliance(device_id=globex_device) == []
        assert repo.owning_company(globex_device) == GLOBEX[0]  # audit-only lookup


@pytest.mark.parametrize("tool_name", sorted(TOOL_ARGS))
@pytest.mark.parametrize(("company_id", "name"), TENANTS)
def test_every_tool_returns_only_own_evidence(fleet_settings, tool_name, company_id, name):
    owners = _device_owners(fleet_settings)
    executor = ToolExecutor(
        fleet_settings, tenant(company_id, name), AuditLogger(fleet_settings.paths.audit_db)
    )
    result = executor.execute(tool_name, TOOL_ARGS[tool_name])

    assert result.status == "ok", result.summary
    assert result.tenant_id == company_id
    devices = {e.device_id for e in result.evidence} | {
        item["device_id"] for item in result.items if "device_id" in item
    }
    assert all(owners[d] == company_id for d in devices), f"{tool_name} leaked data"
    for evidence in result.evidence:
        assert evidence.snapshot_id.startswith(evidence.device_id + "@")


def test_all_tools_are_covered_by_isolation_test():
    assert set(TOOL_ARGS) | {"get_device_details", "get_device_history"} == set(TOOL_REGISTRY)


@pytest.mark.parametrize(
    ("tool_name", "extra"),
    [("get_device_details", {}), ("get_device_history", {"metrics": ["disk_free_pct"]})],
)
def test_cross_tenant_device_is_denied_and_audited(fleet_settings, tool_name, extra):
    owners = _device_owners(fleet_settings)
    initech_device = next(d for d, c in owners.items() if c == INITECH[0])
    audit = AuditLogger(fleet_settings.paths.audit_db)
    executor = ToolExecutor(fleet_settings, tenant(*ACME, actor="alice"), audit)

    result = executor.execute(tool_name, {"device_id": initech_device, **extra})

    assert result.status == "not_found"
    assert "Initech" not in result.summary and INITECH[0] not in result.summary
    assert result.items == [] and result.evidence == []
    event = audit.list_events(limit=1)[0]
    assert (event.status, event.action, event.actor) == ("denied", tool_name, "alice")
    assert event.tenant_id == ACME[0]
    assert event.details["reason"] == "cross_tenant_device"


def test_unknown_device_is_not_found_but_not_denied(fleet_settings):
    audit = AuditLogger(fleet_settings.paths.audit_db)
    executor = ToolExecutor(fleet_settings, tenant(*ACME), audit)
    result = executor.execute("get_device_details", {"device_id": "DOES-NOT-EXIST"})
    assert result.status == "not_found"
    assert audit.list_events(limit=1)[0].status == "succeeded"


def test_tools_have_no_tenant_argument():
    for spec in TOOL_REGISTRY.values():
        fields = set(spec.input_model.model_fields)
        assert not fields & {"company_id", "tenant", "tenant_id", "company"}, spec.name


def test_extra_tenant_argument_is_rejected(fleet_settings):
    executor = ToolExecutor(
        fleet_settings, tenant(*ACME), AuditLogger(fleet_settings.paths.audit_db)
    )
    result = executor.execute("list_devices", {"company_id": GLOBEX[0]})
    assert result.status == "error"
    assert "company_id" in result.summary


@pytest.mark.parametrize(("value", "expected"), [("acme-001", "acme-001"), ("ACME", "acme-001")])
def test_resolve_tenant_by_id_or_name(fleet_settings, value, expected):
    with closing(connect(fleet_settings.paths.fleet_db, read_only=True)) as conn:
        assert resolve_tenant(conn, value, actor="a").company_id == expected


def test_resolve_unknown_tenant_fails(fleet_settings):
    with (
        closing(connect(fleet_settings.paths.fleet_db, read_only=True)) as conn,
        pytest.raises(TenantNotFoundError),
    ):
        resolve_tenant(conn, "umbrella", actor="a")
