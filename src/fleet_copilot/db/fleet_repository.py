"""Tenant-scoped, read-only queries over the fleet database.

Tenant isolation rule: every query in this class filters on
`company_id = :company_id`, taken from the TenantContext the repository was
built with. Callers cannot pass a company id. The single exception,
`owning_company`, is marked audit-only: it lets the tool layer tell a
cross-tenant attempt from a typo, and its result is never shown to the model.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from fleet_copilot.guardrails.tenant_context import TenantContext

# One wide row per snapshot: device attributes + snapshot metrics + system disk.
_STATE_COLUMNS = """
    s.snapshot_id, s.device_id, s.employee_id, s.collected_at, s.collected_date,
    d.device_type, d.model_name, d.serial_number,
    s.os_name, s.os_version, s.os_version_major, s.os_version_minor,
    s.os_version_patch, s.os_release,
    s.memory_total_bytes, s.memory_used_bytes, s.memory_used_pct,
    v.free_pct AS disk_free_pct, v.available_bytes AS disk_available_bytes,
    v.size_bytes AS disk_size_bytes, v.encrypted AS disk_encrypted,
    s.battery_reported, s.battery_percentage, s.battery_condition,
    s.battery_cycle_count, s.battery_full_charge_capacity
"""

_STATE_FROM = """
    FROM snapshots s
    JOIN devices d ON d.device_id = s.device_id AND d.company_id = s.company_id
    LEFT JOIN disk_volumes v ON v.snapshot_id = s.snapshot_id AND v.is_system_volume = 1
"""

_IS_LATEST = """
    s.collected_at = (SELECT MAX(x.collected_at) FROM snapshots x WHERE x.device_id = s.device_id)
"""


class FleetRepository:
    def __init__(self, conn: sqlite3.Connection, tenant: TenantContext) -> None:
        self._conn = conn
        self._tenant = tenant

    @property
    def tenant(self) -> TenantContext:
        return self._tenant

    # ----------------------------------------------------------------- metadata
    def as_of(self) -> str | None:
        """Latest telemetry time in the dataset; the reference point for "now"."""
        row = self._conn.execute(
            "SELECT value FROM dataset_metadata WHERE key = 'as_of'"
        ).fetchone()
        return row["value"] if row else None

    # ------------------------------------------------------------ current state
    def latest_device_states(self, device_type: str | None = None) -> list[dict[str, Any]]:
        """Latest snapshot of every device in the tenant."""
        sql = f"""SELECT {_STATE_COLUMNS} {_STATE_FROM}
                  WHERE s.company_id = :company_id AND {_IS_LATEST}"""
        params: dict[str, Any] = self._params()
        if device_type:
            sql += " AND d.device_type = :device_type"
            params["device_type"] = device_type
        return self._fetch(sql + " ORDER BY s.device_id", params)

    def latest_device_state(self, device_id: str) -> dict[str, Any] | None:
        rows = self._fetch(
            f"""SELECT {_STATE_COLUMNS} {_STATE_FROM}
                WHERE s.company_id = :company_id AND s.device_id = :device_id AND {_IS_LATEST}""",
            self._params(device_id=device_id),
        )
        return rows[0] if rows else None

    def latest_compliance(
        self,
        check_id: str | None = None,
        severity: str | None = None,
        status: str | None = None,
        device_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = f"""
            SELECT s.snapshot_id, s.device_id, s.collected_at, d.device_type, d.model_name,
                   s.employee_id, c.check_id, c.severity, c.status
            FROM compliance_results c
            JOIN snapshots s ON s.snapshot_id = c.snapshot_id
            JOIN devices d ON d.device_id = s.device_id AND d.company_id = s.company_id
            WHERE s.company_id = :company_id AND {_IS_LATEST}
        """
        params = self._params()
        for column, value in (
            ("c.check_id", check_id),
            ("c.severity", severity),
            ("c.status", status),
            ("s.device_id", device_id),
        ):
            if value is not None:
                key = column.split(".")[1]
                sql += f" AND {column} = :{key}"
                params[key] = value
        return self._fetch(sql + " ORDER BY s.device_id, c.check_id", params)

    def latest_software(
        self, name_contains: str | None = None, device_id: str | None = None
    ) -> list[dict[str, Any]]:
        sql = f"""
            SELECT s.snapshot_id, s.device_id, s.collected_at, d.device_type, s.employee_id,
                   i.name, i.version, i.publisher
            FROM installed_software i
            JOIN snapshots s ON s.snapshot_id = i.snapshot_id
            JOIN devices d ON d.device_id = s.device_id AND d.company_id = s.company_id
            WHERE s.company_id = :company_id AND {_IS_LATEST}
        """
        params = self._params()
        if name_contains:
            sql += " AND LOWER(i.name) LIKE :pattern ESCAPE '\\'"
            params["pattern"] = f"%{_escape_like(name_contains.lower())}%"
        if device_id:
            sql += " AND s.device_id = :device_id"
            params["device_id"] = device_id
        return self._fetch(sql + " ORDER BY i.name, s.device_id", params)

    # ------------------------------------------------------------------ history
    def device_state_history(
        self, since: str, device_id: str | None = None, device_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Every snapshot since `since` (ISO UTC), oldest first, per device."""
        sql = f"""SELECT {_STATE_COLUMNS} {_STATE_FROM}
                  WHERE s.company_id = :company_id AND s.collected_at >= :since"""
        params = self._params(since=since)
        if device_id:
            sql += " AND s.device_id = :device_id"
            params["device_id"] = device_id
        if device_type:
            sql += " AND d.device_type = :device_type"
            params["device_type"] = device_type
        return self._fetch(sql + " ORDER BY s.device_id, s.collected_at", params)

    def compliance_history(self, since: str, check_id: str | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT s.snapshot_id, s.device_id, s.collected_at, s.collected_date, d.device_type,
                   s.employee_id, c.check_id, c.severity, c.status
            FROM compliance_results c
            JOIN snapshots s ON s.snapshot_id = c.snapshot_id
            JOIN devices d ON d.device_id = s.device_id AND d.company_id = s.company_id
            WHERE s.company_id = :company_id AND s.collected_at >= :since
        """
        params = self._params(since=since)
        if check_id:
            sql += " AND c.check_id = :check_id"
            params["check_id"] = check_id
        return self._fetch(sql + " ORDER BY c.check_id, s.device_id, s.collected_at", params)

    def employee_device_ids(self, employee_id: str) -> list[str]:
        """Devices assigned to an employee of this tenant (empty if not one of ours)."""
        rows = self._fetch(
            """SELECT device_id FROM devices
               WHERE company_id = :company_id AND employee_id = :employee_id
               ORDER BY device_id""",
            self._params(employee_id=employee_id),
        )
        return [row["device_id"] for row in rows]

    # ---------------------------------------------------------------- audit only
    def owning_company(self, device_id: str) -> str | None:
        """AUDIT ONLY: which company owns a device, across tenants.

        Used to record a cross-tenant attempt as `denied`. Never return this
        value to the model or the user.
        """
        row = self._conn.execute(
            "SELECT company_id FROM devices WHERE device_id = ?", (device_id,)
        ).fetchone()
        return row["company_id"] if row else None

    # ----------------------------------------------------------------- helpers
    def _params(self, **extra: Any) -> dict[str, Any]:
        return {"company_id": self._tenant.company_id, **extra}

    def _fetch(self, sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        return [dict(row) for row in self._conn.execute(sql, params)]


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
