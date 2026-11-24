"""Typed row models, one per table in schema.sql.

Field names match column names exactly; the writer builds INSERT statements
from them, so the two stay in sync by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class CompanyRow:
    table: ClassVar[str] = "companies"
    company_id: str
    company_name: str


@dataclass(frozen=True)
class EmployeeRow:
    table: ClassVar[str] = "employees"
    employee_id: str
    company_id: str


@dataclass(frozen=True)
class DeviceRow:
    table: ClassVar[str] = "devices"
    device_id: str
    company_id: str
    employee_id: str | None
    device_type: str
    serial_number: str | None
    model_name: str
    model_identifier: str | None
    processor: str | None
    memory_total_bytes: int
    os_platform: str
    os_name: str
    first_seen_at: str
    last_seen_at: str
    snapshot_count: int


@dataclass(frozen=True)
class SnapshotRow:
    table: ClassVar[str] = "snapshots"
    snapshot_id: str
    device_id: str
    company_id: str
    employee_id: str | None
    collected_at: str
    collected_date: str
    agent_version: str | None
    os_platform: str
    os_name: str
    os_version: str
    os_version_major: int | None
    os_version_minor: int | None
    os_version_patch: int | None
    os_release: str | None
    os_build: str | None
    architecture: str | None
    hostname: str | None
    memory_total_bytes: int
    memory_used_bytes: int
    memory_free_bytes: int
    memory_used_pct: float
    battery_reported: int
    battery_percentage: int | None
    battery_charging_status: str | None
    battery_condition: str | None
    battery_cycle_count: int | None
    battery_full_charge_capacity: int | None


@dataclass(frozen=True)
class DiskVolumeRow:
    table: ClassVar[str] = "disk_volumes"
    snapshot_id: str
    volume_name: str
    mount_point: str
    file_system: str | None
    size_bytes: int
    available_bytes: int
    free_pct: float
    encrypted: int | None
    is_system_volume: int


@dataclass(frozen=True)
class ComplianceResultRow:
    table: ClassVar[str] = "compliance_results"
    snapshot_id: str
    check_id: str
    severity: str
    status: str


@dataclass(frozen=True)
class InstalledSoftwareRow:
    table: ClassVar[str] = "installed_software"
    snapshot_id: str
    name: str
    version: str | None
    publisher: str | None
