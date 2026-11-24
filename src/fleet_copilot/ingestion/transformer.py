"""Turn validated telemetry records into database rows.

Also enforces tenant integrity: a device or employee that appears under more
than one company stops the load, because tenant isolation later depends on
every row having exactly one owner.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from fleet_copilot.db.rows import (
    CompanyRow,
    ComplianceResultRow,
    DeviceRow,
    DiskVolumeRow,
    EmployeeRow,
    InstalledSoftwareRow,
    SnapshotRow,
)
from fleet_copilot.exceptions import IngestionError
from fleet_copilot.ingestion.raw_models import TelemetryRecord
from fleet_copilot.utils.time_utils import to_compact_utc, to_iso_utc
from fleet_copilot.utils.unit_utils import percent
from fleet_copilot.utils.version_utils import parse_os_version

logger = logging.getLogger(__name__)

SYSTEM_MOUNT_POINTS = frozenset({"/", "C:\\"})


@dataclass(frozen=True)
class SnapshotRows:
    snapshot: SnapshotRow
    disk_volumes: list[DiskVolumeRow]
    compliance_results: list[ComplianceResultRow]
    installed_software: list[InstalledSoftwareRow]


@dataclass(frozen=True)
class DimensionRows:
    companies: list[CompanyRow]
    employees: list[EmployeeRow]
    devices: list[DeviceRow]


def build_snapshot_id(record: TelemetryRecord) -> str:
    """Readable, deterministic evidence id: `<device_id>@<YYYYMMDDTHHMMZ>`."""
    return f"{record.device_id}@{to_compact_utc(record.collected_at)}"


def company_display_name(company_id: str) -> str:
    """Derive a display name from the id ("acme-001" -> "Acme"); the dataset has no names."""
    return company_id.split("-", 1)[0].replace("_", " ").title()


def classify_device_type(model_name: str, desktop_models: Iterable[str]) -> str:
    desktops = {name.casefold() for name in desktop_models}
    return "desktop" if model_name.casefold() in desktops else "laptop"


def to_snapshot_rows(record: TelemetryRecord) -> SnapshotRows:
    snapshot_id = build_snapshot_id(record)
    version = parse_os_version(record.os.product_version)
    if not version.is_parsed:
        logger.warning(
            "Unrecognised OS version %r on %s; version fields left empty.",
            record.os.product_version,
            snapshot_id,
        )
    battery = record.battery
    memory = record.memory

    snapshot = SnapshotRow(
        snapshot_id=snapshot_id,
        device_id=record.device_id,
        company_id=record.company_id,
        employee_id=record.employee_id,
        collected_at=to_iso_utc(record.collected_at),
        collected_date=record.collected_at.date().isoformat(),
        agent_version=record.agent_version,
        os_platform=record.os.platform,
        os_name=record.os.product_name,
        os_version=record.os.product_version,
        os_version_major=version.major,
        os_version_minor=version.minor,
        os_version_patch=version.patch,
        os_release=version.release,
        os_build=record.os.build_version,
        architecture=record.os.architecture,
        hostname=record.os.hostname,
        memory_total_bytes=memory.total_memory_bytes,
        memory_used_bytes=memory.used_memory_bytes,
        memory_free_bytes=memory.free_memory_bytes,
        memory_used_pct=percent(memory.used_memory_bytes, memory.total_memory_bytes) or 0.0,
        battery_reported=int(battery is not None),
        battery_percentage=battery.percentage if battery else None,
        battery_charging_status=battery.charging_status if battery else None,
        battery_condition=battery.condition if battery else None,
        battery_cycle_count=battery.cycle_count if battery else None,
        battery_full_charge_capacity=battery.full_charge_capacity if battery else None,
    )
    disks = [
        DiskVolumeRow(
            snapshot_id=snapshot_id,
            volume_name=volume.volume_name,
            mount_point=volume.mount_point,
            file_system=volume.file_system,
            size_bytes=volume.size_bytes,
            available_bytes=volume.available_bytes,
            free_pct=percent(volume.available_bytes, volume.size_bytes) or 0.0,
            encrypted=None if volume.encrypted is None else int(volume.encrypted),
            is_system_volume=int(volume.mount_point in SYSTEM_MOUNT_POINTS),
        )
        for volume in record.disk_volumes
    ]
    compliance = [
        ComplianceResultRow(
            snapshot_id=snapshot_id,
            check_id=check.check_id,
            severity=check.severity,
            status=check.status,
        )
        for check in record.compliance_results
    ]
    software = [
        InstalledSoftwareRow(
            snapshot_id=snapshot_id,
            name=item.name,
            version=item.version,
            publisher=item.publisher,
        )
        for item in record.installed_software
    ]
    return SnapshotRows(snapshot, disks, compliance, software)


def build_dimensions(
    records: Sequence[TelemetryRecord], desktop_models: Iterable[str]
) -> DimensionRows:
    """Build company, employee and device rows; device attributes come from the latest snapshot."""
    _check_single_owner(records, key="device_id")
    _check_single_owner(records, key="employee_id")

    by_device: dict[str, list[TelemetryRecord]] = defaultdict(list)
    for record in records:
        by_device[record.device_id].append(record)

    devices: list[DeviceRow] = []
    for device_id, history in sorted(by_device.items()):
        history.sort(key=lambda r: r.collected_at)
        latest = history[-1]
        devices.append(
            DeviceRow(
                device_id=device_id,
                company_id=latest.company_id,
                employee_id=latest.employee_id,
                device_type=classify_device_type(latest.device_identity.model_name, desktop_models),
                serial_number=latest.device_identity.serial_number,
                model_name=latest.device_identity.model_name,
                model_identifier=latest.device_identity.model_identifier,
                processor=latest.device_identity.processor,
                memory_total_bytes=latest.memory.total_memory_bytes,
                os_platform=latest.os.platform,
                os_name=latest.os.product_name,
                first_seen_at=to_iso_utc(history[0].collected_at),
                last_seen_at=to_iso_utc(latest.collected_at),
                snapshot_count=len(history),
            )
        )

    company_ids = sorted({r.company_id for r in records})
    employees = sorted(
        {(r.employee_id, r.company_id) for r in records if r.employee_id is not None}
    )
    return DimensionRows(
        companies=[CompanyRow(cid, company_display_name(cid)) for cid in company_ids],
        employees=[EmployeeRow(eid, cid) for eid, cid in employees],
        devices=devices,
    )


def _check_single_owner(records: Sequence[TelemetryRecord], *, key: str) -> None:
    owners: dict[str, set[str]] = defaultdict(set)
    for record in records:
        value = getattr(record, key)
        if value is not None:
            owners[value].add(record.company_id)
    conflicts = {
        value: sorted(companies) for value, companies in owners.items() if len(companies) > 1
    }
    if conflicts:
        raise IngestionError(
            f"Tenant integrity violation: {key} owned by several companies: {conflicts}"
        )
