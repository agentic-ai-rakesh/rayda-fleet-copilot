"""Validation models for one raw telemetry record (one NDJSON line).

Required: identity (device, company, timestamp), OS, hardware, memory and at
least one disk volume. Optional: battery (desktops have none, and laptops
sometimes skip it), installed software and compliance results.
Fields the copilot does not use (e.g. network) are ignored.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fleet_copilot.utils.time_utils import ensure_utc


class _RawModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class RawOsInfo(_RawModel):
    platform: str = Field(min_length=1)
    product_name: str = Field(min_length=1)
    product_version: str = Field(min_length=1)
    build_version: str | None = None
    architecture: str | None = None
    hostname: str | None = None


class RawDeviceIdentity(_RawModel):
    model_name: str = Field(min_length=1)
    serial_number: str | None = None
    model_identifier: str | None = None
    processor: str | None = None


class RawMemory(_RawModel):
    total_memory_bytes: int = Field(gt=0)
    used_memory_bytes: int = Field(ge=0)
    free_memory_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _used_within_total(self) -> RawMemory:
        if self.used_memory_bytes > self.total_memory_bytes:
            raise ValueError("used_memory_bytes exceeds total_memory_bytes")
        return self


class RawDiskVolume(_RawModel):
    volume_name: str = Field(min_length=1)
    mount_point: str = Field(min_length=1)
    file_system: str | None = None
    size_bytes: int = Field(gt=0)
    available_bytes: int = Field(ge=0)
    encrypted: bool | None = None

    @model_validator(mode="after")
    def _available_within_size(self) -> RawDiskVolume:
        if self.available_bytes > self.size_bytes:
            raise ValueError("available_bytes exceeds size_bytes")
        return self


class RawBattery(_RawModel):
    battery_present: bool
    percentage: int | None = Field(default=None, ge=0, le=100)
    charging_status: str | None = None
    condition: str | None = None
    cycle_count: int | None = Field(default=None, ge=0)
    full_charge_capacity: int | None = Field(default=None, ge=0)


class RawComplianceResult(_RawModel):
    check_id: str = Field(min_length=1)
    status: Literal["pass", "fail"]
    severity: Literal["low", "medium", "high", "critical"]


class RawSoftware(_RawModel):
    name: str = Field(min_length=1)
    version: str | None = None
    publisher: str | None = None


class TelemetryRecord(_RawModel):
    device_id: str = Field(min_length=1)
    company_id: str = Field(min_length=1)
    employee_id: str | None = None
    collected_at: datetime
    agent_version: str | None = None
    os: RawOsInfo
    device_identity: RawDeviceIdentity
    memory: RawMemory
    disk_volumes: list[RawDiskVolume] = Field(min_length=1)
    battery: RawBattery | None = None
    installed_software: list[RawSoftware] = Field(default_factory=list)
    compliance_results: list[RawComplianceResult] = Field(default_factory=list)

    @field_validator("collected_at")
    @classmethod
    def _to_utc(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _unique_children(self) -> TelemetryRecord:
        for label, keys in (
            ("disk mount_point", [v.mount_point for v in self.disk_volumes]),
            ("compliance check_id", [c.check_id for c in self.compliance_results]),
            ("software name", [s.name for s in self.installed_software]),
        ):
            if len(keys) != len(set(keys)):
                raise ValueError(f"duplicate {label} in one snapshot")
        return self
