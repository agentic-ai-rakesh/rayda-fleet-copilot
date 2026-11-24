"""Input schemas for every tool.

These are what the model sees as tool parameters. They are deliberately
narrow (enums over free text) so a call is either valid or rejected with a
clear message. No schema has a company/tenant field: tenant comes from the
session, never from the model.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DeviceType(StrEnum):
    LAPTOP = "laptop"
    DESKTOP = "desktop"


class OsName(StrEnum):
    MACOS = "macOS"
    WINDOWS = "Windows"


class Comparison(StrEnum):
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    EQ = "eq"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class DeviceMetric(StrEnum):
    DISK_FREE_PCT = "disk_free_pct"
    MEMORY_USED_PCT = "memory_used_pct"
    BATTERY_CYCLE_COUNT = "battery_cycle_count"
    BATTERY_FULL_CHARGE_CAPACITY = "battery_full_charge_capacity"


class HistoryMetric(StrEnum):
    DISK_FREE_PCT = "disk_free_pct"
    MEMORY_USED_PCT = "memory_used_pct"
    BATTERY_CYCLE_COUNT = "battery_cycle_count"
    BATTERY_FULL_CHARGE_CAPACITY = "battery_full_charge_capacity"
    BATTERY_CONDITION = "battery_condition"
    OS_VERSION = "os_version"


_DEVICE_ID = Field(min_length=1, max_length=64, description="Device id, e.g. '1LYSSFD074BB'.")
_DAYS = Field(
    default=None, ge=1, le=90, description="Look-back window in days. Default: whole dataset."
)


# ------------------------------------------------------------------ read tools
class GetFleetOverviewInput(_ToolInput):
    pass


class ListDevicesInput(_ToolInput):
    device_type: DeviceType | None = None
    os_name: OsName | None = None
    model_name: str | None = Field(
        default=None, max_length=64, description="Substring match, e.g. 'MacBook'."
    )
    employee_id: str | None = Field(default=None, max_length=64)


class GetDeviceDetailsInput(_ToolInput):
    device_id: str = _DEVICE_ID


class FindDevicesByMetricInput(_ToolInput):
    metric: DeviceMetric
    comparison: Comparison
    value: float = Field(description="Threshold, in the metric's unit (percent, cycles, mAh).")
    device_type: DeviceType | None = None


class FindDevicesByOsVersionInput(_ToolInput):
    os_name: OsName
    comparison: Comparison
    version: str = Field(
        min_length=1,
        max_length=16,
        description="Version to compare with: macOS '15' or '14.5'; Windows '11' or '11 23H2'.",
    )
    device_type: DeviceType | None = None


class GetComplianceStatusInput(_ToolInput):
    check_id: str | None = Field(default=None, max_length=64, description="e.g. 'screen_lock'.")
    severity: Severity | None = None
    status: CheckStatus | None = Field(default=CheckStatus.FAIL, description="Default: failing.")
    device_type: DeviceType | None = None


class SearchInstalledSoftwareInput(_ToolInput):
    name: str = Field(min_length=2, max_length=64, description="Substring, case-insensitive.")
    device_type: DeviceType | None = None


class GetDeviceHistoryInput(_ToolInput):
    device_id: str = _DEVICE_ID
    metrics: list[HistoryMetric] = Field(min_length=1, max_length=6)
    days: int | None = _DAYS


# --------------------------------------------------------------- insight tools
class DetectBatteryEndOfLifeInput(_ToolInput):
    days: int | None = _DAYS


class DetectMemoryPressureInput(_ToolInput):
    days: int | None = _DAYS


class DetectStoragePressureInput(_ToolInput):
    days: int | None = _DAYS


class DetectComplianceDriftInput(_ToolInput):
    check_id: str | None = Field(default=None, max_length=64)
    days: int | None = _DAYS


# ---------------------------------------------------------------- action tools
# State-changing. Never bound to the model's tool loop: the model proposes,
# the evidence gate checks, a human approves, and only then do these run.
class CreateUpgradeOrderInput(_ToolInput):
    device_id: str = _DEVICE_ID
    component: Literal["ram", "storage", "battery"]
    spec: str = Field(min_length=1, max_length=64, description="e.g. '16 GB', '1 TB SSD'.")


class OpenRemediationTicketInput(_ToolInput):
    device_id: str = _DEVICE_ID
    check_id: str = Field(min_length=1, max_length=64)
    note: str = Field(min_length=1, max_length=500)


class FlagDeviceForReplacementInput(_ToolInput):
    device_id: str = _DEVICE_ID
    reason: str = Field(min_length=1, max_length=500)


class NotifyEmployeeInput(_ToolInput):
    employee_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=2000)
