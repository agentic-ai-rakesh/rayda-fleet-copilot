"""The tool catalog: name, description for the model, input schema and handler.

Descriptions are written for the model: when to use the tool, what it
returns, and what it does not cover. The README tool catalog mirrors this file.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from fleet_copilot.tools import insight_tools, read_tools, schemas
from fleet_copilot.tools.context import ToolContext
from fleet_copilot.tools.results import ToolOutput

ToolCategory = Literal["read", "insight"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    category: ToolCategory
    description: str
    input_model: type[BaseModel]
    handler: Callable[[ToolContext, BaseModel], ToolOutput]

    def json_schema(self) -> dict:
        return self.input_model.model_json_schema()


_SPECS = [
    ToolSpec(
        name="get_fleet_overview",
        category="read",
        description=(
            "Counts of the organisation's devices by type, OS version and model, plus how many "
            "devices currently fail each compliance check. Use first for broad questions."
        ),
        input_model=schemas.GetFleetOverviewInput,
        handler=read_tools.get_fleet_overview,
    ),
    ToolSpec(
        name="list_devices",
        category="read",
        description=(
            "List devices, optionally filtered by type (laptop/desktop), OS, model substring or "
            "employee. Returns ids, owners, model and OS."
        ),
        input_model=schemas.ListDevicesInput,
        handler=read_tools.list_devices,
    ),
    ToolSpec(
        name="get_device_details",
        category="read",
        description=(
            "Full latest state of one device: hardware, OS, memory, system disk, battery, "
            "compliance results and installed software."
        ),
        input_model=schemas.GetDeviceDetailsInput,
        handler=read_tools.get_device_details,
    ),
    ToolSpec(
        name="find_devices_by_metric",
        category="read",
        description=(
            "Devices whose CURRENT value of a metric meets a condition, e.g. disk_free_pct lt 15 "
            "(low disk), memory_used_pct gte 85, battery_cycle_count gte 1000. "
            "For patterns over time use the detect_* tools instead."
        ),
        input_model=schemas.FindDevicesByMetricInput,
        handler=read_tools.find_devices_by_metric,
    ),
    ToolSpec(
        name="find_devices_by_os_version",
        category="read",
        description=(
            "Devices on macOS or Windows whose version compares to a given version, e.g. "
            "os_name=macOS comparison=lt version=15 for 'older than macOS 15'."
        ),
        input_model=schemas.FindDevicesByOsVersionInput,
        handler=read_tools.find_devices_by_os_version,
    ),
    ToolSpec(
        name="get_compliance_status",
        category="read",
        description=(
            "Current compliance results per device, filtered by check id, severity and status "
            "(default: failing). Returns the known check ids. When nothing matches, the "
            "evidence lists the snapshots that were checked."
        ),
        input_model=schemas.GetComplianceStatusInput,
        handler=read_tools.get_compliance_status,
    ),
    ToolSpec(
        name="search_installed_software",
        category="read",
        description="Devices that currently have software whose name contains the given text.",
        input_model=schemas.SearchInstalledSoftwareInput,
        handler=read_tools.search_installed_software,
    ),
    ToolSpec(
        name="get_device_history",
        category="read",
        description=(
            "Time series of chosen metrics for one device, one point per snapshot, to show "
            "how a value changed."
        ),
        input_model=schemas.GetDeviceHistoryInput,
        handler=read_tools.get_device_history,
    ),
    ToolSpec(
        name="detect_battery_end_of_life",
        category="insight",
        description=(
            "Laptop batteries at or approaching end of life, from service condition, cycle "
            "count against the rated limit, and full-charge capacity decline over time."
        ),
        input_model=schemas.DetectBatteryEndOfLifeInput,
        handler=insight_tools.detect_battery_end_of_life_tool,
    ),
    ToolSpec(
        name="detect_memory_pressure",
        category="insight",
        description=(
            "Devices consistently or intermittently constrained by RAM across the time window, "
            "with share of snapshots over the threshold, average and peak use, and RAM size."
        ),
        input_model=schemas.DetectMemoryPressureInput,
        handler=insight_tools.detect_memory_pressure_tool,
    ),
    ToolSpec(
        name="detect_storage_pressure",
        category="insight",
        description=(
            "Devices low on system disk space now or for much of the window, with the daily "
            "trend and projected days until critically full."
        ),
        input_model=schemas.DetectStoragePressureInput,
        handler=insight_tools.detect_storage_pressure_tool,
    ),
    ToolSpec(
        name="detect_compliance_drift",
        category="insight",
        description=(
            "Compliance changes over time: devices that started failing a check (drift) or "
            "recovered, plus fleet-wide failing counts at the start and end of the window."
        ),
        input_model=schemas.DetectComplianceDriftInput,
        handler=insight_tools.detect_compliance_drift_tool,
    ),
]

TOOL_REGISTRY: dict[str, ToolSpec] = {spec.name: spec for spec in _SPECS}
