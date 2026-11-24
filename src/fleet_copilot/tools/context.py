"""Everything a tool handler may use, built by the executor for one call."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from fleet_copilot.config.settings import InsightSettings, ThresholdSettings, ToolSettings
from fleet_copilot.db.fleet_repository import FleetRepository
from fleet_copilot.utils.stats_utils import parse_iso
from fleet_copilot.utils.time_utils import to_iso_utc


class DeviceNotInScopeError(Exception):
    """A device id is not visible to the current tenant (missing or another company's)."""

    def __init__(self, device_id: str) -> None:
        super().__init__(device_id)
        self.device_id = device_id


class ToolInputError(Exception):
    """Arguments passed schema validation but cannot be used (e.g. unparseable version)."""


@dataclass(frozen=True)
class ToolContext:
    repo: FleetRepository
    thresholds: ThresholdSettings
    insights: InsightSettings
    tool_settings: ToolSettings
    as_of: str

    def window_start(self, days: int | None) -> str:
        """ISO start of a look-back window ending at the dataset's as-of time (not wall clock)."""
        span = days or self.tool_settings.default_history_days
        return to_iso_utc(parse_iso(self.as_of) - timedelta(days=span))
