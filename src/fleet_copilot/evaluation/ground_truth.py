"""Expected answers computed straight from the raw NDJSON.

Deliberately independent of the app: no database, no repository, no tools,
no detectors. Only the definitions (thresholds) are shared, because they are
the specification. If ingestion, a query or a detector is wrong, the tool
suite disagrees with this module.

Each truth is looked up by key, optionally with an argument:
"low_disk_devices", "software_devices:uTorrent".
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from fleet_copilot.config.settings import Settings

Truth = set[str] | int


@dataclass
class GroundTruth:
    dataset: Path
    settings: Settings

    # ------------------------------------------------------------ raw access
    @cached_property
    def _records(self) -> list[dict[str, Any]]:
        with self.dataset.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    @cached_property
    def _history(self) -> dict[str, list[dict[str, Any]]]:
        history: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in self._records:
            history[record["device_id"]].append(record)
        for series in history.values():
            series.sort(key=lambda r: r["collected_at"])
        return dict(history)

    def _devices(self, tenant: str) -> dict[str, list[dict[str, Any]]]:
        return {d: s for d, s in self._history.items() if s[-1]["company_id"] == tenant}

    def _latest(self, tenant: str) -> list[dict[str, Any]]:
        return [series[-1] for series in self._devices(tenant).values()]

    def _is_laptop(self, record: dict[str, Any]) -> bool:
        desktops = {m.casefold() for m in self.settings.device_classification.desktop_models}
        return record["device_identity"]["model_name"].casefold() not in desktops

    @staticmethod
    def _system_free_pct(record: dict[str, Any]) -> float:
        volume = next(v for v in record["disk_volumes"] if v["mount_point"] in ("/", "C:\\"))
        return 100.0 * volume["available_bytes"] / volume["size_bytes"]

    @staticmethod
    def _memory_used_pct(record: dict[str, Any]) -> float:
        memory = record["memory"]
        return 100.0 * memory["used_memory_bytes"] / memory["total_memory_bytes"]

    # ---------------------------------------------------------------- truths
    def device_ids(self, tenant: str) -> set[str]:
        return set(self._devices(tenant))

    def all_device_ids(self) -> set[str]:
        return set(self._history)

    def low_disk_devices(self, tenant: str) -> set[str]:
        limit = self.settings.thresholds.low_disk_free_pct
        return {r["device_id"] for r in self._latest(tenant) if self._system_free_pct(r) < limit}

    def high_severity_failing_laptops(self, tenant: str) -> set[str]:
        return {
            r["device_id"]
            for r in self._latest(tenant)
            if self._is_laptop(r)
            and any(
                c["severity"] in ("high", "critical") and c["status"] == "fail"
                for c in r["compliance_results"]
            )
        }

    def macos_older_than_15_devices(self, tenant: str) -> set[str]:
        return {
            r["device_id"]
            for r in self._latest(tenant)
            if r["os"]["product_name"] == "macOS"
            and int(r["os"]["product_version"].split(".")[0]) < 15
        }

    def macos_older_than_15_count(self, tenant: str) -> int:
        return len(self.macos_older_than_15_devices(tenant))

    def failing_check_devices(self, tenant: str, check_id: str) -> set[str]:
        return {
            r["device_id"]
            for r in self._latest(tenant)
            if any(
                c["check_id"] == check_id and c["status"] == "fail" for c in r["compliance_results"]
            )
        }

    def software_devices(self, tenant: str, name: str) -> set[str]:
        needle = name.casefold()
        return {
            r["device_id"]
            for r in self._latest(tenant)
            if any(needle in s["name"].casefold() for s in r["installed_software"])
        }

    def battery_eol_devices(self, tenant: str) -> set[str]:
        t = self.settings.thresholds
        found = set()
        for device_id, series in self._devices(tenant).items():
            readings = [r for r in series if r.get("battery")]
            if len(readings) < self.settings.insights.min_snapshots:
                continue
            latest = readings[-1]["battery"]
            if (
                latest["condition"] in t.battery_service_conditions
                or latest["cycle_count"] >= t.battery_cycle_limit
            ):
                found.add(device_id)
        return found

    def _share(self, tenant: str, predicate: Callable[[dict[str, Any]], bool]) -> dict[str, float]:
        return {
            device_id: sum(predicate(r) for r in series) / len(series)
            for device_id, series in self._devices(tenant).items()
            if len(series) >= self.settings.insights.min_snapshots
        }

    def memory_consistent_devices(self, tenant: str) -> set[str]:
        limit = self.settings.thresholds.high_memory_used_pct
        shares = self._share(tenant, lambda r: self._memory_used_pct(r) >= limit)
        return {d for d, s in shares.items() if s >= self.settings.insights.consistent_share}

    def memory_pressure_devices(self, tenant: str) -> set[str]:
        limit = self.settings.thresholds.high_memory_used_pct
        shares = self._share(tenant, lambda r: self._memory_used_pct(r) >= limit)
        return {d for d, s in shares.items() if s >= self.settings.insights.intermittent_share}

    def storage_pressure_devices(self, tenant: str) -> set[str]:
        limit = self.settings.thresholds.low_disk_free_pct
        shares = self._share(tenant, lambda r: self._system_free_pct(r) < limit)
        return {
            d
            for d, s in shares.items()
            if s >= self.settings.insights.intermittent_share
            or self._system_free_pct(self._history[d][-1]) < limit
        }

    def compliance_drift_devices(self, tenant: str, check_id: str) -> set[str]:
        found = set()
        for device_id, series in self._devices(tenant).items():
            statuses = [
                c["status"]
                for r in series
                for c in r["compliance_results"]
                if c["check_id"] == check_id
            ]
            if (
                len(statuses) >= self.settings.insights.min_snapshots
                and statuses[0] == "pass"
                and statuses[-1] == "fail"
            ):
                found.add(device_id)
        return found

    def replacement_candidates(self, tenant: str) -> set[str]:
        """Devices a replacement may be justified for (battery at end of life)."""
        return self.battery_eol_devices(tenant)

    # -------------------------------------------------------------- lookup
    def resolve(self, key: str, tenant: str) -> Truth:
        name, _, argument = key.partition(":")
        if name.startswith("_") or not hasattr(self, name):
            raise KeyError(f"Unknown ground truth '{name}'.")
        function = getattr(self, name)
        return function(tenant, argument) if argument else function(tenant)
