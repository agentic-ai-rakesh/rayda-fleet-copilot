"""Shared fixtures: isolated settings and a factory for valid raw records."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from fleet_copilot.audit import AuditLogger
from fleet_copilot.config.settings import PROJECT_ROOT, Settings, load_settings
from fleet_copilot.guardrails.tenant_context import TenantContext
from fleet_copilot.ingestion.pipeline import run_ingestion

REAL_DATASET = PROJECT_ROOT / "data" / "raw" / "device-telemetry-dataset.ndjson"

_BASE_RECORD: dict[str, Any] = {
    "device_id": "DEV001",
    "company_id": "acme-001",
    "employee_id": "emp-acme-1",
    "collected_at": "2026-05-14T09:54:00Z",
    "agent_version": "42.0.0",
    "os": {
        "platform": "darwin",
        "product_name": "macOS",
        "product_version": "14.5",
        "build_version": "23F79",
        "architecture": "arm64",
        "hostname": "acme-macbook-1",
    },
    "device_identity": {"serial_number": "DEV001", "model_name": "MacBook Pro"},
    "memory": {
        "ram_bytes": 1000,
        "total_memory_bytes": 1000,
        "used_memory_bytes": 600,
        "free_memory_bytes": 400,
    },
    "disk_volumes": [
        {
            "volume_name": "Macintosh HD",
            "mount_point": "/",
            "file_system": "apfs",
            "size_bytes": 1000,
            "available_bytes": 100,
            "encrypted": True,
        }
    ],
    "battery": {
        "battery_present": True,
        "charging_status": "charging",
        "percentage": 45,
        "condition": "Normal",
        "cycle_count": 512,
        "full_charge_capacity": 6163,
    },
    "network": [{"address": "192.168.0.2"}],
    "installed_software": [{"name": "Slack", "version": "4.39", "publisher": "Slack"}],
    "compliance_results": [
        {"check_id": "disk_encryption", "status": "pass", "severity": "high"},
        {"check_id": "os_up_to_date", "status": "fail", "severity": "medium"},
    ],
}

RecordFactory = Callable[..., dict[str, Any]]


@pytest.fixture
def make_record() -> RecordFactory:
    """Return a valid raw record; keyword arguments replace top-level fields."""

    def factory(**overrides: Any) -> dict[str, Any]:
        record = copy.deepcopy(_BASE_RECORD)
        record.update(overrides)
        return record

    return factory


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Default settings with every output path inside a temporary directory."""
    loaded = load_settings(PROJECT_ROOT / "config" / "settings.yaml", project_root=tmp_path)
    loaded.paths.raw_dataset = tmp_path / "data" / "raw" / "dataset.ndjson"
    return loaded


def write_ndjson(path: Path, lines: list[dict[str, Any] | str]) -> Path:
    """Write records (dicts) or raw text lines to an NDJSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write((line if isinstance(line, str) else json.dumps(line)) + "\n")
    return path


@pytest.fixture(scope="session")
def fleet_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Settings pointing at a fleet database built once from the real dataset."""
    if not REAL_DATASET.exists():
        pytest.skip("real dataset not present")
    root = tmp_path_factory.mktemp("fleet")
    loaded = load_settings(PROJECT_ROOT / "config" / "settings.yaml", project_root=root)
    loaded.paths.raw_dataset = REAL_DATASET
    run_ingestion(loaded, AuditLogger(loaded.paths.audit_db))
    return loaded


def tenant(company_id: str, name: str, actor: str = "admin@test") -> TenantContext:
    return TenantContext(company_id=company_id, company_name=name, actor=actor)


ACME = ("acme-001", "Acme")
GLOBEX = ("globex-002", "Globex")
INITECH = ("initech-003", "Initech")
