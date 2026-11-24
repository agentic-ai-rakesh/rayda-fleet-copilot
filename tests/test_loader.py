import json
from pathlib import Path

import pytest

from fleet_copilot.exceptions import IngestionError
from fleet_copilot.ingestion.loader import load_telemetry, write_quarantine
from tests.conftest import write_ndjson


def test_valid_records_load(tmp_path: Path, make_record):
    path = write_ndjson(
        tmp_path / "d.ndjson",
        [make_record(), make_record(collected_at="2026-05-15T09:00:00Z"), ""],  # blank line skipped
    )
    result = load_telemetry(path)
    assert len(result.records) == 2
    assert result.rejected == []
    assert result.records[0].collected_at.tzinfo is not None


def test_missing_optional_sections_are_accepted(tmp_path: Path, make_record):
    record = make_record()
    for key in ("battery", "network", "installed_software", "compliance_results"):
        record.pop(key)
    result = load_telemetry(write_ndjson(tmp_path / "d.ndjson", [record]))
    assert result.records[0].battery is None
    assert result.records[0].compliance_results == []


@pytest.mark.parametrize(
    ("line", "reason_prefix"),
    [
        ("{not json", "invalid_json"),
        ("[1, 2]", "invalid_json"),
        ({"device_id": "X"}, "schema_validation"),
    ],
)
def test_bad_lines_are_rejected_not_fatal(tmp_path: Path, make_record, line, reason_prefix):
    path = write_ndjson(tmp_path / "d.ndjson", [make_record(), line])
    result = load_telemetry(path)
    assert len(result.records) == 1
    assert len(result.rejected) == 1
    assert result.rejected[0].line_number == 2
    assert result.rejected[0].reason.startswith(reason_prefix)
    assert result.reject_ratio == 0.5


@pytest.mark.parametrize(
    "overrides",
    [
        {"memory": {"total_memory_bytes": 10, "used_memory_bytes": 20, "free_memory_bytes": 0}},
        {"disk_volumes": []},
        {
            "compliance_results": [
                {"check_id": "screen_lock", "status": "unknown", "severity": "medium"}
            ]
        },
    ],
)
def test_inconsistent_values_are_rejected(tmp_path: Path, make_record, overrides):
    result = load_telemetry(write_ndjson(tmp_path / "d.ndjson", [make_record(**overrides)]))
    assert result.records == []
    assert result.rejected[0].reason.startswith("schema_validation")


def test_duplicate_snapshots_are_rejected(tmp_path: Path, make_record):
    result = load_telemetry(write_ndjson(tmp_path / "d.ndjson", [make_record(), make_record()]))
    assert len(result.records) == 1
    assert result.rejected[0].reason.startswith("duplicate_snapshot")


def test_missing_dataset_raises(tmp_path: Path):
    with pytest.raises(IngestionError, match="not found"):
        load_telemetry(tmp_path / "absent.ndjson")


def test_quarantine_file_records_reason_and_raw_line(tmp_path: Path, make_record):
    result = load_telemetry(write_ndjson(tmp_path / "d.ndjson", [make_record(), "{bad"]))
    quarantine = tmp_path / "q" / "rejected.ndjson"
    write_quarantine(result.rejected, quarantine)
    entry = json.loads(quarantine.read_text().strip())
    assert entry["line_number"] == 2
    assert entry["raw_line"] == "{bad"
    assert entry["reason"].startswith("invalid_json")
