"""End-to-end ingestion tests, including fallbacks and the real dataset."""

import shutil
import sqlite3
from contextlib import closing

import pytest

from fleet_copilot.audit import AuditLogger
from fleet_copilot.db.connection import connect
from fleet_copilot.exceptions import DatabaseNotReadyError, IngestionError
from fleet_copilot.ingestion.pipeline import run_ingestion
from fleet_copilot.profiling.data_profiler import build_profile, render_markdown
from tests.conftest import REAL_DATASET, write_ndjson


def _audit_statuses(settings) -> list[str]:
    events = AuditLogger(settings.paths.audit_db).list_events()
    return [e.status for e in reversed(events)]


def _snapshot_count(settings) -> int:
    with closing(connect(settings.paths.fleet_db, read_only=True)) as conn:
        return conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]


def test_successful_ingestion_builds_db_and_audits(settings, make_record):
    write_ndjson(
        settings.paths.raw_dataset,
        [make_record(), make_record(collected_at="2026-05-15T09:00:00Z")],
    )
    summary = run_ingestion(settings, AuditLogger(settings.paths.audit_db))

    assert summary.snapshots == 2
    assert summary.data_end == "2026-05-15T09:00:00Z"
    assert _snapshot_count(settings) == 2
    assert _audit_statuses(settings) == ["started", "succeeded"]
    assert not settings.paths.fleet_db.with_name("fleet.db.tmp").exists()


def test_too_many_rejects_keeps_previous_database(settings, make_record):
    audit = AuditLogger(settings.paths.audit_db)
    write_ndjson(settings.paths.raw_dataset, [make_record()])
    run_ingestion(settings, audit)

    # Second run: 1 good + 1 bad line = 50% rejects, above the 5% limit.
    write_ndjson(
        settings.paths.raw_dataset,
        [make_record(collected_at="2026-05-16T09:00:00Z"), "{broken"],
    )
    with pytest.raises(IngestionError, match="failed validation"):
        run_ingestion(settings, audit)

    assert _snapshot_count(settings) == 1  # previous good DB untouched
    assert settings.paths.quarantine_file.read_text().count("\n") == 1
    assert _audit_statuses(settings) == ["started", "succeeded", "started", "failed"]
    assert audit.verify_integrity() == 4


def test_rejects_within_limit_are_quarantined(settings, make_record):
    settings.ingestion.max_reject_ratio = 0.5
    write_ndjson(settings.paths.raw_dataset, [make_record(), "{broken"])
    summary = run_ingestion(settings, AuditLogger(settings.paths.audit_db))
    assert (summary.records_loaded, summary.records_rejected) == (1, 1)


def test_missing_dataset_is_audited_as_failure(settings):
    with pytest.raises(IngestionError):
        run_ingestion(settings, AuditLogger(settings.paths.audit_db))
    assert _audit_statuses(settings) == ["started", "failed"]


def test_profile_requires_database(settings):
    with pytest.raises(DatabaseNotReadyError):
        build_profile(settings.paths.fleet_db)


def test_database_is_read_only_for_consumers(settings, make_record):
    write_ndjson(settings.paths.raw_dataset, [make_record()])
    run_ingestion(settings, AuditLogger(settings.paths.audit_db))
    with (
        closing(connect(settings.paths.fleet_db, read_only=True)) as conn,
        pytest.raises(sqlite3.OperationalError, match="readonly"),
    ):
        conn.execute("DELETE FROM snapshots")


@pytest.mark.skipif(not REAL_DATASET.exists(), reason="real dataset not present")
def test_real_dataset_matches_known_shape(settings):
    settings.paths.raw_dataset.parent.mkdir(parents=True)
    shutil.copy(REAL_DATASET, settings.paths.raw_dataset)
    summary = run_ingestion(settings, AuditLogger(settings.paths.audit_db))

    assert (summary.companies, summary.devices, summary.employees) == (3, 25, 14)
    assert (summary.snapshots, summary.records_rejected) == (750, 0)
    assert summary.data_end == "2026-06-12T11:00:00Z"

    with closing(connect(settings.paths.fleet_db, read_only=True)) as conn:
        per_company = dict(
            conn.execute("SELECT company_id, COUNT(*) FROM devices GROUP BY company_id").fetchall()
        )
        unparsed = conn.execute(
            "SELECT COUNT(*) FROM snapshots WHERE os_version_major IS NULL"
        ).fetchone()[0]
    assert per_company == {"acme-001": 10, "globex-002": 8, "initech-003": 7}
    assert unparsed == 0

    report = render_markdown(build_profile(settings.paths.fleet_db))
    assert "No snapshot fails `disk_encryption`" in report
