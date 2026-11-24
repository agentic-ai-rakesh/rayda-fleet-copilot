import sqlite3
from pathlib import Path

import pytest

from fleet_copilot.audit import AuditLogger, AuditStatus
from fleet_copilot.audit.audit_logger import GENESIS_HASH
from fleet_copilot.exceptions import AuditIntegrityError
from fleet_copilot.utils.logging_utils import set_run_id


@pytest.fixture
def audit(tmp_path: Path) -> AuditLogger:
    return AuditLogger(tmp_path / "audit.db")


def _record(audit: AuditLogger, action: str = "ingest_dataset", **kwargs):
    return audit.record(
        event_type="ingestion", action=action, status=AuditStatus.SUCCEEDED, **kwargs
    )


def test_events_are_hash_chained(audit: AuditLogger):
    set_run_id("run-test")
    first = _record(audit, details={"rows": 1})
    second = _record(audit, tenant_id="acme-001")

    assert first.prev_hash == GENESIS_HASH
    assert second.prev_hash == first.entry_hash
    assert second.run_id == "run-test"
    assert audit.verify_integrity() == 2
    assert [e.event_id for e in audit.list_events()] == [second.event_id, first.event_id]


@pytest.mark.parametrize(
    "statement",
    ["UPDATE audit_log SET status = 'failed'", "DELETE FROM audit_log"],
)
def test_audit_log_is_append_only(audit: AuditLogger, tmp_path: Path, statement: str):
    _record(audit)
    with (
        sqlite3.connect(tmp_path / "audit.db") as conn,
        pytest.raises(sqlite3.IntegrityError, match="append-only"),
    ):
        conn.execute(statement)


def test_tampering_is_detected(audit: AuditLogger, tmp_path: Path):
    _record(audit, details={"rows": 1})
    _record(audit)
    with sqlite3.connect(tmp_path / "audit.db") as conn:
        # Simulate an attacker who bypasses the guard trigger.
        conn.execute("DROP TRIGGER audit_log_block_update")
        conn.execute("UPDATE audit_log SET details = '{\"rows\":999}' WHERE sequence = 1")

    with pytest.raises(AuditIntegrityError, match="modified"):
        audit.verify_integrity()
