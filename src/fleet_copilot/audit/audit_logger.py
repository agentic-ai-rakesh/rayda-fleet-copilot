"""Append-only, tamper-evident audit log stored in its own SQLite file.

Audit criteria:
- Append-only: database triggers reject UPDATE and DELETE on audit_log.
- Tamper-evident: each entry stores the SHA-256 of its own content plus the
  previous entry's hash. `verify_integrity()` recomputes the whole chain.
- Fail closed: if an event cannot be written, AuditError is raised and the
  caller must stop. An action that cannot be audited must not happen.
- Separate file: the audit trail survives a rebuild of the fleet database.
- Every entry records who (actor), which tenant, what (event_type/action),
  the outcome (status), and the run_id that links it to the app logs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from fleet_copilot.exceptions import AuditError, AuditIntegrityError
from fleet_copilot.utils.hash_utils import canonical_json, sha256_text
from fleet_copilot.utils.logging_utils import get_run_id
from fleet_copilot.utils.time_utils import utc_now_iso

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64
SYSTEM_ACTOR = "system"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    sequence     INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     TEXT    NOT NULL UNIQUE,
    recorded_at  TEXT    NOT NULL,
    run_id       TEXT    NOT NULL,
    actor        TEXT    NOT NULL,
    tenant_id    TEXT,
    event_type   TEXT    NOT NULL,
    action       TEXT    NOT NULL,
    status       TEXT    NOT NULL,
    details      TEXT    NOT NULL,
    prev_hash    TEXT    NOT NULL,
    entry_hash   TEXT    NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_log_block_update
BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only');
END;
CREATE TRIGGER IF NOT EXISTS audit_log_block_delete
BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log is append-only');
END;
"""


class AuditStatus(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"  # blocked by a guardrail (e.g. cross-tenant access)
    PROPOSED = "proposed"  # action proposal passed the evidence gate, awaiting approval
    APPROVED = "approved"  # a human approved the action
    REJECTED = "rejected"  # a human rejected the action (or did not decide)


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    recorded_at: str
    run_id: str
    actor: str
    tenant_id: str | None
    event_type: str
    action: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)
    prev_hash: str = GENESIS_HASH
    entry_hash: str = ""

    def content_hash(self) -> str:
        """Hash of everything except entry_hash itself."""
        content = {
            "event_id": self.event_id,
            "recorded_at": self.recorded_at,
            "run_id": self.run_id,
            "actor": self.actor,
            "tenant_id": self.tenant_id,
            "event_type": self.event_type,
            "action": self.action,
            "status": self.status,
            "details": self.details,
            "prev_hash": self.prev_hash,
        }
        return sha256_text(canonical_json(content))


class AuditLogger:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(_SCHEMA)
        except (OSError, sqlite3.Error) as exc:
            raise AuditError(f"Cannot open audit log at {db_path}: {exc}") from exc

    def record(
        self,
        *,
        event_type: str,
        action: str,
        status: AuditStatus,
        details: dict[str, Any] | None = None,
        actor: str = SYSTEM_ACTOR,
        tenant_id: str | None = None,
    ) -> AuditEvent:
        """Append one event. Raises AuditError if it cannot be stored."""
        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")  # serialise writers so the chain stays linear
                row = conn.execute(
                    "SELECT entry_hash FROM audit_log ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                draft = AuditEvent(
                    event_id=f"evt-{uuid.uuid4().hex}",
                    recorded_at=utc_now_iso(),
                    run_id=get_run_id(),
                    actor=actor,
                    tenant_id=tenant_id,
                    event_type=event_type,
                    action=action,
                    status=AuditStatus(status).value,
                    details=json.loads(canonical_json(details or {})),
                    prev_hash=row["entry_hash"] if row else GENESIS_HASH,
                )
                event = AuditEvent(**{**draft.__dict__, "entry_hash": draft.content_hash()})
                conn.execute(
                    """
                    INSERT INTO audit_log (event_id, recorded_at, run_id, actor, tenant_id,
                        event_type, action, status, details, prev_hash, entry_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.recorded_at,
                        event.run_id,
                        event.actor,
                        event.tenant_id,
                        event.event_type,
                        event.action,
                        event.status,
                        canonical_json(event.details),
                        event.prev_hash,
                        event.entry_hash,
                    ),
                )
        except sqlite3.Error as exc:
            logger.error("Audit write failed for %s/%s: %s", event_type, action, exc)
            raise AuditError(f"Audit write failed: {exc}") from exc

        logger.info(
            "audit %s.%s -> %s",
            event.event_type,
            event.action,
            event.status,
            extra={"context": {"event_id": event.event_id, "tenant_id": event.tenant_id}},
        )
        return event

    def list_events(self, limit: int = 50) -> list[AuditEvent]:
        """Most recent events first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY sequence DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_event(row) for row in rows]

    def verify_integrity(self) -> int:
        """Recompute the hash chain. Returns the number of entries checked."""
        expected_prev = GENESIS_HASH
        count = 0
        with self._connect() as conn:
            for row in conn.execute("SELECT * FROM audit_log ORDER BY sequence ASC"):
                event = _row_to_event(row)
                if event.prev_hash != expected_prev:
                    raise AuditIntegrityError(
                        f"Chain broken before entry #{row['sequence']} ({event.event_id})."
                    )
                if event.content_hash() != event.entry_hash:
                    raise AuditIntegrityError(
                        f"Entry #{row['sequence']} ({event.event_id}) was modified."
                    )
                expected_prev = event.entry_hash
                count += 1
        return count

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # isolation_level=None: we issue BEGIN explicitly; the context manager commits.
        conn = sqlite3.connect(self._db_path, isolation_level=None, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            if conn.in_transaction:
                conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()


def _row_to_event(row: sqlite3.Row) -> AuditEvent:
    return AuditEvent(
        event_id=row["event_id"],
        recorded_at=row["recorded_at"],
        run_id=row["run_id"],
        actor=row["actor"],
        tenant_id=row["tenant_id"],
        event_type=row["event_type"],
        action=row["action"],
        status=row["status"],
        details=json.loads(row["details"]),
        prev_hash=row["prev_hash"],
        entry_hash=row["entry_hash"],
    )
