"""Record of executed (human-approved) actions.

This is the stand-in for the real integrations (ticketing, procurement,
messaging). Writes are idempotent: the same approved action for the same
tenant, action type and target (e.g. device + component) is stored once, even
if the free-text reason differs, so the same upgrade is never ordered twice.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fleet_copilot.utils.hash_utils import canonical_json, sha256_text
from fleet_copilot.utils.time_utils import utc_now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS actions (
    action_id        TEXT PRIMARY KEY,
    idempotency_key  TEXT NOT NULL UNIQUE,
    tenant_id        TEXT NOT NULL,
    action_type      TEXT NOT NULL,
    arguments        TEXT NOT NULL,
    justification    TEXT NOT NULL,
    evidence_ids     TEXT NOT NULL,
    proposal_id      TEXT NOT NULL,
    approved_by      TEXT NOT NULL,
    executed_at      TEXT NOT NULL,
    run_id           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_actions_tenant ON actions (tenant_id, executed_at);
"""


@dataclass(frozen=True)
class StoredAction:
    action_id: str
    created: bool  # False when an identical action already existed


def idempotency_key(tenant_id: str, action_type: str, identity: dict[str, Any]) -> str:
    """Same tenant, action and identifying arguments -> same key (free text ignored)."""
    return sha256_text(canonical_json({"t": tenant_id, "a": action_type, "id": identity}))


class ActionStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def record(
        self,
        *,
        tenant_id: str,
        action_type: str,
        arguments: dict[str, Any],
        identity: dict[str, Any],
        justification: str,
        evidence_ids: list[str],
        proposal_id: str,
        approved_by: str,
        run_id: str,
    ) -> StoredAction:
        key = idempotency_key(tenant_id, action_type, identity)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT action_id FROM actions WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if existing:
                return StoredAction(existing["action_id"], created=False)
            action_id = f"act-{uuid.uuid4().hex[:12]}"
            conn.execute(
                """INSERT INTO actions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    action_id,
                    key,
                    tenant_id,
                    action_type,
                    canonical_json(arguments),
                    justification,
                    json.dumps(evidence_ids),
                    proposal_id,
                    approved_by,
                    utc_now_iso(),
                    run_id,
                ),
            )
        return StoredAction(action_id, created=True)

    def list_actions(self, tenant_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM actions WHERE tenant_id = ? ORDER BY executed_at", (tenant_id,)
            ).fetchall()
        return [
            {
                **dict(row),
                "arguments": json.loads(row["arguments"]),
                "evidence_ids": json.loads(row["evidence_ids"]),
            }
            for row in rows
        ]

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()
