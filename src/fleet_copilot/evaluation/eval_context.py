"""Shared state for one eval run: settings, ground truth, tenants, scratch stores."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from fleet_copilot.audit import AuditLogger
from fleet_copilot.config import Settings
from fleet_copilot.db.connection import connect
from fleet_copilot.evaluation.ground_truth import GroundTruth
from fleet_copilot.guardrails.tenant_context import TenantContext, resolve_tenant
from fleet_copilot.tools.executor import ToolExecutor

EVAL_ACTOR = "eval-runner"


@dataclass
class EvalContext:
    settings: Settings  # audit/actions/checkpoints already point at the run folder
    ground_truth: GroundTruth
    run_dir: Path

    def tenant(self, tenant: str) -> TenantContext:
        with closing(connect(self.settings.paths.fleet_db, read_only=True)) as conn:
            return resolve_tenant(conn, tenant, actor=EVAL_ACTOR)

    def tool_executor(self, tenant: str) -> ToolExecutor:
        return ToolExecutor(self.settings, self.tenant(tenant), self.audit)

    @cached_property
    def audit(self) -> AuditLogger:
        return AuditLogger(self.settings.paths.audit_db)

    def latest_snapshot_id(self, device_id: str) -> str:
        return self._lookup(
            "SELECT snapshot_id FROM snapshots WHERE device_id = ? "
            "ORDER BY collected_at DESC LIMIT 1",
            device_id,
        )

    def employee_of(self, device_id: str) -> str:
        return self._lookup("SELECT employee_id FROM devices WHERE device_id = ?", device_id)

    def _lookup(self, sql: str, value: str) -> str:
        # Harness-only, unscoped lookups used to build test inputs.
        with closing(connect(self.settings.paths.fleet_db, read_only=True)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(sql, (value,)).fetchone()
        if row is None:
            raise KeyError(value)
        return row[0]
