"""State-changing action tools and their executor.

These are never given to the model as callable tools. The flow is:
model proposes -> evidence gate checks -> human approves -> ActionExecutor runs.
Execution re-checks tenant ownership (defence in depth), is idempotent, and
is audited. Integrations are simulated by the ActionStore.
"""

from __future__ import annotations

import logging
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from fleet_copilot.audit import AuditLogger, AuditStatus
from fleet_copilot.config import Settings
from fleet_copilot.db.action_store import ActionStore
from fleet_copilot.db.connection import connect
from fleet_copilot.db.fleet_repository import FleetRepository
from fleet_copilot.guardrails.tenant_context import TenantContext
from fleet_copilot.tools import schemas
from fleet_copilot.utils.logging_utils import get_run_id

logger = logging.getLogger(__name__)

AUDIT_EVENT_TYPE = "action"


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    target: Literal["device_id", "employee_id"]
    # Arguments that make two actions "the same action". Free text (reason,
    # note, spec) is excluded, so a re-worded duplicate is still a duplicate.
    identity: tuple[str, ...]

    @property
    def argument_names(self) -> list[str]:
        return list(self.input_model.model_fields)

    def identity_of(self, arguments: dict) -> dict:
        return {name: arguments[name] for name in self.identity}


ACTION_REGISTRY: dict[str, ActionSpec] = {
    spec.name: spec
    for spec in [
        ActionSpec(
            "create_upgrade_order",
            "Order a RAM, storage or battery upgrade for a device. Arguments: device_id, "
            "component (ram|storage|battery), spec (short, max 64 chars, e.g. '16 GB').",
            schemas.CreateUpgradeOrderInput,
            "device_id",
            ("device_id", "component"),
        ),
        ActionSpec(
            "open_remediation_ticket",
            "Open an IT ticket to fix a failing compliance check on a device. "
            "Arguments: device_id, check_id, note (max 500 chars).",
            schemas.OpenRemediationTicketInput,
            "device_id",
            ("device_id", "check_id"),
        ),
        ActionSpec(
            "flag_device_for_replacement",
            "Flag a device for hardware replacement. Arguments: device_id, reason (max 500 chars).",
            schemas.FlagDeviceForReplacementInput,
            "device_id",
            ("device_id",),
        ),
        ActionSpec(
            "notify_employee",
            "Send a message to the employee who uses a device. "
            "Arguments: employee_id, message (max 500 chars).",
            schemas.NotifyEmployeeInput,
            "employee_id",
            ("employee_id", "message"),
        ),
    ]
}


@dataclass(frozen=True)
class ActionResult:
    status: Literal["executed", "duplicate", "failed"]
    message: str
    action_id: str | None = None


class ActionExecutor:
    def __init__(
        self, settings: Settings, tenant: TenantContext, audit: AuditLogger, store: ActionStore
    ) -> None:
        self._settings = settings
        self._tenant = tenant
        self._audit = audit
        self._store = store

    def execute(self, proposal: dict[str, Any], approved_by: str) -> ActionResult:
        """Run one approved proposal (a dict as stored in the graph state)."""
        action = proposal["action"]
        spec = ACTION_REGISTRY.get(action)
        result: ActionResult
        if spec is None:
            result = ActionResult("failed", f"Unknown action '{action}'.")
        else:
            result = self._run(spec, proposal, approved_by)

        self._audit.record(
            event_type=AUDIT_EVENT_TYPE,
            action=action,
            status=AuditStatus.SUCCEEDED if result.status != "failed" else AuditStatus.FAILED,
            actor=approved_by,
            tenant_id=self._tenant.company_id,
            details={
                "proposal_id": proposal["proposal_id"],
                "arguments": proposal["arguments"],
                "evidence_ids": proposal["evidence_ids"],
                "result": result.status,
                "action_id": result.action_id,
                "message": result.message,
            },
        )
        return result

    def _run(self, spec: ActionSpec, proposal: dict[str, Any], approved_by: str) -> ActionResult:
        try:
            arguments = spec.input_model.model_validate(proposal["arguments"]).model_dump()
        except ValidationError as exc:
            return ActionResult("failed", f"Invalid arguments: {exc.error_count()} error(s).")

        # Defence in depth: re-check the target belongs to this tenant right before acting.
        with closing(connect(self._settings.paths.fleet_db, read_only=True)) as conn:
            repo = FleetRepository(conn, self._tenant)
            target = arguments[spec.target]
            owned = (
                repo.latest_device_state(target) is not None
                if spec.target == "device_id"
                else bool(repo.employee_device_ids(target))
            )
        if not owned:
            logger.warning("Blocked %s on %s: not in tenant scope", spec.name, target)
            return ActionResult("failed", f"{target} is not part of {self._tenant.company_name}.")

        stored = self._store.record(
            tenant_id=self._tenant.company_id,
            action_type=spec.name,
            arguments=arguments,
            identity=spec.identity_of(arguments),
            justification=proposal["justification"],
            evidence_ids=proposal["evidence_ids"],
            proposal_id=proposal["proposal_id"],
            approved_by=approved_by,
            run_id=get_run_id(),
        )
        if not stored.created:
            return ActionResult(
                "duplicate",
                f"Already recorded as {stored.action_id}; not repeated.",
                stored.action_id,
            )
        return ActionResult("executed", f"Recorded as {stored.action_id}.", stored.action_id)
