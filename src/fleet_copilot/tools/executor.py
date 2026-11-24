"""Runs one tool call safely for one tenant.

For every call the executor:
1. Validates arguments against the tool's schema (bad args -> error result the
   model can correct; it never reaches the database).
2. Opens a read-only connection and a repository scoped to the session tenant.
3. Runs the handler, then truncates to `tools.max_rows`, keeping evidence in step.
4. Logs and audits the call: succeeded, failed, or denied (cross-tenant).

Fallback: no tool error ever crashes the agent loop. Unexpected exceptions
become an `error` result with a generic message (details go to the log),
so the model can say it could not answer instead of guessing.
Audit failures are the one exception: they propagate (fail closed).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from contextlib import closing
from typing import Any

from pydantic import ValidationError

from fleet_copilot.audit import AuditLogger, AuditStatus
from fleet_copilot.config import Settings
from fleet_copilot.db.connection import connect
from fleet_copilot.db.fleet_repository import FleetRepository
from fleet_copilot.exceptions import AuditError, FleetCopilotError
from fleet_copilot.guardrails.tenant_context import TenantContext
from fleet_copilot.tools.context import DeviceNotInScopeError, ToolContext, ToolInputError
from fleet_copilot.tools.registry import TOOL_REGISTRY, ToolSpec
from fleet_copilot.tools.results import ToolOutput, ToolResult

logger = logging.getLogger(__name__)

AUDIT_EVENT_TYPE = "tool_call"


class ToolExecutor:
    def __init__(
        self,
        settings: Settings,
        tenant: TenantContext,
        audit: AuditLogger,
        registry: Mapping[str, ToolSpec] = TOOL_REGISTRY,
    ) -> None:
        self._settings = settings
        self._tenant = tenant
        self._audit = audit
        self._registry = registry

    @property
    def tenant(self) -> TenantContext:
        return self._tenant

    def available_tools(self) -> list[ToolSpec]:
        return list(self._registry.values())

    def execute(self, tool_name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        started = time.perf_counter()
        args = dict(arguments or {})
        spec = self._registry.get(tool_name)
        if spec is None:
            known = ", ".join(sorted(self._registry))
            return self._finish(
                tool_name,
                args,
                started,
                AuditStatus.FAILED,
                self._error(tool_name, f"Unknown tool '{tool_name}'. Available tools: {known}."),
            )

        try:
            parsed = spec.input_model.model_validate(args)
        except ValidationError as exc:
            message = f"Invalid arguments for {tool_name}: {_short_validation(exc)}"
            return self._finish(
                tool_name, args, started, AuditStatus.FAILED, self._error(tool_name, message)
            )

        try:
            with closing(connect(self._settings.paths.fleet_db, read_only=True)) as conn:
                repo = FleetRepository(conn, self._tenant)
                context = ToolContext(
                    repo=repo,
                    thresholds=self._settings.thresholds,
                    insights=self._settings.insights,
                    tool_settings=self._settings.tools,
                    as_of=repo.as_of() or "",
                )
                try:
                    output = spec.handler(context, parsed)
                except DeviceNotInScopeError as exc:
                    return self._device_not_in_scope(tool_name, args, started, repo, exc.device_id)
                except ToolInputError as exc:
                    result = self._error(tool_name, f"Invalid arguments for {tool_name}: {exc}")
                    return self._finish(tool_name, args, started, AuditStatus.FAILED, result)
                result = self._to_result(tool_name, context.as_of, output)
        except AuditError:
            raise
        except FleetCopilotError as exc:  # expected, e.g. database not built yet
            logger.error("Tool %s could not run: %s", tool_name, exc)
            result = self._error(tool_name, str(exc))
            return self._finish(tool_name, args, started, AuditStatus.FAILED, result)
        except Exception:  # fallback: report, don't crash the agent loop
            logger.exception("Tool %s failed", tool_name)
            result = self._error(tool_name, "The tool failed unexpectedly; no data was returned.")
            return self._finish(tool_name, args, started, AuditStatus.FAILED, result)

        return self._finish(tool_name, args, started, AuditStatus.SUCCEEDED, result)

    # ------------------------------------------------------------------ helpers
    def _to_result(self, tool_name: str, as_of: str, output: ToolOutput) -> ToolResult:
        max_rows = self._settings.tools.max_rows
        items, evidence, truncated = output.items, output.evidence, False
        if len(items) > max_rows:
            truncated = True
            items = items[:max_rows]
            kept_ids = {eid for item in items for eid in item.get("evidence_ids", [])}
            evidence = [e for e in evidence if e.snapshot_id in kept_ids]
        coverage = dict(output.coverage)
        if truncated:
            coverage["items_total"] = len(output.items)
            coverage["items_returned"] = len(items)
        return ToolResult(
            tool=tool_name,
            status=output.status,
            summary=output.summary,
            tenant_id=self._tenant.company_id,
            as_of=as_of or None,
            items=items,
            evidence=evidence,
            assumptions=output.assumptions,
            coverage=coverage,
            truncated=truncated,
        )

    def _device_not_in_scope(
        self,
        tool_name: str,
        args: dict[str, Any],
        started: float,
        repo: FleetRepository,
        device_id: str,
    ) -> ToolResult:
        owner = repo.owning_company(device_id)
        cross_tenant = owner is not None and owner != self._tenant.company_id
        # Same message either way: the model must not learn that the device exists elsewhere.
        result = ToolResult(
            tool=tool_name,
            status="not_found",
            summary=f"Device '{device_id}' was not found in {self._tenant.company_name}'s fleet.",
            tenant_id=self._tenant.company_id,
            as_of=repo.as_of(),
        )
        if cross_tenant:
            logger.warning("Cross-tenant access blocked: %s requested %s", tool_name, device_id)
        status = AuditStatus.DENIED if cross_tenant else AuditStatus.SUCCEEDED
        extra = {"reason": "cross_tenant_device"} if cross_tenant else {}
        return self._finish(tool_name, args, started, status, result, extra)

    def _error(self, tool_name: str, message: str) -> ToolResult:
        return ToolResult(
            tool=tool_name,
            status="error",
            summary=message,
            tenant_id=self._tenant.company_id,
            error=message,
        )

    def _finish(
        self,
        tool_name: str,
        args: dict[str, Any],
        started: float,
        status: AuditStatus,
        result: ToolResult,
        extra_details: dict[str, Any] | None = None,
    ) -> ToolResult:
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        details = {
            "arguments": args,
            "result_status": result.status,
            "items": len(result.items),
            "evidence_ids": result.evidence_ids,
            "truncated": result.truncated,
            "duration_ms": duration_ms,
            **(extra_details or {}),
        }
        self._audit.record(
            event_type=AUDIT_EVENT_TYPE,
            action=tool_name,
            status=status,
            actor=self._tenant.actor,
            tenant_id=self._tenant.company_id,
            details=details,
        )
        logger.info(
            "tool %s -> %s (%d items, %.1f ms)",
            tool_name,
            result.status,
            len(result.items),
            duration_ms,
            extra={"context": {"tool": tool_name, "arguments": args, "status": status.value}},
        )
        return result


def _short_validation(exc: ValidationError, limit: int = 3) -> str:
    parts = []
    for err in exc.errors()[:limit]:
        location = ".".join(str(p) for p in err["loc"]) or "arguments"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)
