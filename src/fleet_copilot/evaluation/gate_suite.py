"""Gate suite (no LLM): red-team action proposals against the evidence gate."""

from __future__ import annotations

import re
from contextlib import closing
from typing import Any

from fleet_copilot.db.connection import connect
from fleet_copilot.db.fleet_repository import FleetRepository
from fleet_copilot.evaluation.cases import GateCase
from fleet_copilot.evaluation.eval_context import EvalContext
from fleet_copilot.evaluation.metrics import CaseResult
from fleet_copilot.guardrails.evidence_gate import EvidenceGate, TurnEvidence
from fleet_copilot.prompts.output_schemas import ProposedAction

_PLACEHOLDER = re.compile(r"\{(latest|employee|repeat):([^}]+)\}")


def run_gate_case(case: GateCase, ctx: EvalContext) -> CaseResult:
    result = CaseResult(case.id, "gate", "gate_redteam", case.tenant)
    executor = ctx.tool_executor(case.tenant)
    tool_results = [
        executor.execute(spec.name, spec.args).model_dump() for spec in case.evidence_tools
    ]
    result.check(
        "evidence_tools_ok",
        all(r["status"] == "ok" for r in tool_results),
        "; ".join(r["summary"] for r in tool_results if r["status"] != "ok"),
    )

    proposal = ProposedAction.model_validate(_resolve(case.proposal, ctx))
    with closing(connect(ctx.settings.paths.fleet_db, read_only=True)) as conn:
        gate = EvidenceGate(
            FleetRepository(conn, executor.tenant), ctx.settings.thresholds, ctx.settings.agent
        )
        (decision,) = gate.review([proposal], TurnEvidence.from_tool_results(tool_results))

    result.metrics["gate_reasons"] = decision.reasons
    expected = "allowed" if case.expect_allowed else "blocked"
    actual = "allowed" if decision.allowed else "blocked"
    result.check(
        "gate_decision",
        decision.allowed == case.expect_allowed,
        f"expected {expected}, got {actual}: {decision.reasons}",
    )
    return result


def _resolve(value: Any, ctx: EvalContext) -> Any:
    if isinstance(value, dict):
        return {k: _resolve(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, ctx) for v in value]
    if isinstance(value, str):
        return _PLACEHOLDER.sub(lambda m: _substitute(m.group(1), m.group(2), ctx), value)
    return value


def _substitute(kind: str, argument: str, ctx: EvalContext) -> str:
    if kind == "latest":
        return ctx.latest_snapshot_id(argument)
    if kind == "employee":
        return ctx.employee_of(argument)
    text, _, count = argument.partition(":")
    return text * int(count)
