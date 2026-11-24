"""Input guard node: deterministic checks before any model call."""

from __future__ import annotations

from fleet_copilot.audit import AuditStatus
from fleet_copilot.graphs.dependencies import GraphDependencies
from fleet_copilot.graphs.state import CopilotState
from fleet_copilot.guardrails.input_guard import check_input


def make_input_guard_node(deps: GraphDependencies):
    def input_guard(state: CopilotState) -> dict:
        decision = check_input(
            state["question"],
            max_chars=deps.settings.agent.max_question_chars,
            other_companies=deps.other_companies,
        )
        if not decision.allowed:
            deps.audit.record(
                event_type="input_guard",
                action=decision.category,
                status=AuditStatus.DENIED,
                actor=deps.tenant.actor,
                tenant_id=deps.tenant.company_id,
                details={"question": state["question"][:500]},
            )
        return {
            "guard": {
                "allowed": decision.allowed,
                "category": decision.category,
                "message": decision.message,
            }
        }

    return input_guard
