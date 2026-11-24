"""Human approval node: pause the graph until an administrator decides.

Uses LangGraph `interrupt()`. The graph state is checkpointed, so the
decision can arrive later or from another process. Code before `interrupt()`
re-runs on resume, so nothing with side effects happens before it.
"""

from __future__ import annotations

from langgraph.types import interrupt

from fleet_copilot.audit import AuditStatus
from fleet_copilot.graphs.dependencies import GraphDependencies
from fleet_copilot.graphs.state import CopilotState
from fleet_copilot.guardrails.approval_policy import apply_decisions, parse_resume_payload

APPROVAL_REQUEST = "approval_request"


def make_human_approval_node(deps: GraphDependencies):
    def human_approval(state: CopilotState) -> dict:
        proposals = state.get("proposals", [])
        pending = [p for p in proposals if p["gate"] == "ready"]

        payload = interrupt({"type": APPROVAL_REQUEST, "proposals": pending})

        decisions, approver = parse_resume_payload(payload)
        decided = {p["proposal_id"]: p for p in apply_decisions(pending, decisions, approver)}
        for proposal in decided.values():
            deps.audit.record(
                event_type="action_approval",
                action=proposal["action"],
                status=AuditStatus.APPROVED
                if proposal["decision"] == "approved"
                else AuditStatus.REJECTED,
                actor=approver,
                tenant_id=deps.tenant.company_id,
                details={
                    "proposal_id": proposal["proposal_id"],
                    "arguments": proposal["arguments"],
                    "comment": proposal["decision_comment"],
                },
            )
        return {"proposals": [decided.get(p["proposal_id"], p) for p in proposals]}

    return human_approval
