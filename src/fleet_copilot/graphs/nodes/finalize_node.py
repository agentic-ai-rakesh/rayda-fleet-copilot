"""Finalize node: compose the reply and write one audit event for the whole turn."""

from __future__ import annotations

from fleet_copilot.audit import AuditStatus
from fleet_copilot.graphs.dependencies import GraphDependencies
from fleet_copilot.graphs.state import CopilotState

NO_ACTION_NOTE = "No actions were taken."


def make_finalize_node(deps: GraphDependencies):
    def finalize(state: CopilotState) -> dict:
        response, status = _compose(state)
        proposals = state.get("proposals", [])
        deps.audit.record(
            event_type="agent_turn",
            action="answer",
            status=AuditStatus.FAILED if status == "error" else AuditStatus.SUCCEEDED,
            actor=deps.tenant.actor,
            tenant_id=deps.tenant.company_id,
            details={
                "question": state["question"][:500],
                "intent": (state.get("plan") or {}).get("intent"),
                "answer_status": status,
                "tools": [r["tool"] for r in state.get("tool_results", [])],
                "evidence_ids": sorted(
                    {e["snapshot_id"] for r in state.get("tool_results", []) for e in r["evidence"]}
                ),
                "verification": state.get("verification"),
                "proposals": [
                    {k: p.get(k) for k in ("proposal_id", "action", "gate", "decision")}
                    for p in proposals
                ],
            },
        )
        return {"final_response": response, "answer_status": status}

    return finalize


def _compose(state: CopilotState) -> tuple[str, str]:
    guard = state.get("guard") or {}
    if guard and not guard.get("allowed", True):
        return guard["message"], "refused"
    if state.get("error"):
        return f"Sorry, {state['error']} {NO_ACTION_NOTE}", "error"

    plan = state.get("plan") or {}
    if plan.get("intent") == "out_of_scope":
        reason = plan.get("out_of_scope_reason") or "it is not about your fleet's telemetry"
        return (
            "I can only help with your company's device health, compliance and related "
            f"actions; this request is out of scope ({reason}).",
            "out_of_scope",
        )

    parts = [state.get("answer", "")]
    proposals = state.get("proposals", [])
    if proposals:
        parts.append("\n**Proposed actions**")
        parts.extend(_describe(p) for p in proposals)
    elif plan.get("wants_actions") and state.get("answer_status") == "verified":
        reason = state.get("no_action_reason") or "the evidence does not justify any action"
        parts.append(f"\nNo actions proposed: {reason}.")
    elif plan.get("wants_actions"):
        parts.append("\nNo actions proposed because the answer could not be fully verified.")
    return "\n".join(parts).strip(), state.get("answer_status", "verified")


def _describe(proposal: dict) -> str:
    args = ", ".join(f"{k}={v}" for k, v in proposal["arguments"].items())
    head = f"- `{proposal['action']}`({args})"
    if proposal["gate"] == "blocked":
        return f"{head}: **refused**: " + " ".join(proposal["gate_reasons"])
    decision = proposal.get("decision")
    if decision is None:
        return f"{head}: awaiting approval"
    if decision == "rejected":
        return f"{head}: rejected by {proposal['decided_by']}"
    execution = proposal.get("execution") or {}
    return f"{head}: approved by {proposal['decided_by']}, {execution.get('status')}: " + str(
        execution.get("message", "")
    )
