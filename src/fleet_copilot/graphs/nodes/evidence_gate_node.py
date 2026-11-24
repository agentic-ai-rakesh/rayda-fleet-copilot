"""Evidence gate node: block proposals the evidence does not support; audit every decision."""

from __future__ import annotations

from contextlib import closing

from fleet_copilot.audit import AuditStatus
from fleet_copilot.db.connection import connect
from fleet_copilot.db.fleet_repository import FleetRepository
from fleet_copilot.graphs.dependencies import GraphDependencies
from fleet_copilot.graphs.state import CopilotState
from fleet_copilot.guardrails.evidence_gate import EvidenceGate, TurnEvidence
from fleet_copilot.prompts.output_schemas import ProposedAction
from fleet_copilot.utils.hash_utils import canonical_json, sha256_text


def make_evidence_gate_node(deps: GraphDependencies):
    def evidence_gate(state: CopilotState) -> dict:
        raw = [ProposedAction.model_validate(p) for p in state.get("proposals", [])]
        turn = TurnEvidence.from_tool_results(state.get("tool_results", []))
        with closing(connect(deps.settings.paths.fleet_db, read_only=True)) as conn:
            gate = EvidenceGate(
                FleetRepository(conn, deps.tenant), deps.settings.thresholds, deps.settings.agent
            )
            decisions = gate.review(raw, turn)

        proposals = []
        for proposal, decision in zip(raw, decisions, strict=True):
            record = {
                "proposal_id": "prop-"
                + sha256_text(
                    canonical_json([proposal.action, decision.arguments, state["question"]])
                )[:12],
                "action": proposal.action,
                "arguments": decision.arguments,
                "justification": proposal.justification,
                "evidence_ids": proposal.evidence_ids,
                "gate": "ready" if decision.allowed else "blocked",
                "gate_reasons": decision.reasons,
            }
            proposals.append(record)
            deps.audit.record(
                event_type="action_proposal",
                action=proposal.action,
                status=AuditStatus.PROPOSED if decision.allowed else AuditStatus.DENIED,
                actor=deps.tenant.actor,
                tenant_id=deps.tenant.company_id,
                details={
                    k: record[k]
                    for k in ("proposal_id", "arguments", "evidence_ids", "gate_reasons")
                },
            )
        return {"proposals": proposals}

    return evidence_gate
