"""Human approval policy for state-changing actions.

- Every action that passed the evidence gate needs an explicit decision.
- Default deny: a proposal with no decision is rejected.
- Decisions for unknown proposal ids are ignored (and logged).
- The approver is recorded on each decision and in the audit log.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApprovalDecision:
    proposal_id: str
    approved: bool
    comment: str = ""


def apply_decisions(
    pending: list[dict[str, Any]], decisions: list[ApprovalDecision], approver: str
) -> list[dict[str, Any]]:
    """Return the pending proposals with `decision` and `decided_by` filled in."""
    by_id = {d.proposal_id: d for d in decisions}
    known = {p["proposal_id"] for p in pending}
    for unknown in set(by_id) - known:
        logger.warning("Ignoring decision for unknown proposal %s", unknown)

    decided = []
    for proposal in pending:
        decision = by_id.get(proposal["proposal_id"])
        approved = bool(decision and decision.approved)
        decided.append(
            {
                **proposal,
                "decision": "approved" if approved else "rejected",
                "decided_by": approver,
                "decision_comment": decision.comment
                if decision
                else "No decision given (default deny).",
            }
        )
    return decided


def parse_resume_payload(payload: Any) -> tuple[list[ApprovalDecision], str]:
    """Read the value passed to resume the graph after an approval interrupt."""
    if not isinstance(payload, dict):
        return [], "unknown"
    approver = str(payload.get("approver") or "unknown")
    decisions = []
    for item in payload.get("decisions", []):
        if isinstance(item, dict) and "proposal_id" in item:
            decisions.append(
                ApprovalDecision(
                    proposal_id=str(item["proposal_id"]),
                    approved=item.get("approved") is True,
                    comment=str(item.get("comment", "")),
                )
            )
    return decisions, approver
