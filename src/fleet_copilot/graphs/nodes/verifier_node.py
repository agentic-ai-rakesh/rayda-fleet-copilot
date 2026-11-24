"""Verifier node: check citations and numbers; retry, or fall back to a deterministic answer."""

from __future__ import annotations

import logging

from fleet_copilot.graphs.dependencies import GraphDependencies
from fleet_copilot.graphs.state import CopilotState
from fleet_copilot.guardrails.answer_verifier import verify_answer
from fleet_copilot.prompts.output_schemas import GroundedAnswer

logger = logging.getLogger(__name__)


def make_verifier_node(deps: GraphDependencies):
    max_attempts = deps.settings.agent.max_answer_retries + 1

    def verifier(state: CopilotState) -> dict:
        draft = GroundedAnswer.model_validate(state["draft"])
        result = verify_answer(draft, state.get("tool_results", []))
        verification = {"passed": result.passed, "issues": result.issues}

        if result.passed:
            answer = draft.answer
            if draft.caveats:
                answer += "\n\n" + "\n".join(f"_Note: {c}_" for c in draft.caveats)
            return {"verification": verification, "answer": answer, "answer_status": "verified"}

        logger.warning(
            "Answer failed verification (attempt %d/%d): %s",
            state.get("answer_attempts", 1),
            max_attempts,
            result.issues,
        )
        if state.get("answer_attempts", 1) < max_attempts:
            return {"verification": verification}  # routed back to the responder

        return {
            "verification": verification,
            "answer": fallback_answer(state),
            "answer_status": "unverified_fallback",
        }

    return verifier


def fallback_answer(state: CopilotState) -> str:
    """Deterministic answer built only from tool summaries and their evidence ids."""
    lines = [
        "I could not produce a narrative answer that passed citation checks, "
        "so here are the verified tool results:"
    ]
    for result in state.get("tool_results", []):
        if result.get("status") != "ok":
            continue
        ids = [e["snapshot_id"] for e in result.get("evidence", [])][:5]
        cites = f" [{', '.join(ids)}]" if ids else ""
        lines.append(f"- {result['summary']}{cites}")
        for item in result.get("items", [])[:10]:
            if "finding" in item:
                lines.append(f"  - {item['finding']} [{', '.join(item['evidence_ids'])}]")
    if len(lines) == 1:
        lines = ["I could not find data to answer this question."]
    return "\n".join(lines)
