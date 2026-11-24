"""Responder node: draft a grounded, cited answer from the tool results."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from fleet_copilot.exceptions import LLMError
from fleet_copilot.graphs.dependencies import GraphDependencies
from fleet_copilot.graphs.nodes.node_utils import system_message, tool_results_json
from fleet_copilot.graphs.state import CopilotState
from fleet_copilot.prompts import responder_prompt
from fleet_copilot.prompts.output_schemas import GroundedAnswer

logger = logging.getLogger(__name__)


def make_responder_node(deps: GraphDependencies):
    def responder(state: CopilotState) -> dict:
        issues = (state.get("verification") or {}).get("issues", [])
        prompt = responder_prompt.render_responder_prompt(
            state["question"], tool_results_json(state), issues
        )
        attempts = state.get("answer_attempts", 0) + 1
        try:
            draft = deps.llm.structured(
                [system_message(state), HumanMessage(prompt)], GroundedAnswer, purpose="responder"
            )
        except LLMError as exc:
            logger.error("Responder failed: %s", exc)
            # Empty draft -> verifier fails it -> deterministic fallback answer.
            draft = GroundedAnswer(can_answer=True, answer="", claims=[])
        return {"draft": draft.model_dump(), "answer_attempts": attempts}

    return responder
