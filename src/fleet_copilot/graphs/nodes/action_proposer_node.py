"""Action proposer node: ask the model for actions justified by the verified answer."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from fleet_copilot.exceptions import LLMError
from fleet_copilot.graphs.dependencies import GraphDependencies
from fleet_copilot.graphs.nodes.node_utils import (
    action_catalog_text,
    system_message,
    tool_results_json,
)
from fleet_copilot.graphs.state import CopilotState
from fleet_copilot.prompts import action_proposer_prompt
from fleet_copilot.prompts.output_schemas import ActionPlan

logger = logging.getLogger(__name__)


def make_action_proposer_node(deps: GraphDependencies):
    def action_proposer(state: CopilotState) -> dict:
        prompt = action_proposer_prompt.render_action_proposer_prompt(
            state["question"], state["answer"], tool_results_json(state), action_catalog_text()
        )
        try:
            plan = deps.llm.structured(
                [system_message(state), HumanMessage(prompt)], ActionPlan, purpose="action_proposer"
            )
        except LLMError as exc:
            logger.error("Action proposer failed: %s", exc)
            return {
                "proposals": [],
                "no_action_reason": "Actions could not be generated right now.",
            }
        return {
            "proposals": [p.model_dump() for p in plan.proposals],
            "no_action_reason": plan.no_action_reason,
        }

    return action_proposer
