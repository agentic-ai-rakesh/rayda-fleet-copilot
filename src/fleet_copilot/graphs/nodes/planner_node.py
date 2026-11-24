"""Planner node: classify intent and choose the tool calls (structured output)."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from fleet_copilot.exceptions import LLMError
from fleet_copilot.graphs.dependencies import GraphDependencies
from fleet_copilot.graphs.nodes.node_utils import (
    action_catalog_text,
    system_message,
    tool_catalog_text,
)
from fleet_copilot.graphs.state import CopilotState
from fleet_copilot.prompts import planner_prompt
from fleet_copilot.prompts.output_schemas import Plan

logger = logging.getLogger(__name__)


def make_planner_node(deps: GraphDependencies):
    def planner(state: CopilotState) -> dict:
        prompt = planner_prompt.render_planner_prompt(
            state["question"], tool_catalog_text(), action_catalog_text()
        )
        try:
            plan = deps.llm.structured(
                [system_message(state), HumanMessage(prompt)], Plan, purpose="planner"
            )
        except LLMError as exc:
            logger.error("Planner failed: %s", exc)
            return {"error": "I could not plan this request because the language model failed."}

        logger.info(
            "plan: intent=%s steps=%s wants_actions=%s",
            plan.intent,
            [s.tool for s in plan.steps],
            plan.wants_actions,
            extra={
                "context": {
                    "prompt": planner_prompt.PROMPT_NAME,
                    "prompt_version": planner_prompt.PROMPT_VERSION,
                }
            },
        )
        return {"plan": plan.model_dump()}

    return planner
