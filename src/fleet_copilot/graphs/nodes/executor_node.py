"""Executor node: one model step of the tool loop (may request tool calls)."""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage

from fleet_copilot.exceptions import LLMError
from fleet_copilot.graphs.dependencies import GraphDependencies
from fleet_copilot.graphs.nodes.node_utils import system_message, tool_definitions
from fleet_copilot.graphs.state import CopilotState
from fleet_copilot.prompts import executor_prompt

logger = logging.getLogger(__name__)


def make_executor_node(deps: GraphDependencies):
    tools = tool_definitions()

    def executor(state: CopilotState) -> dict:
        new_messages = []
        if not state.get("messages"):
            plan_text = json.dumps(state["plan"], indent=1)
            instructions = executor_prompt.render_executor_prompt(plan_text)
            new_messages = [system_message(state, instructions), HumanMessage(state["question"])]
        history = list(state.get("messages", [])) + new_messages
        try:
            response = deps.llm.call_tools(history, tools, purpose="executor")
        except LLMError as exc:
            logger.error("Executor failed: %s", exc)
            if state.get("tool_results"):
                # Fallback: answer from the evidence already gathered.
                return {"messages": new_messages, "tool_loop_done": True}
            return {"error": "I could not gather data because the language model failed."}
        return {"messages": [*new_messages, response]}

    return executor
