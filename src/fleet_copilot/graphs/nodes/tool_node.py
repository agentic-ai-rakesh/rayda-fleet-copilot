"""Tool node: run the tool calls the executor requested, through the tenant-scoped ToolExecutor."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from fleet_copilot.graphs.dependencies import GraphDependencies
from fleet_copilot.graphs.state import CopilotState


def make_tool_node(deps: GraphDependencies):
    def run_tools(state: CopilotState) -> dict:
        last = state["messages"][-1]
        calls = last.tool_calls if isinstance(last, AIMessage) else []
        messages, results = [], []
        for call in calls:
            # Only read/insight tools exist in the executor's registry: an action
            # name here comes back as "unknown tool", so the loop cannot act.
            result = deps.tools.execute(call["name"], call.get("args") or {})
            results.append(result.model_dump())
            messages.append(
                ToolMessage(
                    content=json.dumps(result.to_model_payload(), default=str),
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error" if result.status == "error" else "success",
                )
            )
        return {
            "messages": messages,
            "tool_results": results,
            "tool_iterations": state.get("tool_iterations", 0) + 1,
        }

    return run_tools
