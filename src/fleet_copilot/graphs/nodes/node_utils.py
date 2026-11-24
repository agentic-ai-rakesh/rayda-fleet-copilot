"""Helpers shared by graph nodes: catalogs for prompts and compact tool-result JSON."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import SystemMessage

from fleet_copilot.graphs.state import CopilotState
from fleet_copilot.prompts.system_prompt import render_system_prompt
from fleet_copilot.tools.action_tools import ACTION_REGISTRY
from fleet_copilot.tools.registry import TOOL_REGISTRY
from fleet_copilot.utils.schema_utils import inline_json_schema


def system_message(state: CopilotState, instructions: str = "") -> SystemMessage:
    """One system message: shared ground rules, then node-specific instructions."""
    tenant = state["tenant"]
    text = render_system_prompt(
        tenant["company_name"], tenant["company_id"], state.get("as_of", "")
    )
    return SystemMessage(f"{text}\n{instructions}" if instructions else text)


def tool_catalog_text() -> str:
    return "\n".join(f"- {spec.name}: {spec.description}" for spec in TOOL_REGISTRY.values())


def action_catalog_text() -> str:
    return "\n".join(f"- {spec.name}: {spec.description}" for spec in ACTION_REGISTRY.values())


def tool_definitions() -> list[dict[str, Any]]:
    """Read and insight tools as function definitions. Action tools are never included.

    The OpenAI function format is accepted by the Anthropic, OpenAI and Gemini
    LangChain integrations alike, so one definition serves every provider.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": inline_json_schema(spec.input_model),
            },
        }
        for spec in TOOL_REGISTRY.values()
    ]


def tool_results_json(state: CopilotState) -> str:
    """Compact JSON of this turn's tool results for the responder and proposer."""
    results = [
        {key: value for key, value in result.items() if value not in (None, [], {}, False)}
        for result in state.get("tool_results", [])
    ]
    return json.dumps(results, indent=1, default=str)
