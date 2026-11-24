"""Conditional edges: where the graph goes after each decision point."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from fleet_copilot.config.settings import AgentSettings
from fleet_copilot.graphs.state import CopilotState

# Node names
INPUT_GUARD = "input_guard"
PLANNER = "planner"
EXECUTOR = "executor"
TOOLS = "tools"
RESPONDER = "responder"
VERIFIER = "verifier"
ACTION_PROPOSER = "action_proposer"
EVIDENCE_GATE = "evidence_gate"
HUMAN_APPROVAL = "human_approval"
EXECUTE_ACTIONS = "execute_actions"
FINALIZE = "finalize"


def after_input_guard(state: CopilotState) -> str:
    return PLANNER if state["guard"]["allowed"] else FINALIZE


def after_planner(state: CopilotState) -> str:
    if state.get("error") or state["plan"]["intent"] == "out_of_scope":
        return FINALIZE
    return EXECUTOR


def make_after_executor(settings: AgentSettings):
    def after_executor(state: CopilotState) -> str:
        if state.get("error"):
            return FINALIZE
        if state.get("tool_loop_done"):
            return RESPONDER
        last = state["messages"][-1]
        wants_tools = isinstance(last, AIMessage) and bool(last.tool_calls)
        if wants_tools and state.get("tool_iterations", 0) < settings.max_tool_iterations:
            return TOOLS
        return RESPONDER

    return after_executor


def after_verifier(state: CopilotState) -> str:
    if not state["verification"]["passed"] and state.get("answer_status") is None:
        return RESPONDER  # retry with the verifier's feedback
    plan = state["plan"]
    if state.get("answer_status") == "verified" and plan.get("wants_actions"):
        return ACTION_PROPOSER
    return FINALIZE


def after_action_proposer(state: CopilotState) -> str:
    return EVIDENCE_GATE if state.get("proposals") else FINALIZE


def after_evidence_gate(state: CopilotState) -> str:
    ready = [p for p in state.get("proposals", []) if p["gate"] == "ready"]
    return HUMAN_APPROVAL if ready else FINALIZE
