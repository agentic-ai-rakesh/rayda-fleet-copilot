"""The Fleet Copilot graph.

START
  │
input_guard ──(blocked)──────────────────────────────────────────┐
  │                                                              │
planner ──(out of scope / model error)───────────────────────────┤
  │                                                              │
executor ◄──────┐   explicit tool loop, at most                  │
  │   (tool calls)  agent.max_tool_iterations rounds             │
  ├──► tools ───┘                                                │
  │                                                              │
responder ◄─────┐                                                │
  │             │ (citations failed, retries left)               │
verifier ───────┘                                                │
  │  (verified and the user asked for actions)                   │
  ├──► action_proposer ──► evidence_gate ──(none ready)──────────┤
  │                             │                                │
  │                      human_approval  (interrupt: waits)      │
  │                             │                                │
  │                      execute_actions                         │
  │                             │                                │
  └─────────────────────────► finalize ◄─────────────────────────┘
                                │
                               END
"""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fleet_copilot.graphs import routing as r
from fleet_copilot.graphs.dependencies import GraphDependencies
from fleet_copilot.graphs.nodes.action_execution_node import make_action_execution_node
from fleet_copilot.graphs.nodes.action_proposer_node import make_action_proposer_node
from fleet_copilot.graphs.nodes.evidence_gate_node import make_evidence_gate_node
from fleet_copilot.graphs.nodes.executor_node import make_executor_node
from fleet_copilot.graphs.nodes.finalize_node import make_finalize_node
from fleet_copilot.graphs.nodes.human_approval_node import make_human_approval_node
from fleet_copilot.graphs.nodes.input_guard_node import make_input_guard_node
from fleet_copilot.graphs.nodes.planner_node import make_planner_node
from fleet_copilot.graphs.nodes.responder_node import make_responder_node
from fleet_copilot.graphs.nodes.tool_node import make_tool_node
from fleet_copilot.graphs.nodes.verifier_node import make_verifier_node
from fleet_copilot.graphs.state import CopilotState


def build_fleet_copilot_graph(
    deps: GraphDependencies, checkpointer: BaseCheckpointSaver
) -> CompiledStateGraph:
    graph = StateGraph(CopilotState)

    graph.add_node(r.INPUT_GUARD, make_input_guard_node(deps))
    graph.add_node(r.PLANNER, make_planner_node(deps))
    graph.add_node(r.EXECUTOR, make_executor_node(deps))
    graph.add_node(r.TOOLS, make_tool_node(deps))
    graph.add_node(r.RESPONDER, make_responder_node(deps))
    graph.add_node(r.VERIFIER, make_verifier_node(deps))
    graph.add_node(r.ACTION_PROPOSER, make_action_proposer_node(deps))
    graph.add_node(r.EVIDENCE_GATE, make_evidence_gate_node(deps))
    graph.add_node(r.HUMAN_APPROVAL, make_human_approval_node(deps))
    graph.add_node(r.EXECUTE_ACTIONS, make_action_execution_node(deps))
    graph.add_node(r.FINALIZE, make_finalize_node(deps))

    graph.add_edge(START, r.INPUT_GUARD)
    graph.add_conditional_edges(r.INPUT_GUARD, r.after_input_guard, [r.PLANNER, r.FINALIZE])
    graph.add_conditional_edges(r.PLANNER, r.after_planner, [r.EXECUTOR, r.FINALIZE])
    graph.add_conditional_edges(
        r.EXECUTOR,
        r.make_after_executor(deps.settings.agent),
        [r.TOOLS, r.RESPONDER, r.FINALIZE],
    )
    graph.add_edge(r.TOOLS, r.EXECUTOR)
    graph.add_edge(r.RESPONDER, r.VERIFIER)
    graph.add_conditional_edges(
        r.VERIFIER, r.after_verifier, [r.RESPONDER, r.ACTION_PROPOSER, r.FINALIZE]
    )
    graph.add_conditional_edges(
        r.ACTION_PROPOSER, r.after_action_proposer, [r.EVIDENCE_GATE, r.FINALIZE]
    )
    graph.add_conditional_edges(
        r.EVIDENCE_GATE, r.after_evidence_gate, [r.HUMAN_APPROVAL, r.FINALIZE]
    )
    graph.add_edge(r.HUMAN_APPROVAL, r.EXECUTE_ACTIONS)
    graph.add_edge(r.EXECUTE_ACTIONS, r.FINALIZE)
    graph.add_edge(r.FINALIZE, END)

    return graph.compile(checkpointer=checkpointer)
