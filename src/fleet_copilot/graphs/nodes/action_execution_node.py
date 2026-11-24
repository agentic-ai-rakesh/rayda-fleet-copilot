"""Action execution node: run only the proposals a human approved."""

from __future__ import annotations

from dataclasses import asdict

from fleet_copilot.graphs.dependencies import GraphDependencies
from fleet_copilot.graphs.state import CopilotState


def make_action_execution_node(deps: GraphDependencies):
    def execute_actions(state: CopilotState) -> dict:
        updated = []
        for proposal in state.get("proposals", []):
            if proposal.get("decision") == "approved" and "execution" not in proposal:
                result = deps.actions.execute(proposal, approved_by=proposal["decided_by"])
                proposal = {**proposal, "execution": asdict(result)}
            updated.append(proposal)
        return {"proposals": updated}

    return execute_actions
