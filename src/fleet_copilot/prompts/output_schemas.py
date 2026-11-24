"""Structured outputs the prompts ask the model to return."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field

Intent = Literal["question", "insight", "action", "out_of_scope"]
ActionName = Literal[
    "create_upgrade_order",
    "open_remediation_ticket",
    "flag_device_for_replacement",
    "notify_employee",
]


class PlanStep(BaseModel):
    tool: str = Field(description="Name of a tool from the catalog.")
    purpose: str = Field(description="What this call establishes, in a few words.")


class Plan(BaseModel):
    intent: Intent
    restated_question: str = Field(description="The question in one clear sentence.")
    steps: list[PlanStep] = Field(description="Tool calls to gather evidence, in order.")
    wants_actions: bool = Field(
        description="True only if the user asks to take, propose or recommend actions."
    )
    out_of_scope_reason: str | None = Field(
        default=None, description="Set when intent is out_of_scope."
    )


class Claim(BaseModel):
    statement: str = Field(description="One factual statement made in the answer.")
    evidence_ids: list[str] = Field(description="Snapshot ids that support the statement.")


class GroundedAnswer(BaseModel):
    can_answer: bool = Field(description="False if the tool results cannot answer the question.")
    answer: str = Field(description="Answer in Markdown. Cite snapshot ids in square brackets.")
    claims: list[Claim] = Field(description="Every factual statement in the answer, with sources.")
    caveats: list[str] = Field(default_factory=list, description="Limits of the answer.")


class ProposedAction(BaseModel):
    """Flat argument fields (not a free-form dict) so strict JSON-schema output works.

    Each action uses only its own fields; the evidence gate validates them
    against the action's real input schema.
    """

    action: ActionName
    device_id: str | None = Field(default=None, description="Target device.")
    employee_id: str | None = Field(default=None, description="notify_employee only.")
    component: Literal["ram", "storage", "battery"] | None = Field(
        default=None, description="create_upgrade_order only."
    )
    spec: str | None = Field(
        default=None, description="create_upgrade_order: short, max 64 chars, e.g. '16 GB'."
    )
    check_id: str | None = Field(default=None, description="open_remediation_ticket only.")
    note: str | None = Field(default=None, description="open_remediation_ticket only.")
    reason: str | None = Field(default=None, description="flag_device_for_replacement only.")
    message: str | None = Field(default=None, description="notify_employee only.")
    justification: str = Field(description="Why this action, referring to the evidence.")
    evidence_ids: list[str] = Field(description="Snapshot ids that justify the action.")

    def arguments(self, names: Iterable[str]) -> dict[str, Any]:
        """The values of the given argument fields."""
        return {name: getattr(self, name) for name in names}


class ActionPlan(BaseModel):
    proposals: list[ProposedAction] = Field(default_factory=list)
    no_action_reason: str | None = Field(
        default=None, description="Why no action is proposed, if none."
    )
