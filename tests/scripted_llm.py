"""A deterministic stand-in for the LLM, scripted per call purpose.

Each purpose ("planner", "executor", "responder", "action_proposer") has a
queue of responses. An item may be a value, a callable taking the messages
(to build a response from what the model would have seen), or an exception
to raise. The last item repeats; calling a purpose with no script fails the test.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from fleet_copilot.prompts.output_schemas import Claim, GroundedAnswer, Plan, PlanStep

SNAPSHOT_ID = re.compile(r"[A-Za-z0-9_-]+@\d{8}T\d{4}Z")


class ScriptedLLM:
    def __init__(self, **scripts: list[Any]) -> None:
        self._scripts = {purpose: list(items) for purpose, items in scripts.items()}
        self.calls: list[str] = []

    def structured(self, messages: Sequence[BaseMessage], schema: type, *, purpose: str) -> Any:
        return self._next(purpose, messages)

    def call_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[dict], *, purpose: str
    ) -> AIMessage:
        return self._next(purpose, messages)

    def _next(self, purpose: str, messages: Sequence[BaseMessage]) -> Any:
        self.calls.append(purpose)
        queue = self._scripts.get(purpose)
        if not queue:
            raise AssertionError(f"Unexpected LLM call for '{purpose}'")
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, Exception):
            raise item
        result = item(messages) if callable(item) else item
        if isinstance(result, BaseMessage):
            # A real model returns a new message each time; reusing one object would
            # make LangGraph's message reducer treat it as the same message.
            result = result.model_copy(deep=True, update={"id": None})
        return result


# ---------------------------------------------------------------- builders
def plan(intent: str = "question", tools: Sequence[str] = (), wants_actions: bool = False) -> Plan:
    return Plan(
        intent=intent,
        restated_question="test",
        steps=[PlanStep(tool=t, purpose="test") for t in tools],
        wants_actions=wants_actions,
        out_of_scope_reason="not about devices" if intent == "out_of_scope" else None,
    )


def tool_call(name: str, args: dict | None = None, call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id}])


DONE = AIMessage(content="Evidence gathered.")


def answer_citing(device_ids: Sequence[str], text: str = "is affected") -> Callable:
    """Responder script: cite the latest evidence id seen for each device in the prompt."""

    def build(messages: Sequence[BaseMessage]) -> GroundedAnswer:
        prompt = str(messages[-1].content)
        claims = []
        for device_id in device_ids:
            ids = sorted(i for i in SNAPSHOT_ID.findall(prompt) if i.startswith(device_id + "@"))
            claims.append(Claim(statement=f"{device_id} {text}", evidence_ids=ids[-1:]))
        answer = "\n".join(
            f"- {c.statement} [{c.evidence_ids[0]}]" for c in claims if c.evidence_ids
        )
        return GroundedAnswer(can_answer=True, answer=answer, claims=claims)

    return build


def bad_answer(evidence_id: str = "FAKE@20990101T0000Z") -> GroundedAnswer:
    return GroundedAnswer(
        can_answer=True,
        answer=f"Something [{evidence_id}]",
        claims=[Claim(statement="Something", evidence_ids=[evidence_id])],
    )
