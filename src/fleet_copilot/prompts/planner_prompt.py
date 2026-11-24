"""Planner: classify the request and plan which tools to call."""

PROMPT_NAME = "planner_prompt"
PROMPT_VERSION = "1.0"

PLANNER_PROMPT = """\
Plan how to answer the administrator's request.

Available tools:
{tool_catalog}

Actions that can be proposed later (they need human approval and are never \
run during planning):
{action_catalog}

Decide:
1. intent:
   - "question": facts about the current state of devices.
   - "insight": patterns or trends over time (batteries wearing out, devices \
consistently short of RAM or storage, compliance drift).
   - "action": the user asks what to do, or asks to take or propose actions.
   - "out_of_scope": anything not about this company's device telemetry \
(general chat, other companies, writing code, changing settings).
2. steps: the smallest set of tool calls that gathers the evidence. Prefer \
detect_* tools for patterns over time and find_*/get_* tools for current state. \
For action requests, include the detect_* or get_* calls whose results would \
justify each action.
3. wants_actions: true only if the user asks to act, fix, order, notify, flag \
or recommend actions.

Request: {question}
"""


def render_planner_prompt(question: str, tool_catalog: str, action_catalog: str) -> str:
    return PLANNER_PROMPT.format(
        question=question, tool_catalog=tool_catalog, action_catalog=action_catalog
    )
