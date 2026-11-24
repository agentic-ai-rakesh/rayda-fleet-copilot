"""Action proposer: suggest operational actions justified by the evidence."""

PROMPT_NAME = "action_proposer_prompt"
PROMPT_VERSION = "1.1"

ACTION_PROPOSER_PROMPT = """\
Propose operational actions for the administrator's request, based only on \
the verified answer and tool results below. Nothing runs until a human \
approves it.

Request: {question}

Verified answer:
{answer}

Tool results (JSON):
{tool_results}

Available actions and their arguments:
{action_catalog}

Rules:
- Propose an action only for a device or employee that appears in the tool \
results, and cite the snapshot ids that justify it.
- Match the action to the evidence: batteries at end of life or hardware that \
cannot be upgraded -> flag_device_for_replacement or a battery upgrade order; \
consistent memory pressure -> RAM upgrade order; low storage -> storage \
upgrade order; a failing compliance check -> open_remediation_ticket; tell an \
employee about something affecting their device -> notify_employee.
- Respect each argument's length limit; keep `spec` short (e.g. "16 GB", \
"1 TB SSD"), choosing a sensible size from the evidence (e.g. double the current RAM).
- One proposal per device and problem. Do not propose actions the evidence does \
not support; explain in no_action_reason instead.
"""


def render_action_proposer_prompt(
    question: str, answer: str, tool_results: str, action_catalog: str
) -> str:
    return ACTION_PROPOSER_PROMPT.format(
        question=question, answer=answer, tool_results=tool_results, action_catalog=action_catalog
    )
