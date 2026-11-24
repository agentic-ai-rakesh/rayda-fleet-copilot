"""Responder: write the grounded answer from tool results only."""

PROMPT_NAME = "responder_prompt"
PROMPT_VERSION = "1.3"

RESPONDER_PROMPT = """\
Write the answer to the administrator's request using only the tool results below.

Request: {question}

Tool results (JSON):
{tool_results}

Requirements:
- Lead with the direct answer (a count, a list of devices, or "none").
- After each fact, cite the snapshot id(s) it comes from in square brackets, \
e.g. "MT7PJB7N5LRE has 2.0% disk free [MT7PJB7N5LRE@20260612T0839Z]". \
Only cite ids that appear in the tool results' evidence.
- Copy numbers exactly as the tools give them; do not compute new ones.
- For insights, give the finding, the supporting evidence and a one-line explanation.
- State the thresholds or assumptions the tools used when they matter, in \
`caveats` (not in `claims`: claims are facts about devices and need evidence ids).
- An empty result is an answer: say there are none, how many devices were \
checked, and cite the checked snapshots from the evidence.
- If the results do not answer the request, set can_answer to false and say \
what is missing.
- If the request asks for actions (flag, order, notify, open a ticket), answer \
with the evidence for or against them only. Actions are proposed and approved \
in a separate step after your answer: do not say that you cannot perform them \
or that no tool for them exists.
- List every factual statement about devices or counts in `claims`, each with \
at least one evidence id.
{retry_feedback}"""

RETRY_FEEDBACK = """
Your previous draft failed verification. Fix these problems:
{issues}
"""


def render_responder_prompt(question: str, tool_results: str, issues: list[str]) -> str:
    feedback = RETRY_FEEDBACK.format(issues="\n".join(f"- {i}" for i in issues)) if issues else ""
    return RESPONDER_PROMPT.format(
        question=question, tool_results=tool_results, retry_feedback=feedback
    )
