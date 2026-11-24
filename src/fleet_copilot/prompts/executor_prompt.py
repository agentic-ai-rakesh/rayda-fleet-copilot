"""Executor: run the plan with tools, one round at a time."""

PROMPT_NAME = "executor_prompt"
PROMPT_VERSION = "1.0"

EXECUTOR_PROMPT = """\
Gather the evidence needed to answer the request by calling tools.

Plan:
{plan}

Rules:
- Follow the plan, but adapt if a result shows another call is needed (for \
example, look up details of a device a detector flagged).
- Call independent tools in parallel.
- If a tool returns an error, fix the arguments once; do not repeat the same call.
- Stop calling tools when you have enough evidence, and reply with one short \
sentence saying the evidence is gathered. Do not write the final answer here.
"""


def render_executor_prompt(plan: str) -> str:
    return EXECUTOR_PROMPT.format(plan=plan)
