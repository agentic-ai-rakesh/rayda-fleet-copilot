"""The agent's state for one request (one graph thread).

Values are plain dicts and LangChain messages so the SQLite checkpointer can
store them; that is what lets an approval resume in a later process.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

AnswerStatus = Literal["verified", "unverified_fallback", "refused", "out_of_scope", "error"]


class CopilotState(TypedDict, total=False):
    # Input
    question: str
    tenant: dict[str, str]  # company_id, company_name, actor
    as_of: str

    # Input guard
    guard: dict[str, Any]  # allowed, category, message

    # Planning
    plan: dict[str, Any]  # output_schemas.Plan

    # Tool loop (explicit: executor -> tools -> executor ...)
    messages: Annotated[list[BaseMessage], add_messages]
    tool_iterations: int
    tool_loop_done: bool  # set when the loop must stop early (e.g. model failure mid-loop)
    tool_results: Annotated[list[dict[str, Any]], operator.add]  # ToolResult dumps

    # Answer drafting and verification
    draft: dict[str, Any]  # output_schemas.GroundedAnswer
    answer_attempts: int
    verification: dict[str, Any]  # passed, issues
    answer: str
    answer_status: AnswerStatus

    # Actions
    proposals: list[dict[str, Any]]  # proposal dicts, updated through gate/approval/execution
    no_action_reason: str | None

    # Output
    final_response: str
    error: str | None
