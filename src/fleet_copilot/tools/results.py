"""What a tool returns.

Handlers return a ToolOutput (the facts). The executor wraps it into a
ToolResult, adding tenant, as-of time and truncation, so every tool answers
in the same shape and the model always sees what scope and assumptions
produced the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from fleet_copilot.evidence import Evidence

ToolStatus = Literal["ok", "not_found", "error"]


@dataclass
class ToolOutput:
    summary: str
    items: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    status: ToolStatus = "ok"


class ToolResult(BaseModel):
    tool: str
    status: ToolStatus
    summary: str
    tenant_id: str
    as_of: str | None = None
    items: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    truncated: bool = False
    error: str | None = None

    @property
    def evidence_ids(self) -> list[str]:
        return [item.snapshot_id for item in self.evidence]

    def to_model_payload(self) -> dict[str, Any]:
        """Compact form for the LLM: drops empty and default fields."""
        return self.model_dump(exclude_none=True, exclude_defaults=True)
