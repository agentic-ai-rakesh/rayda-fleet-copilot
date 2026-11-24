"""Eval case models and loading from evals/cases/*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fleet_copilot.exceptions import ConfigError

CaseSuite = Literal["grounded_qa", "insight", "action", "guardrail"]
GATE_CASES_FILE = "gate_redteam.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCallSpec(_Strict):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ProposalExpectation(_Strict):
    actions: list[str]
    allowed_targets: str  # ground-truth key: targets an approved proposal may have
    min_ready: int = 1


class Expectation(_Strict):
    truth: str | None = None  # ground-truth key: device set or count
    tools_any: list[str] = Field(default_factory=list)
    answer_status: list[str] = Field(default_factory=lambda: ["verified"])
    llm_calls: int | None = None  # 0 = must be decided without the model
    cannot_answer: bool = False
    proposals: ProposalExpectation | None = None


class EvalCase(_Strict):
    id: str
    suite: CaseSuite
    tenant: str
    question: str | None = None
    reference_tool: ToolCallSpec | None = None
    expect: Expectation = Field(default_factory=Expectation)

    @property
    def needs_llm(self) -> bool:
        return self.question is not None and self.expect.llm_calls != 0


class GateCase(_Strict):
    id: str
    tenant: str
    evidence_tools: list[ToolCallSpec]
    proposal: dict[str, Any]
    expect_allowed: bool


def load_cases(cases_dir: Path) -> tuple[list[EvalCase], list[GateCase]]:
    cases: list[EvalCase] = []
    gate_cases: list[GateCase] = []
    for path in sorted(cases_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        try:
            if path.name == GATE_CASES_FILE:
                gate_cases.extend(GateCase.model_validate(item) for item in raw)
            else:
                cases.extend(EvalCase.model_validate(item) for item in raw)
        except Exception as exc:
            raise ConfigError(f"Invalid eval case in {path}: {exc}") from exc

    ids = [c.id for c in cases] + [c.id for c in gate_cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ConfigError(f"Duplicate eval case ids: {duplicates}")
    return cases, gate_cases
