"""Result types and scoring helpers shared by the suites."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

CaseStatus = Literal["passed", "failed", "skipped", "error"]
Layer = Literal["tools", "gate", "agent"]

SNAPSHOT_ID = re.compile(r"[A-Za-z0-9_-]+@\d{8}T\d{4}Z")


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case_id: str
    layer: Layer
    suite: str
    tenant: str
    status: CaseStatus = "passed"
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, passed, detail))
        if not passed and self.status == "passed":
            self.status = "failed"
        return passed

    def skip(self, reason: str) -> CaseResult:
        self.status, self.note = "skipped", reason
        return self

    def fail_with_error(self, reason: str) -> CaseResult:
        self.status, self.note = "error", reason
        return self


def precision_recall(found: set[str], expected: set[str]) -> tuple[float, float]:
    """Empty sets count as perfect: finding nothing when nothing is expected is correct."""
    precision = len(found & expected) / len(found) if found else 1.0
    recall = len(found & expected) / len(expected) if expected else 1.0
    return round(precision, 3), round(recall, 3)


def strip_snapshot_ids(text: str) -> str:
    """Remove citations so a cited snapshot does not count as naming its device."""
    return SNAPSHOT_ID.sub(" ", text)


def mentioned_devices(text: str, candidates: Iterable[str]) -> set[str]:
    plain = strip_snapshot_ids(text)
    return {d for d in candidates if re.search(rf"(?<![\w@]){re.escape(d)}(?![\w@])", plain)}


def contains_number(text: str, value: int) -> bool:
    plain = strip_snapshot_ids(text)
    return re.search(rf"(?<![\d.]){value}(?![\d.])", plain) is not None


_NONE_PATTERNS = re.compile(
    r"(?<![\w])(none|no (devices?|laptops?|batter(y|ies)|findings?)|0 of \d+|zero|not any)(?![\w])",
    re.IGNORECASE,
)


def says_none(text: str) -> bool:
    """True if the answer states that nothing matches."""
    return _NONE_PATTERNS.search(strip_snapshot_ids(text)) is not None
