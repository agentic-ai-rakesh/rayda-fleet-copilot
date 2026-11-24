"""The shape of one insight: finding, supporting evidence, brief explanation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from fleet_copilot.evidence import Evidence

InsightSeverity = Literal["high", "medium", "low"]
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass(frozen=True)
class Insight:
    insight_type: str  # e.g. "battery_end_of_life"
    device_id: str
    severity: InsightSeverity
    finding: str  # one sentence: what is true
    explanation: str  # why it matters / how it was judged
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)

    def to_item(self) -> dict[str, Any]:
        return {
            "insight_type": self.insight_type,
            "device_id": self.device_id,
            "severity": self.severity,
            "finding": self.finding,
            "explanation": self.explanation,
            "metrics": self.metrics,
            "evidence_ids": [e.snapshot_id for e in self.evidence],
        }


def group_by_device(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rows per device, each list sorted oldest first."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["device_id"]].append(row)
    for series in grouped.values():
        series.sort(key=lambda r: r["collected_at"])
    return dict(grouped)


def sort_insights(insights: list[Insight]) -> list[Insight]:
    return sorted(insights, key=lambda i: (SEVERITY_ORDER[i.severity], i.device_id))
