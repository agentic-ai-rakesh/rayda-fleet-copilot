"""Check a drafted answer against this turn's tool results.

Checks:
- every claim cites at least one snapshot id;
- every cited id (in claims and in the answer text) exists in this turn's evidence;
- every number in a claim appears somewhere in the tool results
  (exact for whole numbers, within rounding for decimals).
The answer text itself is not parsed for numbers beyond its claims: the
responder is required to list every factual statement as a claim.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from fleet_copilot.prompts.output_schemas import GroundedAnswer

# Snapshot ids look like DEVICE@20260612T0839Z.
SNAPSHOT_ID = re.compile(r"[A-Za-z0-9_-]+@\d{8}T\d{4}Z")
_NUMBER = re.compile(r"(?<![\w@.])-?\d+(?:\.\d+)?(?![\w@])")


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    issues: list[str]


def verify_answer(draft: GroundedAnswer, tool_results: list[dict[str, Any]]) -> VerificationResult:
    if not draft.can_answer:
        return VerificationResult(True, [])

    known_ids = {e["snapshot_id"] for r in tool_results for e in r.get("evidence", [])}
    source_numbers = _numbers_in(json.dumps(tool_results))
    issues: list[str] = []

    if not draft.claims:
        issues.append("The answer has no claims; list each factual statement with its evidence.")
    for claim in draft.claims:
        if not claim.evidence_ids:
            issues.append(f"Claim has no evidence: '{claim.statement}'.")
        unknown = [eid for eid in claim.evidence_ids if eid not in known_ids]
        if unknown:
            issues.append(
                f"Claim cites ids not in the tool results {unknown}: '{claim.statement}'."
            )
        statement = SNAPSHOT_ID.sub(" ", claim.statement)
        unsupported = [n for n in _numbers_in(statement) if not _supported(n, source_numbers)]
        if unsupported:
            issues.append(
                f"Numbers {unsupported} in '{claim.statement}' do not appear in the tool results."
            )

    text_ids = set(SNAPSHOT_ID.findall(draft.answer))
    unknown_in_text = sorted(text_ids - known_ids)
    if unknown_in_text:
        issues.append(f"The answer cites ids not in the tool results: {unknown_in_text}.")
    return VerificationResult(not issues, issues)


def _numbers_in(text: str) -> list[float]:
    return [float(match) for match in _NUMBER.findall(text)]


def _supported(value: float, sources: list[float]) -> bool:
    if value.is_integer():
        return any(abs(s - value) < 0.5 for s in sources)  # allows rounding, e.g. 2 from 2.04
    return any(abs(value - s) <= 0.051 for s in sources)
