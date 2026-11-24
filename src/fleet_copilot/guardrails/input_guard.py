"""Deterministic checks on the user's request before any model call.

Blocks: empty or oversized input, common prompt-injection phrasing, and
requests that name another company (tenant isolation is also enforced in the
data layer; this stops the request early and audits it as denied).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

GuardCategory = Literal["ok", "empty", "too_long", "prompt_injection", "cross_tenant"]

_INJECTION_PATTERNS = [
    r"\bignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)\b",
    r"\bdisregard (all |any |the )?(previous|prior|above|system) (instructions|rules|prompts?)\b",
    r"\b(reveal|show|print|repeat) (me )?(your|the) (system|hidden) prompt\b",
    r"\byou are no longer\b",
    r"\b(act|pretend) as (an? )?(admin|administrator|root|system)\b",
    r"\bswitch (to )?(tenant|company)\b",
    r"\bbypass (the )?(approval|guardrails?|policy)\b",
    r"\bwithout (human )?approval\b",
]
_INJECTION = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class GuardDecision:
    allowed: bool
    category: GuardCategory
    message: str


def check_input(
    question: str,
    *,
    max_chars: int,
    other_companies: dict[str, str],
) -> GuardDecision:
    """`other_companies` maps company id -> name for every tenant except the session's."""
    text = question.strip()
    if not text:
        return GuardDecision(False, "empty", "Please ask a question about your fleet.")
    if len(text) > max_chars:
        return GuardDecision(
            False, "too_long", f"Please keep requests under {max_chars} characters."
        )
    if _INJECTION.search(text):
        return GuardDecision(
            False,
            "prompt_injection",
            "This request tries to change how the copilot works, so it was not processed. "
            "Ask about your devices instead.",
        )
    for company_id, name in other_companies.items():
        for term in (company_id, name):
            if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
                return GuardDecision(
                    False,
                    "cross_tenant",
                    "You can only access your own company's devices.",
                )
    return GuardDecision(True, "ok", "")
