"""Deterministic trend detectors over time-series telemetry.

Detectors are pure functions: rows in, Insight objects out. No database, no
LLM. The model explains insights; it never computes them.
"""

from fleet_copilot.insights.models import Insight

__all__ = ["Insight"]
