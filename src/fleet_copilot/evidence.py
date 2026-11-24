"""Evidence: the unit of grounding.

Every fact a tool or detector returns points to the snapshot(s) it came from.
The agent cites `snapshot_id`s; a later verifier checks each citation exists
in that turn's tool results and that the quoted values match.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict


class Evidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    device_id: str
    collected_at: str
    values: dict[str, Any]


def evidence_from_row(row: Mapping[str, Any], fields: Iterable[str]) -> Evidence:
    """Build evidence from a snapshot row, keeping only the fields that support the claim."""
    return Evidence(
        snapshot_id=row["snapshot_id"],
        device_id=row["device_id"],
        collected_at=row["collected_at"],
        values={field: row.get(field) for field in fields},
    )


def merge_evidence(items: Iterable[Evidence]) -> list[Evidence]:
    """De-duplicate by snapshot, merging the cited values; keeps first-seen order."""
    merged: dict[str, Evidence] = {}
    for item in items:
        existing = merged.get(item.snapshot_id)
        if existing is None:
            merged[item.snapshot_id] = item
        else:
            merged[item.snapshot_id] = existing.model_copy(
                update={"values": {**existing.values, **item.values}}
            )
    return list(merged.values())
