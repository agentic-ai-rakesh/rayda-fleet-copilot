"""Read the NDJSON dataset, validate each line, and quarantine bad records.

Fallback policy: a bad line never stops the load. It is rejected with a
reason, written to the quarantine file for review, and logged. The pipeline
decides afterwards whether the reject ratio is acceptable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from fleet_copilot.exceptions import IngestionError
from fleet_copilot.ingestion.raw_models import TelemetryRecord
from fleet_copilot.utils.logging_utils import get_run_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RejectedRecord:
    line_number: int
    reason: str
    raw_line: str


@dataclass
class LoadResult:
    records: list[TelemetryRecord] = field(default_factory=list)
    rejected: list[RejectedRecord] = field(default_factory=list)

    @property
    def lines_read(self) -> int:
        return len(self.records) + len(self.rejected)

    @property
    def reject_ratio(self) -> float:
        return len(self.rejected) / self.lines_read if self.lines_read else 0.0


def load_telemetry(path: Path) -> LoadResult:
    if not path.is_file():
        raise IngestionError(f"Dataset not found at {path}.")

    result = LoadResult()
    seen_snapshots: set[tuple[str, str]] = set()

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            record, reason = _parse_line(text)
            if record is not None:
                key = (record.device_id, record.collected_at.isoformat())
                if key in seen_snapshots:
                    record, reason = None, "duplicate_snapshot: same device_id and collected_at"
                else:
                    seen_snapshots.add(key)

            if record is None:
                result.rejected.append(RejectedRecord(line_number, reason, text))
                logger.warning("Rejected line %d: %s", line_number, reason)
            else:
                result.records.append(record)

    logger.info(
        "Loaded %d records, rejected %d (%.1f%%) from %s",
        len(result.records),
        len(result.rejected),
        100 * result.reject_ratio,
        path,
    )
    return result


def write_quarantine(rejected: list[RejectedRecord], path: Path) -> None:
    """Replace the quarantine file with this run's rejects (empty file when none)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    run_id = get_run_id()
    with path.open("w", encoding="utf-8") as handle:
        for item in rejected:
            entry = {
                "run_id": run_id,
                "line_number": item.line_number,
                "reason": item.reason,
                "raw_line": item.raw_line,
            }
            handle.write(json.dumps(entry) + "\n")


def _parse_line(text: str) -> tuple[TelemetryRecord | None, str]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid_json: {exc.msg} (column {exc.colno})"
    if not isinstance(payload, dict):
        return None, "invalid_json: line is not a JSON object"
    try:
        return TelemetryRecord.model_validate(payload), ""
    except ValidationError as exc:
        return None, f"schema_validation: {_summarise(exc)}"


def _summarise(exc: ValidationError, limit: int = 3) -> str:
    parts = [
        f"{'.'.join(str(p) for p in err['loc']) or 'record'}: {err['msg']}"
        for err in exc.errors()[:limit]
    ]
    extra = exc.error_count() - limit
    if extra > 0:
        parts.append(f"(+{extra} more)")
    return "; ".join(parts)
