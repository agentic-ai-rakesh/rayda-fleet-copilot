"""Ingestion pipeline: raw NDJSON -> validated records -> SQLite fleet database.

Fallback and safety criteria:
- Bad lines are quarantined, not fatal (see loader.py).
- If the reject ratio exceeds `ingestion.max_reject_ratio`, the load aborts.
- The database is built in a temporary file and swapped in atomically, so a
  failed run always leaves the previous good database in place.
- Start, success and failure are written to the audit log. If the audit
  write fails, the ingestion fails too.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from fleet_copilot import __version__
from fleet_copilot.audit import AuditLogger, AuditStatus
from fleet_copilot.config import Settings
from fleet_copilot.db.connection import connect, create_schema
from fleet_copilot.db.writer import insert_rows, write_metadata
from fleet_copilot.exceptions import IngestionError
from fleet_copilot.ingestion.loader import load_telemetry, write_quarantine
from fleet_copilot.ingestion.transformer import (
    DimensionRows,
    SnapshotRows,
    build_dimensions,
    to_snapshot_rows,
)
from fleet_copilot.utils.hash_utils import sha256_file
from fleet_copilot.utils.logging_utils import get_run_id
from fleet_copilot.utils.time_utils import utc_now_iso

logger = logging.getLogger(__name__)

AUDIT_EVENT_TYPE = "ingestion"
AUDIT_ACTION = "ingest_dataset"


@dataclass(frozen=True)
class IngestionSummary:
    run_id: str
    source_path: str
    source_sha256: str
    db_path: str
    lines_read: int
    records_loaded: int
    records_rejected: int
    companies: int
    employees: int
    devices: int
    snapshots: int
    data_start: str
    data_end: str  # the dataset's "as of" time: the latest snapshot


def run_ingestion(settings: Settings, audit: AuditLogger) -> IngestionSummary:
    paths = settings.paths
    audit.record(
        event_type=AUDIT_EVENT_TYPE,
        action=AUDIT_ACTION,
        status=AuditStatus.STARTED,
        details={"source_path": str(paths.raw_dataset)},
    )
    try:
        summary = _ingest(settings)
    except Exception as exc:
        logger.error("Ingestion failed: %s", exc)
        audit.record(
            event_type=AUDIT_EVENT_TYPE,
            action=AUDIT_ACTION,
            status=AuditStatus.FAILED,
            details={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise

    audit.record(
        event_type=AUDIT_EVENT_TYPE,
        action=AUDIT_ACTION,
        status=AuditStatus.SUCCEEDED,
        details=asdict(summary),
    )
    return summary


def _ingest(settings: Settings) -> IngestionSummary:
    paths = settings.paths
    load = load_telemetry(paths.raw_dataset)
    write_quarantine(load.rejected, paths.quarantine_file)

    if not load.records:
        raise IngestionError("No valid records in the dataset.")
    max_ratio = settings.ingestion.max_reject_ratio
    if load.reject_ratio > max_ratio:
        raise IngestionError(
            f"{len(load.rejected)} of {load.lines_read} records failed validation "
            f"({load.reject_ratio:.1%} > limit {max_ratio:.1%}). "
            f"See {paths.quarantine_file}. The previous database was kept."
        )

    dimensions = build_dimensions(load.records, settings.device_classification.desktop_models)
    snapshots = [to_snapshot_rows(record) for record in load.records]
    times = sorted(row.snapshot.collected_at for row in snapshots)

    summary = IngestionSummary(
        run_id=get_run_id(),
        source_path=str(paths.raw_dataset),
        source_sha256=sha256_file(paths.raw_dataset),
        db_path=str(paths.fleet_db),
        lines_read=load.lines_read,
        records_loaded=len(load.records),
        records_rejected=len(load.rejected),
        companies=len(dimensions.companies),
        employees=len(dimensions.employees),
        devices=len(dimensions.devices),
        snapshots=len(snapshots),
        data_start=times[0],
        data_end=times[-1],
    )
    _build_database_atomically(paths.fleet_db, dimensions, snapshots, summary)
    logger.info(
        "Fleet database ready: %d devices, %d snapshots, data up to %s",
        summary.devices,
        summary.snapshots,
        summary.data_end,
        extra={"context": asdict(summary)},
    )
    return summary


def _build_database_atomically(
    target: Path,
    dimensions: DimensionRows,
    snapshots: list[SnapshotRows],
    summary: IngestionSummary,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_name(target.name + ".tmp")
    temp_path.unlink(missing_ok=True)

    try:
        conn = connect(temp_path)
        try:
            create_schema(conn)
            with conn:  # one transaction for all inserts
                insert_rows(conn, dimensions.companies)
                insert_rows(conn, dimensions.employees)
                insert_rows(conn, dimensions.devices)
                insert_rows(conn, [s.snapshot for s in snapshots])
                insert_rows(conn, [d for s in snapshots for d in s.disk_volumes])
                insert_rows(conn, [c for s in snapshots for c in s.compliance_results])
                insert_rows(conn, [i for s in snapshots for i in s.installed_software])
                write_metadata(
                    conn,
                    {
                        "as_of": summary.data_end,
                        "data_start": summary.data_start,
                        "source_path": summary.source_path,
                        "source_sha256": summary.source_sha256,
                        "ingestion_run_id": summary.run_id,
                        "ingested_at": utc_now_iso(),
                        "app_version": __version__,
                    },
                )
        finally:
            conn.close()
        os.replace(temp_path, target)  # atomic swap; readers never see a half-built file
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        if isinstance(exc, IngestionError):
            raise
        raise IngestionError(f"Failed to build the fleet database: {exc}") from exc
