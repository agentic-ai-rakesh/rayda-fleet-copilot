"""SQLite connection helpers for the fleet database."""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path

from fleet_copilot.exceptions import DatabaseNotReadyError


def connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the fleet database. Read-only mode is used by everything except ingestion."""
    if read_only:
        if not db_path.exists():
            raise DatabaseNotReadyError(
                f"Fleet database not found at {db_path}. Run `fleet-copilot ingest` first."
            )
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    schema_sql = resources.files("fleet_copilot.db").joinpath("schema.sql").read_text("utf-8")
    conn.executescript(schema_sql)
