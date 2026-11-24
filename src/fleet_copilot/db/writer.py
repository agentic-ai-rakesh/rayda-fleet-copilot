"""Bulk inserts of typed rows into the fleet database."""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import astuple, fields
from typing import Any

from fleet_copilot.db.rows import (
    CompanyRow,
    ComplianceResultRow,
    DeviceRow,
    DiskVolumeRow,
    EmployeeRow,
    InstalledSoftwareRow,
    SnapshotRow,
)

Row = (
    CompanyRow
    | EmployeeRow
    | DeviceRow
    | SnapshotRow
    | DiskVolumeRow
    | ComplianceResultRow
    | InstalledSoftwareRow
)


def insert_rows(conn: sqlite3.Connection, rows: Sequence[Row]) -> int:
    """Insert rows of a single type. Table and column names come from the row class."""
    if not rows:
        return 0
    row_type = type(rows[0])
    if any(type(row) is not row_type for row in rows):
        raise TypeError("insert_rows expects rows of a single type.")

    columns = [f.name for f in fields(row_type)]
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO {row_type.table} ({', '.join(columns)}) VALUES ({placeholders})"
    conn.executemany(sql, (astuple(row) for row in rows))
    return len(rows)


def write_metadata(conn: sqlite3.Connection, items: Mapping[str, Any]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO dataset_metadata (key, value) VALUES (?, ?)",
        ((key, str(value)) for key, value in items.items()),
    )
