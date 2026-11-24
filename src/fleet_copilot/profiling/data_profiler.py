"""Describe what is in the fleet database: coverage, distributions and data gaps.

This is a data-quality view, not fleet insights: it states facts about the
dataset that later steps (tools, thresholds, eval cases) rely on.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from fleet_copilot.db.connection import connect
from fleet_copilot.utils.markdown_utils import markdown_table

TableRows = list[tuple[object, ...]]


@dataclass(frozen=True)
class ProfileSection:
    title: str
    headers: tuple[str, ...]
    rows: TableRows
    note: str = ""


@dataclass
class DataProfile:
    metadata: dict[str, str]
    sections: list[ProfileSection] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


_LATEST_SNAPSHOT_CTE = """
WITH latest AS (
    SELECT s.*
    FROM snapshots s
    JOIN (SELECT device_id, MAX(collected_at) AS collected_at
          FROM snapshots GROUP BY device_id) m
      ON m.device_id = s.device_id AND m.collected_at = s.collected_at
)
"""


def build_profile(db_path: Path) -> DataProfile:
    with closing(connect(db_path, read_only=True)) as conn:
        metadata = {
            row["key"]: row["value"] for row in conn.execute("SELECT * FROM dataset_metadata")
        }
        profile = DataProfile(metadata=metadata)
        profile.sections = [
            _overview(conn),
            _companies(conn),
            _device_models(conn),
            _os_versions(conn),
            _compliance(conn),
            _battery_coverage(conn),
            _software(conn),
            _value_ranges(conn),
        ]
        profile.observations = _observations(conn)
    return profile


def render_markdown(profile: DataProfile) -> str:
    meta = profile.metadata
    lines = [
        "# Fleet Data Profile",
        "",
        f"- Source: `{meta.get('source_path', '?')}`",
        f"- Source SHA-256: `{meta.get('source_sha256', '?')}`",
        f"- Data window: {meta.get('data_start', '?')} → {meta.get('as_of', '?')} (as-of)",
        f"- Ingestion run: `{meta.get('ingestion_run_id', '?')}` at {meta.get('ingested_at', '?')}",
        "",
    ]
    for section in profile.sections:
        lines += [f"## {section.title}", ""]
        if section.note:
            lines += [section.note, ""]
        lines += [markdown_table(section.headers, section.rows), ""]
    if profile.observations:
        lines += ["## Observations", ""]
        lines += [f"- {item}" for item in profile.observations]
        lines.append("")
    return "\n".join(lines)


def _rows(conn: sqlite3.Connection, sql: str) -> TableRows:
    return [tuple(row) for row in conn.execute(sql)]


def _overview(conn: sqlite3.Connection) -> ProfileSection:
    counts = [
        ("Companies", "SELECT COUNT(*) FROM companies"),
        ("Employees", "SELECT COUNT(*) FROM employees"),
        ("Devices", "SELECT COUNT(*) FROM devices"),
        ("Snapshots", "SELECT COUNT(*) FROM snapshots"),
        ("Days covered", "SELECT COUNT(DISTINCT collected_date) FROM snapshots"),
        ("Snapshots per device (min)", "SELECT MIN(snapshot_count) FROM devices"),
        ("Snapshots per device (max)", "SELECT MAX(snapshot_count) FROM devices"),
    ]
    rows = [(label, conn.execute(sql).fetchone()[0]) for label, sql in counts]
    return ProfileSection("Overview", ("Metric", "Value"), rows)


def _companies(conn: sqlite3.Connection) -> ProfileSection:
    rows = _rows(
        conn,
        """
        SELECT c.company_id, c.company_name,
               (SELECT COUNT(*) FROM devices d WHERE d.company_id = c.company_id),
               (SELECT COUNT(*) FROM devices d WHERE d.company_id = c.company_id
                    AND d.device_type = 'laptop'),
               (SELECT COUNT(*) FROM employees e WHERE e.company_id = c.company_id),
               (SELECT COUNT(*) FROM snapshots s WHERE s.company_id = c.company_id)
        FROM companies c ORDER BY c.company_id
        """,
    )
    return ProfileSection(
        "Tenants",
        ("Company ID", "Name", "Devices", "Laptops", "Employees", "Snapshots"),
        rows,
        note="Company names are derived from the id; the dataset has no display names.",
    )


def _device_models(conn: sqlite3.Connection) -> ProfileSection:
    rows = _rows(
        conn,
        """
        SELECT model_name, device_type, os_name, COUNT(*),
               ROUND(memory_total_bytes / 1073741824.0) AS ram_gb
        FROM devices GROUP BY model_name, device_type, os_name, ram_gb
        ORDER BY model_name, ram_gb
        """,
    )
    return ProfileSection("Device models", ("Model", "Type", "OS", "Devices", "RAM (GB)"), rows)


def _os_versions(conn: sqlite3.Connection) -> ProfileSection:
    rows = _rows(
        conn,
        _LATEST_SNAPSHOT_CTE
        + """
        SELECT os_name, os_version, os_version_major, COUNT(*)
        FROM latest GROUP BY os_name, os_version, os_version_major
        ORDER BY os_name, os_version_major, os_version
        """,
    )
    return ProfileSection(
        "OS versions (latest snapshot per device)",
        ("OS", "Version", "Major", "Devices"),
        rows,
    )


def _compliance(conn: sqlite3.Connection) -> ProfileSection:
    rows = _rows(
        conn,
        """
        SELECT check_id, severity,
               SUM(status = 'pass'), SUM(status = 'fail'),
               COUNT(DISTINCT CASE WHEN status = 'fail' THEN s.device_id END)
        FROM compliance_results c JOIN snapshots s USING (snapshot_id)
        GROUP BY check_id, severity ORDER BY check_id
        """,
    )
    return ProfileSection(
        "Compliance checks (all snapshots)",
        ("Check", "Severity", "Pass", "Fail", "Devices ever failing"),
        rows,
    )


def _battery_coverage(conn: sqlite3.Connection) -> ProfileSection:
    rows = _rows(
        conn,
        """
        SELECT d.device_type, COUNT(*), SUM(s.battery_reported), SUM(1 - s.battery_reported)
        FROM snapshots s JOIN devices d USING (device_id)
        GROUP BY d.device_type ORDER BY d.device_type
        """,
    )
    return ProfileSection(
        "Battery data coverage",
        ("Device type", "Snapshots", "With battery data", "Without battery data"),
        rows,
        note="Desktops have no battery. Laptop snapshots without battery data are telemetry gaps.",
    )


def _software(conn: sqlite3.Connection) -> ProfileSection:
    rows = _rows(
        conn,
        """
        SELECT i.name, COUNT(*), COUNT(DISTINCT s.device_id)
        FROM installed_software i JOIN snapshots s USING (snapshot_id)
        GROUP BY i.name ORDER BY COUNT(*) DESC, i.name
        """,
    )
    return ProfileSection("Installed software", ("Name", "Snapshots", "Devices"), rows)


def _value_ranges(conn: sqlite3.Connection) -> ProfileSection:
    # (label, column, table, filter) - fixed internal values, never user input.
    metrics = [
        ("Memory used %", "memory_used_pct", "snapshots", "1 = 1"),
        ("System disk free %", "free_pct", "disk_volumes", "is_system_volume = 1"),
        ("Battery cycle count", "battery_cycle_count", "snapshots", "1 = 1"),
        (
            "Battery full-charge capacity (mAh)",
            "battery_full_charge_capacity",
            "snapshots",
            "1 = 1",
        ),
    ]
    rows = []
    for label, column, table, where in metrics:
        sql = f"SELECT MIN({column}), AVG({column}), MAX({column}) FROM {table} WHERE {where}"
        low, mean, high = conn.execute(sql).fetchone()
        rows.append((label, _fmt(low), _fmt(mean), _fmt(high)))
    return ProfileSection("Value ranges", ("Metric", "Min", "Mean", "Max"), rows)


def _observations(conn: sqlite3.Connection) -> list[str]:
    """Facts later steps must handle correctly (e.g. an empty answer is the right answer)."""
    notes: list[str] = []
    for check_id, severity in conn.execute(
        """SELECT check_id, severity FROM compliance_results
           GROUP BY check_id, severity HAVING SUM(status = 'fail') = 0"""
    ):
        notes.append(
            f"No snapshot fails `{check_id}` ({severity}); questions about it should return none."
        )

    gaps = conn.execute(
        """SELECT COUNT(*) FROM snapshots s JOIN devices d USING (device_id)
           WHERE d.device_type = 'laptop' AND s.battery_reported = 0"""
    ).fetchone()[0]
    if gaps:
        notes.append(
            f"{gaps} laptop snapshots have no battery data; "
            "trend logic must skip them, not treat them as zero."
        )

    os_changes = conn.execute(
        "SELECT COUNT(*) FROM (SELECT device_id FROM snapshots GROUP BY device_id "
        "HAVING COUNT(DISTINCT os_version) > 1)"
    ).fetchone()[0]
    notes.append(f"{os_changes} devices changed OS version during the window.")

    no_design_capacity = (
        "Battery records have no design capacity; battery health must come from cycle count, "
    )
    notes.append(no_design_capacity + "condition and the full-charge capacity trend.")
    return notes


def _fmt(value: float | None) -> str | None:
    if value is None:
        return None
    if float(value).is_integer() or abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:.2f}"
