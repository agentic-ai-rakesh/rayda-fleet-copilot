"""The tenant a session acts for.

The tenant is fixed when a session starts (from the signed-in admin, or the
CLI `--tenant` flag) and is passed to the data layer by code. It is never a
tool argument, so nothing the model writes can change which company's data
is read.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from fleet_copilot.exceptions import TenantNotFoundError


@dataclass(frozen=True)
class TenantContext:
    company_id: str
    company_name: str
    actor: str  # the administrator acting in this session


def resolve_tenant(conn: sqlite3.Connection, tenant: str, actor: str) -> TenantContext:
    """Look up a company by id (`acme-001`) or name (`Acme`, case-insensitive)."""
    row = conn.execute(
        """
        SELECT company_id, company_name FROM companies
        WHERE company_id = :tenant OR LOWER(company_name) = LOWER(:tenant)
        """,
        {"tenant": tenant.strip()},
    ).fetchone()
    if row is None:
        raise TenantNotFoundError(f"Unknown company '{tenant}'.")
    return TenantContext(
        company_id=row["company_id"], company_name=row["company_name"], actor=actor
    )
