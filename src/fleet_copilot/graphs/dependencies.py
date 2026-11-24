"""Services the graph nodes use. Built once per session and passed to each node factory."""

from __future__ import annotations

from dataclasses import dataclass

from fleet_copilot.audit import AuditLogger
from fleet_copilot.config import Settings
from fleet_copilot.guardrails.tenant_context import TenantContext
from fleet_copilot.llm.llm_client import LLMClient
from fleet_copilot.tools.action_tools import ActionExecutor
from fleet_copilot.tools.executor import ToolExecutor


@dataclass(frozen=True)
class GraphDependencies:
    settings: Settings
    tenant: TenantContext
    llm: LLMClient
    tools: ToolExecutor
    actions: ActionExecutor
    audit: AuditLogger
    other_companies: dict[str, str]  # id -> name of every tenant except this one
