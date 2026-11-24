"""FleetCopilotAgent: one administrator's session with the copilot.

    agent = FleetCopilotAgent.create(settings, tenant="acme", actor="alice@acme")
    turn = agent.ask("Which batteries need replacing? Propose actions.")
    if turn.awaiting_approval:
        turn = agent.decide(turn.thread_id, [ApprovalDecision(pid, approved=True)])

Each question runs in its own graph thread. The thread is checkpointed in
SQLite, so an approval can be given later, even from another process, with
`decide(thread_id, ...)`.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from fleet_copilot.audit import AuditLogger
from fleet_copilot.config import Settings
from fleet_copilot.db.action_store import ActionStore
from fleet_copilot.db.connection import connect
from fleet_copilot.db.fleet_repository import FleetRepository
from fleet_copilot.exceptions import FleetCopilotError
from fleet_copilot.graphs.dependencies import GraphDependencies
from fleet_copilot.graphs.fleet_copilot_graph import build_fleet_copilot_graph
from fleet_copilot.graphs.nodes.human_approval_node import APPROVAL_REQUEST
from fleet_copilot.guardrails.approval_policy import ApprovalDecision
from fleet_copilot.guardrails.tenant_context import TenantContext, resolve_tenant
from fleet_copilot.llm import LLMClient, build_llm_client
from fleet_copilot.tools.action_tools import ActionExecutor
from fleet_copilot.tools.executor import ToolExecutor
from fleet_copilot.utils.logging_utils import set_run_id

logger = logging.getLogger(__name__)

RECURSION_LIMIT = 60


@dataclass(frozen=True)
class AgentTurn:
    thread_id: str
    response: str
    answer_status: str | None
    pending_proposals: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def awaiting_approval(self) -> bool:
        return bool(self.pending_proposals)


class FleetCopilotAgent:
    def __init__(self, deps: GraphDependencies, checkpoint_conn: sqlite3.Connection) -> None:
        self._deps = deps
        self._checkpoint_conn = checkpoint_conn
        self._graph = build_fleet_copilot_graph(deps, SqliteSaver(checkpoint_conn))

    @classmethod
    def create(
        cls,
        settings: Settings,
        tenant: str,
        actor: str,
        llm: LLMClient | None = None,
    ) -> FleetCopilotAgent:
        """Resolve the tenant, build every service, and open the checkpoint store."""
        with closing(connect(settings.paths.fleet_db, read_only=True)) as conn:
            tenant_ctx = resolve_tenant(conn, tenant, actor=actor)
            others = {
                row["company_id"]: row["company_name"]
                for row in conn.execute("SELECT company_id, company_name FROM companies")
                if row["company_id"] != tenant_ctx.company_id
            }
        audit = AuditLogger(settings.paths.audit_db)
        deps = GraphDependencies(
            settings=settings,
            tenant=tenant_ctx,
            llm=llm or build_llm_client(settings.llm),
            tools=ToolExecutor(settings, tenant_ctx, audit),
            actions=ActionExecutor(
                settings, tenant_ctx, audit, ActionStore(settings.paths.actions_db)
            ),
            audit=audit,
            other_companies=others,
        )
        settings.paths.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_conn = sqlite3.connect(settings.paths.checkpoint_db, check_same_thread=False)
        return cls(deps, checkpoint_conn)

    @property
    def tenant(self) -> TenantContext:
        return self._deps.tenant

    def ask(self, question: str) -> AgentTurn:
        thread_id = f"{self.tenant.company_id}-{uuid.uuid4().hex[:12]}"
        set_run_id(thread_id)
        logger.info("New question on thread %s", thread_id)
        initial = {
            "question": question,
            "tenant": {
                "company_id": self.tenant.company_id,
                "company_name": self.tenant.company_name,
                "actor": self.tenant.actor,
            },
            "as_of": self._as_of(),
        }
        self._graph.invoke(initial, self._config(thread_id))
        return self._turn(thread_id)

    def decide(
        self, thread_id: str, decisions: list[ApprovalDecision], approver: str | None = None
    ) -> AgentTurn:
        """Resume a thread waiting for approval. Proposals without a decision are rejected."""
        if not thread_id.startswith(self.tenant.company_id + "-"):
            raise FleetCopilotError("This conversation belongs to another company.")
        if not self._turn(thread_id).awaiting_approval:
            raise FleetCopilotError(f"Thread {thread_id} is not waiting for approval.")
        set_run_id(thread_id)
        payload = {
            "approver": approver or self.tenant.actor,
            "decisions": [
                {"proposal_id": d.proposal_id, "approved": d.approved, "comment": d.comment}
                for d in decisions
            ],
        }
        self._graph.invoke(Command(resume=payload), self._config(thread_id))
        return self._turn(thread_id)

    def close(self) -> None:
        self._checkpoint_conn.close()

    # ----------------------------------------------------------------- helpers
    def _turn(self, thread_id: str) -> AgentTurn:
        snapshot = self._graph.get_state(self._config(thread_id))
        values = dict(snapshot.values)
        pending: list[dict[str, Any]] = []
        for interrupt in snapshot.interrupts:
            if (
                isinstance(interrupt.value, dict)
                and interrupt.value.get("type") == APPROVAL_REQUEST
            ):
                pending = interrupt.value["proposals"]
        # While waiting for approval there is no final response yet: show the answer.
        response = values.get("answer", "") if pending else values.get("final_response", "")
        return AgentTurn(
            thread_id=thread_id,
            response=response,
            answer_status=values.get("answer_status"),
            pending_proposals=pending,
            state=values,
        )

    def _as_of(self) -> str:
        with closing(connect(self._deps.settings.paths.fleet_db, read_only=True)) as conn:
            return FleetRepository(conn, self.tenant).as_of() or ""

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
