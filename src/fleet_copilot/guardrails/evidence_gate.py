"""Evidence gate: refuse action proposals the evidence does not support.

Runs on every proposal before a human sees it. All checks are deterministic
and use only this turn's tool results plus tenant-scoped lookups:

1. Arguments match the action's schema.
2. The target device/employee belongs to the session tenant.
3. Cited evidence ids exist in this turn's results and concern the target.
4. The action is justified by a matching finding:
   - flag_device_for_replacement: battery at end of life, or two or more
     high-severity findings on the device
   - create_upgrade_order: battery -> battery finding; ram -> memory-pressure
     finding; storage -> storage finding or current low disk evidence
   - open_remediation_ticket: the check is failing in the device's latest snapshot
   - notify_employee: one of the employee's devices has a finding or failing
     check in this turn; message within the length limit
5. No duplicate proposal (same action, same target) in one turn.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from fleet_copilot.config.settings import AgentSettings, ThresholdSettings
from fleet_copilot.db.fleet_repository import FleetRepository
from fleet_copilot.prompts.output_schemas import ProposedAction
from fleet_copilot.tools.action_tools import ACTION_REGISTRY


@dataclass
class TurnEvidence:
    """Facts gathered by tools in this turn, indexed for the gate."""

    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)  # snapshot_id -> evidence
    insights: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # device -> items

    @classmethod
    def from_tool_results(cls, results: list[dict[str, Any]]) -> TurnEvidence:
        turn = cls()
        insights: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            for item in result.get("evidence", []):
                turn.evidence[item["snapshot_id"]] = item
            for item in result.get("items", []):
                if "insight_type" in item:
                    insights[item["device_id"]].append(item)
        turn.insights = dict(insights)
        return turn

    def insight(self, device_id: str, insight_type: str) -> dict[str, Any] | None:
        return next(
            (i for i in self.insights.get(device_id, []) if i["insight_type"] == insight_type), None
        )

    def evidence_for(self, device_id: str) -> list[dict[str, Any]]:
        return [e for e in self.evidence.values() if e["device_id"] == device_id]


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    arguments: dict[str, Any]
    reasons: list[str]


class EvidenceGate:
    def __init__(
        self, repo: FleetRepository, thresholds: ThresholdSettings, agent: AgentSettings
    ) -> None:
        self._repo = repo
        self._thresholds = thresholds
        self._agent = agent

    def review(self, proposals: list[ProposedAction], turn: TurnEvidence) -> list[GateDecision]:
        decisions, seen = [], set()
        for proposal in proposals:
            decision = self._review_one(proposal, turn)
            target = (proposal.action, proposal.device_id or proposal.employee_id)
            if decision.allowed and target in seen:
                decision = GateDecision(
                    False, decision.arguments, ["Duplicate proposal in this turn."]
                )
            seen.add(target)
            decisions.append(decision)
        return decisions

    def _review_one(self, proposal: ProposedAction, turn: TurnEvidence) -> GateDecision:
        spec = ACTION_REGISTRY[proposal.action]
        arguments = proposal.arguments(spec.argument_names)
        try:
            arguments = spec.input_model.model_validate(arguments).model_dump()
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()[:3]
            )
            return GateDecision(False, arguments, [f"Invalid arguments: {problems}"])

        devices = self._target_devices(spec.target, arguments[spec.target])
        if not devices:
            return GateDecision(
                False,
                arguments,
                [f"{arguments[spec.target]} is not in {self._repo.tenant.company_name}'s fleet."],
            )

        reasons = self._check_citations(proposal.evidence_ids, devices, turn)
        reasons += self._check_support(proposal.action, arguments, devices, turn)
        return GateDecision(not reasons, arguments, reasons)

    def _target_devices(self, target: str, value: str) -> list[str]:
        if target == "employee_id":
            return self._repo.employee_device_ids(value)
        return [value] if self._repo.latest_device_state(value) is not None else []

    @staticmethod
    def _check_citations(
        evidence_ids: list[str], devices: list[str], turn: TurnEvidence
    ) -> list[str]:
        if not evidence_ids:
            return ["No evidence cited."]
        unknown = [eid for eid in evidence_ids if eid not in turn.evidence]
        if unknown:
            return [f"Cited evidence not found in this turn's tool results: {unknown}."]
        if not any(turn.evidence[eid]["device_id"] in devices for eid in evidence_ids):
            return ["None of the cited evidence concerns the target."]
        return []

    def _check_support(
        self, action: str, args: dict[str, Any], devices: list[str], turn: TurnEvidence
    ) -> list[str]:
        if action == "flag_device_for_replacement":
            device = args["device_id"]
            battery = turn.insight(device, "battery_end_of_life")
            high = [i for i in turn.insights.get(device, []) if i["severity"] == "high"]
            battery_eol = battery and battery["metrics"].get("status") == "end_of_life"
            if battery_eol or len(high) >= 2:
                return []
            return [
                "Replacement needs a battery at end of life or at least two high-severity "
                "findings for the device in this turn."
            ]

        if action == "create_upgrade_order":
            device, component = args["device_id"], args["component"]
            needed = {
                "battery": "battery_end_of_life",
                "ram": "memory_pressure",
                "storage": "storage_pressure",
            }[component]
            if turn.insight(device, needed):
                return []
            if component == "storage" and any(
                (e["values"].get("disk_free_pct") or 100) < self._thresholds.low_disk_free_pct
                for e in turn.evidence_for(device)
            ):
                return []
            return [
                f"A {component} upgrade needs a {needed.replace('_', ' ')} finding for {device}."
            ]

        if action == "open_remediation_ticket":
            device, check_id = args["device_id"], args["check_id"]
            latest = self._repo.latest_device_state(device)
            failing_now = latest is not None and any(
                check_id in failing_checks(e) and e["collected_at"] == latest["collected_at"]
                for e in turn.evidence_for(device)
            )
            return (
                []
                if failing_now
                else [f"No evidence that '{check_id}' is failing in {device}'s latest snapshot."]
            )

        if action == "notify_employee":
            reasons = []
            if len(args["message"]) > self._agent.max_notification_chars:
                reasons.append(
                    f"Message is longer than {self._agent.max_notification_chars} characters."
                )
            has_issue = any(
                turn.insights.get(d)
                or any(failing_checks(e) for e in turn.evidence_for(d))
                or any(
                    (e["values"].get("disk_free_pct") or 100) < self._thresholds.low_disk_free_pct
                    for e in turn.evidence_for(d)
                )
                for d in devices
            )
            if not has_issue:
                reasons.append("No finding about this employee's devices in this turn.")
            return reasons

        return [f"No evidence rule for action '{action}'."]


def failing_checks(evidence: dict[str, Any]) -> set[str]:
    """Failing check ids in one evidence record.

    Compliance evidence comes in two shapes: one check per record
    (`check_id`/`status`, from compliance tools) or all checks of a snapshot
    (`compliance: {check_id: status}`, from get_device_details).
    """
    values = evidence["values"]
    failing = {
        check for check, status in (values.get("compliance") or {}).items() if status == "fail"
    }
    if values.get("status") == "fail" and values.get("check_id"):
        failing.add(values["check_id"])
    return failing
