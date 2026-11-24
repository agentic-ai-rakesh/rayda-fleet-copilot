"""Tool suite (no LLM): each case's reference tool call against independent ground truth.

Measures whether the tools the agent relies on are correct and grounded:
- device precision/recall (or exact count) against ground truth;
- every item cites evidence that is present in the same result;
- every evidence record belongs to the case's tenant.
"""

from __future__ import annotations

import time

from fleet_copilot.evaluation.cases import EvalCase
from fleet_copilot.evaluation.eval_context import EvalContext
from fleet_copilot.evaluation.metrics import CaseResult, precision_recall


def run_tool_case(case: EvalCase, ctx: EvalContext) -> CaseResult:
    result = CaseResult(case.id, "tools", case.suite, case.tenant)
    if case.reference_tool is None or case.expect.truth is None:
        return result.skip("no reference tool or ground truth")

    started = time.perf_counter()
    executor = ctx.tool_executor(case.tenant)
    tool = executor.execute(case.reference_tool.name, case.reference_tool.args)
    result.metrics["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    tenant_id = executor.tenant.company_id

    if not result.check("tool_ok", tool.status == "ok", tool.summary):
        return result

    truth = ctx.ground_truth.resolve(case.expect.truth, tenant_id)
    found = {
        item["device_id"]
        for item in tool.items
        if "device_id" in item
        and item.get("metrics", {}).get("change", "new_failure") == "new_failure"
    }
    if isinstance(truth, int):
        result.metrics.update(expected_count=truth, found_count=len(found))
        result.check("count_correct", len(found) == truth, f"expected {truth}, got {len(found)}")
    else:
        precision, recall = precision_recall(found, truth)
        result.metrics.update(precision=precision, recall=recall)
        result.check(
            "devices_correct",
            found == truth,
            f"missing {sorted(truth - found)}, extra {sorted(found - truth)}",
        )

    evidence_ids = set(tool.evidence_ids)
    uncited = [i for i in tool.items if not set(i.get("evidence_ids", [])) <= evidence_ids]
    result.check("items_cite_present_evidence", not uncited, f"{len(uncited)} item(s)")
    owned = ctx.ground_truth.device_ids(tenant_id)
    foreign = {e.device_id for e in tool.evidence} - owned
    result.check("evidence_in_tenant", not foreign, f"foreign devices {sorted(foreign)}")
    return result
