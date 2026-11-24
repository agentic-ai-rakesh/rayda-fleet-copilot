"""Summarise case results into a Markdown scorecard and a JSON file."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from fleet_copilot.evaluation.metrics import CaseResult
from fleet_copilot.utils.markdown_utils import markdown_table


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    by_layer: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_layer[r.layer].append(r)

    def counts(items: list[CaseResult]) -> dict[str, Any]:
        scored = [r for r in items if r.status in ("passed", "failed", "error")]
        passed = sum(r.status == "passed" for r in scored)
        return {
            "cases": len(items),
            "passed": passed,
            "failed": sum(r.status in ("failed", "error") for r in items),
            "errors": sum(r.status == "error" for r in items),
            "skipped": sum(r.status == "skipped" for r in items),
            "pass_rate": round(passed / len(scored), 3) if scored else None,
        }

    def avg(items: list[CaseResult], key: str) -> float | None:
        values = [r.metrics[key] for r in items if isinstance(r.metrics.get(key), (int, float))]
        return round(mean(values), 3) if values else None

    agent = [r for r in by_layer.get("agent", []) if r.status != "skipped"]
    check_rates: dict[str, float] = {}
    for name in sorted({c.name for r in agent for c in r.checks}):
        relevant = [c.passed for r in agent for c in r.checks if c.name == name]
        check_rates[name] = round(sum(relevant) / len(relevant), 3)

    return {
        "overall": counts(results),
        "by_layer": {layer: counts(items) for layer, items in sorted(by_layer.items())},
        "by_suite": {
            f"{r_layer}/{suite}": counts(items)
            for (r_layer, suite), items in sorted(_group(results).items())
        },
        "tools": {
            "mean_precision": avg(by_layer.get("tools", []), "precision"),
            "mean_recall": avg(by_layer.get("tools", []), "recall"),
        },
        "agent": {
            "mean_answer_precision": avg(agent, "answer_precision"),
            "mean_answer_recall": avg(agent, "answer_recall"),
            "first_draft_verified_rate": avg(agent, "first_draft_verified"),
            "mean_latency_s": avg(agent, "latency_s"),
            "mean_llm_calls": avg(agent, "llm_calls"),
            "check_pass_rates": check_rates,
        },
    }


def write_reports(
    results: list[CaseResult], summary: dict[str, Any], meta: dict[str, Any], out_dir: Path
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval_results.json"
    json_path.write_text(
        json.dumps(
            {"meta": meta, "summary": summary, "cases": [asdict(r) for r in results]},
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    md_path = out_dir / "eval_scorecard.md"
    md_path.write_text(render_markdown(results, summary, meta), encoding="utf-8")
    return md_path, json_path


def render_markdown(
    results: list[CaseResult], summary: dict[str, Any], meta: dict[str, Any]
) -> str:
    overall = summary["overall"]
    lines = [
        "# Fleet Copilot Eval Scorecard",
        "",
        f"- Run: `{meta['run_id']}` at {meta['started_at']}",
        f"- Mode: `{meta['mode']}`; model: `{meta.get('provider', '?')}/{meta['model']}`; "
        f"dataset as of {meta['as_of']}",
        f"- Overall: **{overall['passed']} passed, {overall['failed']} failed, "
        f"{overall['skipped']} skipped** (pass rate {_pct(overall['pass_rate'])})",
        "",
        "## By suite",
        "",
        markdown_table(
            ("Layer / suite", "Cases", "Passed", "Failed", "Skipped", "Pass rate"),
            [
                (name, c["cases"], c["passed"], c["failed"], c["skipped"], _pct(c["pass_rate"]))
                for name, c in summary["by_suite"].items()
            ],
        ),
        "",
        "## Key metrics",
        "",
        markdown_table(
            ("Metric", "Value"),
            [
                ("Tool device precision (mean)", summary["tools"]["mean_precision"]),
                ("Tool device recall (mean)", summary["tools"]["mean_recall"]),
                ("Agent answer precision (mean)", summary["agent"]["mean_answer_precision"]),
                ("Agent answer recall (mean)", summary["agent"]["mean_answer_recall"]),
                (
                    "Answers verified on first draft",
                    _pct(summary["agent"]["first_draft_verified_rate"]),
                ),
                ("Agent latency, s (mean)", summary["agent"]["mean_latency_s"]),
                ("LLM calls per question (mean)", summary["agent"]["mean_llm_calls"]),
            ],
        ),
    ]
    if summary["agent"]["check_pass_rates"]:
        lines += [
            "",
            "## Agent check pass rates",
            "",
            markdown_table(
                ("Check", "Pass rate"),
                [(k, _pct(v)) for k, v in summary["agent"]["check_pass_rates"].items()],
            ),
        ]
    lines += [
        "",
        "## Cases",
        "",
        markdown_table(
            ("Case", "Layer", "Suite", "Tenant", "Status", "Failed checks / note"),
            [(r.case_id, r.layer, r.suite, r.tenant, r.status, _problems(r)) for r in results],
        ),
        "",
    ]
    return "\n".join(lines)


def _group(results: list[CaseResult]) -> dict[tuple[str, str], list[CaseResult]]:
    grouped: dict[tuple[str, str], list[CaseResult]] = defaultdict(list)
    for r in results:
        grouped[(r.layer, r.suite)].append(r)
    return grouped


def _problems(result: CaseResult) -> str:
    failed = [f"{c.name}: {c.detail}" for c in result.checks if not c.passed]
    return "; ".join(failed) or result.note


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"
