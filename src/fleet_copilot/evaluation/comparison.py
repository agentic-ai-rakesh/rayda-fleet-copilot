"""Compare eval runs across providers and models.

Reads every reports/eval_runs/*/eval_results.json and keeps one run per
(provider, model): the latest run that called the model (record or live),
or the latest replay if there is none. Latency is only reported for runs that
called the model; a replay's timing says nothing about the model. Runs with
skipped agent cases or provider errors are flagged, so a partial run is never
compared as if it were complete.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fleet_copilot.utils.markdown_utils import markdown_table


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(runs_dir.glob("*/eval_results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        agent = data["summary"]["by_layer"].get("agent")
        if data["meta"].get("mode") == "offline" or not agent:
            continue
        runs.append(data)
    return runs


LIVE_MODES = ("record", "live")


def latest_per_model(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One run per model: prefer runs that called the model, then the most recent."""
    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    ordered = sorted(runs, key=lambda r: (r["meta"]["mode"] in LIVE_MODES, r["meta"]["started_at"]))
    for run in ordered:
        meta = run["meta"]
        chosen[(meta.get("provider", "?"), meta.get("model", "?"))] = run
    return [chosen[key] for key in sorted(chosen)]


def comparison_row(run: dict[str, Any]) -> dict[str, str]:
    meta, summary = run["meta"], run["summary"]
    agent_layer = summary["by_layer"]["agent"]
    agent = summary["agent"]
    checks = agent.get("check_pass_rates", {})
    passed = f"{agent_layer['passed']}/{agent_layer['cases'] - agent_layer['skipped']}"
    if agent_layer["skipped"]:
        passed += f" ({agent_layer['skipped']} skipped)"
    if agent_layer.get("errors"):
        passed += f" ({agent_layer['errors']} provider errors)"
    live = meta["mode"] in LIVE_MODES
    return {
        "Provider": meta.get("provider", "?"),
        "Model": meta.get("model", "?"),
        "Mode": meta["mode"],
        "Agent passed": passed,
        "Pass rate": _pct(agent_layer["pass_rate"]),
        "Answer recall": _num(agent.get("mean_answer_recall")),
        "Answer precision": _num(agent.get("mean_answer_precision")),
        "Verified 1st draft": _pct(agent.get("first_draft_verified_rate")),
        "Right tools": _pct(checks.get("tool_selection")),
        "Proposals justified": _pct(checks.get("proposals_justified")),
        "LLM calls / q": _num(agent.get("mean_llm_calls")),
        "Latency s": _num(agent.get("mean_latency_s")) if live else "— (replay)",
        "Run": meta["run_id"],
    }


def render_comparison(runs: list[dict[str, Any]]) -> str:
    rows = [comparison_row(run) for run in runs]
    if not rows:
        return (
            "# Provider comparison\n\nNo agent eval runs yet. Record one per model, e.g.\n"
            "`fleet-copilot --provider openai --model <model> eval --mode record`.\n"
        )
    return "\n".join(
        [
            "# Provider comparison",
            "",
            "One agent eval run per provider and model (the latest that called the model,",
            "else the latest replay). The tool and gate suites do not depend on the model",
            "and are the same for every row. Latency is shown only for runs that called",
            "the model.",
            "",
            markdown_table(tuple(rows[0]), [tuple(r.values()) for r in rows]),
            "",
        ]
    )


def write_comparison(runs_dir: Path, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_comparison(latest_per_model(load_runs(runs_dir))), encoding="utf-8")
    return output


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:g}"
