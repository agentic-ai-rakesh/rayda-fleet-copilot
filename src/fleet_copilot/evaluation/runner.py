"""Run the eval suites and write the scorecard.

Modes:
- offline  tools + gate suites and guardrail cases that need no model. No API key.
- replay   also runs agent cases from recorded model responses (evals/recordings).
           Cases with no recording are skipped. No API key, fully deterministic.
- record   runs agent cases against Claude and saves every response for replay.
- live     runs agent cases against Claude without saving.

Each run writes to reports/eval_runs/<run_id>/ (its own audit log, action
store and checkpoints, so evals never touch the app's stores), and copies the
scorecard to reports/eval_scorecard.md.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fleet_copilot.config import PROJECT_ROOT, Settings
from fleet_copilot.db.connection import connect
from fleet_copilot.db.fleet_repository import FleetRepository
from fleet_copilot.evaluation.agent_suite import run_agent_case
from fleet_copilot.evaluation.cases import EvalCase, GateCase, load_cases
from fleet_copilot.evaluation.eval_context import EvalContext
from fleet_copilot.evaluation.gate_suite import run_gate_case
from fleet_copilot.evaluation.ground_truth import GroundTruth
from fleet_copilot.evaluation.metrics import CaseResult
from fleet_copilot.evaluation.scorecard import summarise, write_reports
from fleet_copilot.evaluation.tool_suite import run_tool_case
from fleet_copilot.guardrails.tenant_context import TenantContext
from fleet_copilot.llm import LLMClient, build_llm_client
from fleet_copilot.llm.recording_llm_client import RecordingLLMClient
from fleet_copilot.utils.logging_utils import set_run_id
from fleet_copilot.utils.time_utils import utc_now_iso

logger = logging.getLogger(__name__)

EvalMode = Literal["offline", "replay", "record", "live"]
Layer = Literal["tools", "gate", "agent"]
DEFAULT_CASES_DIR = PROJECT_ROOT / "evals" / "cases"
DEFAULT_RECORDINGS_DIR = PROJECT_ROOT / "evals" / "recordings"


@dataclass(frozen=True)
class EvalRun:
    results: list[CaseResult]
    summary: dict
    scorecard_path: Path
    results_path: Path


def run_evals(
    settings: Settings,
    mode: EvalMode = "replay",
    layers: tuple[Layer, ...] = ("tools", "gate", "agent"),
    case_filter: str | None = None,
    cases_dir: Path = DEFAULT_CASES_DIR,
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR,
) -> EvalRun:
    run_id = f"eval-{utc_now_iso().replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:6]}"
    set_run_id(run_id)
    run_dir = settings.paths.reports_dir / "eval_runs" / run_id
    ctx = EvalContext(
        settings=_isolated_settings(settings, run_dir),
        ground_truth=GroundTruth(settings.paths.raw_dataset, settings),
        run_dir=run_dir,
    )

    cases, gate_cases = load_cases(cases_dir)
    if case_filter:
        cases = [c for c in cases if case_filter in c.id]
        gate_cases = [c for c in gate_cases if case_filter in c.id]

    llm_factory = _llm_factory(settings, mode, recordings_dir)
    results: list[CaseResult] = []
    if "tools" in layers:
        results += [_safely(run_tool_case, c, ctx) for c in cases if c.reference_tool]
    if "gate" in layers:
        results += [_safely(run_gate_case, c, ctx) for c in gate_cases]
    if "agent" in layers:

        def run_agent(case: EvalCase, context: EvalContext) -> CaseResult:
            return run_agent_case(case, context, llm_factory)

        results += [_safely(run_agent, c, ctx) for c in cases if c.question]

    summary = summarise(results)
    meta = {
        "run_id": run_id,
        "started_at": utc_now_iso(),
        "mode": mode,
        "provider": settings.llm.provider if mode != "offline" else "none",
        "model": settings.llm.model if mode != "offline" else "none",
        "as_of": _as_of(settings),
        "layers": list(layers),
        "case_filter": case_filter,
    }
    scorecard, results_json = write_reports(results, summary, meta, run_dir)
    shutil.copy(scorecard, settings.paths.reports_dir / "eval_scorecard.md")
    shutil.copy(results_json, settings.paths.reports_dir / "eval_results.json")
    return EvalRun(results, summary, scorecard, results_json)


def _isolated_settings(settings: Settings, run_dir: Path) -> Settings:
    isolated = settings.model_copy(deep=True)
    isolated.paths.audit_db = run_dir / "audit.db"
    isolated.paths.actions_db = run_dir / "actions.db"
    isolated.paths.checkpoint_db = run_dir / "checkpoints.db"
    return isolated


def _llm_factory(
    settings: Settings, mode: EvalMode, recordings_dir: Path
) -> Callable[[], LLMClient | None]:
    if mode == "offline":
        return lambda: None
    if mode == "replay":
        return lambda: RecordingLLMClient(recordings_dir, "replay", settings.llm.model)
    real = build_llm_client(settings.llm)  # fails early if no API key
    if mode == "record":
        return lambda: RecordingLLMClient(recordings_dir, "record", settings.llm.model, inner=real)
    return lambda: real


def _safely(run: Callable, case: EvalCase | GateCase, ctx: EvalContext) -> CaseResult:
    """One broken case must not stop the run; it is reported as an error."""
    try:
        return run(case, ctx)
    except Exception as exc:
        logger.exception("Eval case %s crashed", case.id)
        layer = "gate" if isinstance(case, GateCase) else "agent"
        suite = "gate_redteam" if isinstance(case, GateCase) else case.suite
        return CaseResult(case.id, layer, suite, case.tenant).fail_with_error(
            f"{type(exc).__name__}: {exc}"
        )


def _as_of(settings: Settings) -> str:
    with closing(connect(settings.paths.fleet_db, read_only=True)) as conn:
        return FleetRepository(conn, TenantContext("-", "-", "eval")).as_of() or "?"
