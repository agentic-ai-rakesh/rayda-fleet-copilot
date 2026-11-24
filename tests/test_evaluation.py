"""The eval harness: cases are valid, ground truth is right, and scoring catches mistakes."""

from __future__ import annotations

import pytest

from fleet_copilot.evaluation.agent_suite import run_agent_case
from fleet_copilot.evaluation.cases import EvalCase, Expectation, GateCase, ToolCallSpec, load_cases
from fleet_copilot.evaluation.eval_context import EvalContext
from fleet_copilot.evaluation.gate_suite import run_gate_case
from fleet_copilot.evaluation.ground_truth import GroundTruth
from fleet_copilot.evaluation.metrics import contains_number, mentioned_devices, precision_recall
from fleet_copilot.evaluation.runner import DEFAULT_CASES_DIR, run_evals
from fleet_copilot.evaluation.tool_suite import run_tool_case
from fleet_copilot.llm.recording_llm_client import RecordingLLMClient, ReplayMissError
from fleet_copilot.prompts.output_schemas import Plan
from fleet_copilot.tools import TOOL_REGISTRY
from tests.conftest import REAL_DATASET
from tests.scripted_llm import DONE, ScriptedLLM, answer_citing, plan, tool_call

LOW_DISK = ToolCallSpec(
    name="find_devices_by_metric", args={"metric": "disk_free_pct", "comparison": "lt", "value": 15}
)


@pytest.fixture
def ctx(fleet_settings, tmp_path) -> EvalContext:
    settings = fleet_settings.model_copy(deep=True)
    settings.paths.audit_db = tmp_path / "audit.db"
    settings.paths.actions_db = tmp_path / "actions.db"
    settings.paths.checkpoint_db = tmp_path / "checkpoints.db"
    return EvalContext(settings, GroundTruth(REAL_DATASET, settings), tmp_path)


# ------------------------------------------------------------------ cases
def test_case_files_are_valid_and_consistent(ctx):
    cases, gate_cases = load_cases(DEFAULT_CASES_DIR)
    assert len(cases) >= 25 and len(gate_cases) >= 10
    for case in cases:
        if case.reference_tool:
            assert case.reference_tool.name in TOOL_REGISTRY, case.id
        assert set(case.expect.tools_any) <= set(TOOL_REGISTRY), case.id
        if case.expect.truth:
            ctx.ground_truth.resolve(case.expect.truth, ctx.tenant(case.tenant).company_id)
        if case.expect.proposals:
            ctx.ground_truth.resolve(case.expect.proposals.allowed_targets, case.tenant)
    suites = {c.suite for c in cases}
    assert suites == {"grounded_qa", "insight", "action", "guardrail"}


def test_ground_truth_matches_known_facts(ctx):
    gt = ctx.ground_truth
    assert gt.low_disk_devices("acme-001") == {"MT7PJB7N5LRE", "8NM23J95R5I6"}
    assert gt.high_severity_failing_laptops("acme-001") == set()
    assert gt.macos_older_than_15_count("globex-002") == 2
    assert gt.battery_eol_devices("acme-001") == {"7VP16KHV88LM", "8NM23J95R5I6", "M4XVHUV1MEPZ"}
    assert gt.resolve("software_devices:uTorrent", "initech-003") == {
        "I3JLKILKQTAC",
        "W2656QDAE2ZJ",
    }
    with pytest.raises(KeyError):
        gt.resolve("_records", "acme-001")


# ------------------------------------------------------------ scoring
def test_metric_helpers():
    assert precision_recall({"A", "B"}, {"A"}) == (0.5, 1.0)
    assert precision_recall(set(), set()) == (1.0, 1.0)
    text = "MT7PJB7N5LRE is low [8NM23J95R5I6@20260612T0902Z]; 4 devices"
    assert mentioned_devices(text, {"MT7PJB7N5LRE", "8NM23J95R5I6"}) == {"MT7PJB7N5LRE"}
    assert contains_number(text, 4) and not contains_number(text, 2026)


def test_tool_suite_passes_and_catches_a_wrong_tool(ctx):
    right = EvalCase(
        id="ok",
        suite="grounded_qa",
        tenant="acme-001",
        reference_tool=LOW_DISK,
        expect=Expectation(truth="low_disk_devices"),
    )
    wrong = right.model_copy(
        update={"id": "bad", "expect": Expectation(truth="battery_eol_devices")}
    )
    assert run_tool_case(right, ctx).status == "passed"
    failed = run_tool_case(wrong, ctx)
    assert failed.status == "failed"
    assert any(c.name == "devices_correct" and not c.passed for c in failed.checks)


def test_gate_suite_catches_a_wrong_expectation(ctx):
    _, gate_cases = load_cases(DEFAULT_CASES_DIR)
    allowed = next(c for c in gate_cases if c.id == "gate_replace_worn_battery_allowed")
    assert run_gate_case(allowed, ctx).status == "passed"
    flipped = GateCase.model_validate({**allowed.model_dump(), "expect_allowed": False})
    assert run_gate_case(flipped, ctx).status == "failed"


def _low_disk_llm(devices: list[str]) -> ScriptedLLM:
    return ScriptedLLM(
        planner=[plan("question", ["find_devices_by_metric"])],
        executor=[tool_call("find_devices_by_metric", LOW_DISK.args), DONE],
        responder=[answer_citing(devices, "is low on disk")],
    )


LOW_DISK_CASE = EvalCase(
    id="qa",
    suite="grounded_qa",
    tenant="acme-001",
    question="Which devices at Acme are low on disk space?",
    expect=Expectation(truth="low_disk_devices", tools_any=["find_devices_by_metric"]),
)


def test_agent_suite_passes_a_correct_answer(ctx):
    result = run_agent_case(
        LOW_DISK_CASE, ctx, lambda: _low_disk_llm(["MT7PJB7N5LRE", "8NM23J95R5I6"])
    )
    assert result.status == "passed", result.checks
    assert result.metrics["answer_recall"] == 1.0


def test_agent_suite_fails_an_incomplete_answer(ctx):
    result = run_agent_case(LOW_DISK_CASE, ctx, lambda: _low_disk_llm(["MT7PJB7N5LRE"]))
    assert result.status == "failed"
    assert result.metrics["answer_recall"] == 0.5


def test_agent_case_needing_a_model_is_skipped_offline(ctx):
    assert run_agent_case(LOW_DISK_CASE, ctx, lambda: None).status == "skipped"


# ------------------------------------------------------------ record/replay
def test_record_then_replay_is_identical_and_needs_no_model(ctx, tmp_path):
    recordings = tmp_path / "recordings"
    recorder = RecordingLLMClient(
        recordings, "record", "test-model", inner=_low_disk_llm(["MT7PJB7N5LRE", "8NM23J95R5I6"])
    )
    recorded = run_agent_case(LOW_DISK_CASE, ctx, lambda: recorder)
    assert recorded.status == "passed"
    assert recorder.misses == 4 and len(list(recordings.glob("*.json"))) == 4

    replayer = RecordingLLMClient(recordings, "replay", "test-model")
    replayed = run_agent_case(LOW_DISK_CASE, ctx, lambda: replayer)
    assert replayed.status == "passed"
    assert (replayer.hits, replayer.misses) == (4, 0)
    assert replayed.metrics["response"] == recorded.metrics["response"]


def test_replay_miss_raises_and_marks_case_skipped(ctx, tmp_path):
    replayer = RecordingLLMClient(tmp_path / "empty", "replay", "test-model")
    with pytest.raises(ReplayMissError):
        replayer.structured([], Plan, purpose="planner")
    assert (
        run_agent_case(
            LOW_DISK_CASE,
            ctx,
            lambda: RecordingLLMClient(tmp_path / "empty", "replay", "test-model"),
        ).status
        == "skipped"
    )


# ------------------------------------------------------------ runner
def test_offline_run_passes_every_scored_case(fleet_settings, tmp_path):
    settings = fleet_settings.model_copy(deep=True)
    settings.paths.reports_dir = tmp_path / "reports"
    run = run_evals(settings, mode="offline")
    overall = run.summary["overall"]
    assert overall["failed"] == 0, [
        r for r in run.results if r.status != "passed" and r.status != "skipped"
    ]
    assert overall["passed"] >= 30
    assert (
        run.scorecard_path.exists() and (settings.paths.reports_dir / "eval_scorecard.md").exists()
    )
    guard = [r for r in run.results if r.suite == "guardrail" and r.status == "passed"]
    assert len(guard) >= 5 and all(r.metrics["llm_calls"] == 0 for r in guard)


def test_says_none_detects_negative_answers():
    from fleet_copilot.evaluation.metrics import says_none

    assert says_none("None. 0 of 5 laptops fail [A@20260612T0000Z]")
    assert says_none("No laptops fail a high-severity check.")
    assert not says_none("3 devices fail screen_lock.")


def test_empty_truth_case_passes_when_answer_lists_passing_evidence(ctx):
    case = EvalCase(
        id="none",
        suite="grounded_qa",
        tenant="initech-003",
        question="Are any laptops failing a high-severity compliance check?",
        expect=Expectation(truth="high_severity_failing_laptops"),
    )
    llm = ScriptedLLM(
        planner=[plan("question", ["get_compliance_status"])],
        executor=[
            tool_call("get_compliance_status", {"severity": "high", "device_type": "laptop"}),
            DONE,
        ],
        responder=[answer_citing(["2MAN6T4DJD95", "DARFCP0BQM2G"], "passes; none fail")],
    )
    result = run_agent_case(case, ctx, lambda: llm)
    assert result.status == "passed", result.checks


def test_provider_failure_is_an_error_not_a_wrong_answer(ctx):
    from fleet_copilot.exceptions import LLMError

    llm = ScriptedLLM(planner=[LLMError("planner: 429 quota exceeded")])
    result = run_agent_case(LOW_DISK_CASE, ctx, lambda: llm)
    assert result.status == "error"
    assert "model call failed" in result.note
