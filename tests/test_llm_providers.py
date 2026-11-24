"""Provider switch: settings, client construction and request shape (no network)."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from fleet_copilot.config import load_settings, with_llm_overrides
from fleet_copilot.config.settings import PROJECT_ROOT, LLMSettings
from fleet_copilot.evaluation.comparison import (
    comparison_row,
    latest_per_model,
    load_runs,
    render_comparison,
)
from fleet_copilot.exceptions import ConfigError, LLMError, LLMUnavailableError
from fleet_copilot.graphs.nodes.node_utils import tool_definitions
from fleet_copilot.llm import build_llm_client
from fleet_copilot.llm.llm_client import LangChainLLMClient, _raise_on_refusal, refusal_reason

KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}
MODELS = {"anthropic": "claude-opus-5-5", "openai": "test-gpt-model", "google": "test-gemini-model"}


@pytest.fixture
def settings():
    return load_settings(PROJECT_ROOT / "config" / "settings.yaml")


@pytest.fixture(autouse=True)
def no_real_keys(monkeypatch):
    for key in KEYS.values():
        monkeypatch.delenv(key, raising=False)


# ----------------------------------------------------------------- settings
def test_configured_default_is_gpt_and_builtin_default_is_claude(settings):
    assert (settings.llm.provider, settings.llm.model) == ("openai", "gpt-5.5")
    assert (LLMSettings().provider, LLMSettings().model) == ("anthropic", "claude-opus-5-5")


def test_switching_provider_requires_a_model(settings):
    with pytest.raises(ConfigError, match="--model"):
        with_llm_overrides(settings, provider="google")
    switched = with_llm_overrides(settings, provider="google", model="test-gemini-model")
    assert (switched.llm.provider, switched.llm.model) == ("google", "test-gemini-model")
    assert settings.llm.provider == "openai"  # original untouched


def test_claude_model_with_other_provider_is_rejected():
    with pytest.raises(ValueError, match="Claude model"):
        LLMSettings(provider="google", model="claude-opus-5-5")


def test_unknown_provider_is_rejected(settings):
    with pytest.raises(ConfigError):
        with_llm_overrides(settings, provider="mistral", model="m")


# ------------------------------------------------------------ construction
@pytest.mark.parametrize("provider", ["anthropic", "openai", "google"])
def test_missing_key_gives_a_clear_error(settings, provider):
    llm = with_llm_overrides(settings, provider, MODELS[provider]).llm
    with pytest.raises(LLMUnavailableError, match=KEYS[provider]):
        build_llm_client(llm)


def _client(settings, monkeypatch, provider, **llm_changes):
    monkeypatch.setenv(KEYS[provider], "test-key")
    llm = with_llm_overrides(settings, provider, MODELS[provider]).llm
    llm = llm.model_copy(update=llm_changes)
    client = build_llm_client(llm)
    assert isinstance(client, LangChainLLMClient)
    return client._model


def test_claude_request_has_effort_fallback_and_no_temperature(settings, monkeypatch):
    model = _client(settings, monkeypatch, "anthropic", temperature=0.0)
    payload = model._get_request_payload([HumanMessage("q")])
    assert payload["model"] == "claude-opus-5-5"
    assert payload["output_config"] == {"effort": "medium"}
    assert payload["fallbacks"] == "default"
    assert "temperature" not in payload  # ignored for Claude even when configured


def test_openai_client_options(settings, monkeypatch):
    model = _client(settings, monkeypatch, "openai", temperature=0.0, reasoning_effort="low")
    assert model.model_name == "test-gpt-model"
    assert (model.temperature, model.reasoning_effort, model.max_tokens) == (0.0, "low", 16000)


def test_gemini_client_options(settings, monkeypatch):
    model = _client(settings, monkeypatch, "google", temperature=0.0)
    assert model.model.endswith("test-gemini-model")
    assert (model.temperature, model.max_output_tokens) == (0.0, 16000)


@pytest.mark.parametrize("provider", ["anthropic", "openai", "google"])
def test_every_provider_accepts_the_tool_definitions(settings, monkeypatch, provider):
    model = _client(settings, monkeypatch, provider)
    bound = model.bind_tools(tool_definitions())
    names = json.dumps(bound.kwargs["tools"])
    assert "detect_battery_end_of_life" in names
    assert "flag_device_for_replacement" not in names  # action tools are never bound


def test_missing_integration_package_gives_install_hint(settings, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(__import__("sys").modules, "langchain_openai", None)
    llm = with_llm_overrides(settings, "openai", "test-gpt-model").llm
    with pytest.raises(LLMUnavailableError, match=r"\.\[openai\]"):
        build_llm_client(llm)


# ----------------------------------------------------------------- refusals
@pytest.mark.parametrize(
    ("message", "reason"),
    [
        (
            AIMessage(
                "",
                response_metadata={"stop_reason": "refusal", "stop_details": {"category": "cyber"}},
            ),
            "cyber",
        ),
        (AIMessage("", additional_kwargs={"refusal": "I can't help"}), "refusal"),
        (AIMessage("", response_metadata={"finish_reason": "content_filter"}), "content_filter"),
        (AIMessage("", response_metadata={"finish_reason": "SAFETY"}), "SAFETY"),
        (AIMessage("ok", response_metadata={"finish_reason": "STOP"}), None),
        (AIMessage("ok", response_metadata={"stop_reason": "end_turn"}), None),
    ],
)
def test_refusals_are_detected_for_every_provider(message, reason):
    assert refusal_reason(message) == reason
    if reason:
        with pytest.raises(LLMError, match="declined"):
            _raise_on_refusal("planner", message)


# --------------------------------------------------------------- comparison
def _run(tmp_path, run_id, provider, model, started, passed, skipped=0, mode="replay"):
    folder = tmp_path / run_id
    folder.mkdir()
    agent = {
        "cases": 4,
        "passed": passed,
        "failed": 4 - passed - skipped,
        "skipped": skipped,
        "pass_rate": passed / (4 - skipped),
    }
    data = {
        "meta": {
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "mode": mode,
            "started_at": started,
        },
        "summary": {
            "by_layer": {"agent": agent},
            "agent": {
                "mean_answer_recall": 1.0,
                "mean_latency_s": 20.0 if mode == "record" else 0.1,
                "check_pass_rates": {},
            },
        },
    }
    (folder / "eval_results.json").write_text(json.dumps(data))


def test_comparison_keeps_latest_run_per_model(tmp_path):
    _run(tmp_path, "r1", "anthropic", "claude-opus-5-5", "2026-01-01T00:00:00Z", 2)
    _run(tmp_path, "r2", "anthropic", "claude-opus-5-5", "2026-01-02T00:00:00Z", 4)
    _run(tmp_path, "r3", "openai", "test-gpt-model", "2026-01-02T00:00:00Z", 3, skipped=1)
    runs = latest_per_model(load_runs(tmp_path))
    assert [r["meta"]["run_id"] for r in runs] == ["r2", "r3"]
    table = render_comparison(runs)
    assert "4/4" in table and "3/3 (1 skipped)" in table


def test_comparison_without_runs_explains_how_to_record(tmp_path):
    assert "--mode record" in render_comparison(latest_per_model(load_runs(tmp_path)))


def test_comparison_prefers_a_run_that_called_the_model(tmp_path):
    _run(tmp_path, "rec", "openai", "gpt-x", "2026-01-01T00:00:00Z", 4, mode="record")
    _run(tmp_path, "rep", "openai", "gpt-x", "2026-01-02T00:00:00Z", 4, mode="replay")
    (run,) = latest_per_model(load_runs(tmp_path))
    assert run["meta"]["run_id"] == "rec"  # a newer replay does not replace the recording
    assert comparison_row(run)["Latency s"] == "20"


def test_replay_latency_is_not_reported(tmp_path):
    _run(tmp_path, "rep", "openai", "gpt-x", "2026-01-02T00:00:00Z", 4, mode="replay")
    (run,) = latest_per_model(load_runs(tmp_path))
    assert comparison_row(run)["Latency s"] == "— (replay)"
