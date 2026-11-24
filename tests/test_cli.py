"""CLI smoke tests: the commands a reviewer runs, end to end, without any model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from fleet_copilot import cli


@pytest.fixture
def run_cli(fleet_settings, tmp_path, monkeypatch):
    """Invoke `fleet-copilot` with a settings file whose outputs go to tmp_path."""
    monkeypatch.setattr(cli, "load_dotenv", lambda *a, **k: None)  # never load real keys
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    config = {
        "paths": {
            "raw_dataset": str(fleet_settings.paths.raw_dataset),
            "fleet_db": str(fleet_settings.paths.fleet_db),
            "audit_db": str(tmp_path / "audit.db"),
            "actions_db": str(tmp_path / "actions.db"),
            "checkpoint_db": str(tmp_path / "checkpoints.db"),
            "quarantine_file": str(tmp_path / "quarantine.ndjson"),
            "log_dir": str(tmp_path / "logs"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "llm": {"provider": "openai", "model": "gpt-5.5"},
    }
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.safe_dump(config))
    runner = CliRunner()

    def invoke(*args: str):
        return runner.invoke(cli.app, ["--config", str(config_path), *args])

    invoke.tmp_path = tmp_path  # type: ignore[attr-defined]
    return invoke


def test_tools_list_shows_every_tool(run_cli):
    result = run_cli("tools", "list")
    assert result.exit_code == 0, result.output
    for name in ("find_devices_by_metric", "detect_battery_end_of_life"):
        assert name in result.output


def test_tools_run_returns_grounded_json(run_cli):
    args = json.dumps({"metric": "disk_free_pct", "comparison": "lt", "value": 15})
    result = run_cli("tools", "run", "find_devices_by_metric", "--tenant", "acme", "--args", args)
    assert result.exit_code == 0, result.output
    assert "MT7PJB7N5LRE" in result.output and "8NM23J95R5I6" in result.output
    assert '"tenant_id": "acme-001"' in result.output


def test_tools_run_with_bad_json_or_unknown_tenant_fails_cleanly(run_cli):
    bad_json = run_cli("tools", "run", "list_devices", "--tenant", "acme", "--args", "{oops")
    unknown = run_cli("tools", "run", "list_devices", "--tenant", "umbrella")
    assert bad_json.exit_code == 1 and "not valid JSON" in bad_json.output
    assert unknown.exit_code == 1 and "Unknown company" in unknown.output


def test_ask_without_api_key_explains_what_to_set(run_cli):
    result = run_cli("ask", "Which devices are low on disk?", "--tenant", "acme")
    assert result.exit_code == 1
    assert "OPENAI_API_KEY" in result.output


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--provider", "google"], "--model"),
        (["--provider", "google", "--model", "claude-opus-5-5"], "Claude model"),
    ],
)
def test_invalid_provider_overrides_are_refused(run_cli, flags, message):
    result = run_cli(*flags, "tools", "list")
    assert result.exit_code == 1
    assert message in result.output.replace("\n", " ")


def test_offline_eval_writes_a_passing_scorecard(run_cli):
    result = run_cli("eval", "--mode", "offline", "--layer", "tools", "--layer", "gate")
    assert result.exit_code == 0, result.output
    scorecard = Path(run_cli.tmp_path) / "reports" / "eval_scorecard.md"
    assert scorecard.exists()
    assert "27 passed, 0 failed, 0 skipped" in scorecard.read_text()


def test_audit_verify_after_tool_calls(run_cli):
    run_cli("tools", "run", "list_devices", "--tenant", "globex")
    result = run_cli("audit-verify")
    assert result.exit_code == 0
    assert "intact" in result.output
