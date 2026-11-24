"""Settings loaded from config/settings.yaml, with safe defaults.

Fallback policy:
- Settings file missing  -> run on the built-in defaults and log a warning.
- Settings file invalid  -> raise ConfigError. Running on silently "fixed"
  settings could change thresholds or paths without anyone noticing.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from fleet_copilot.exceptions import ConfigError

logger = logging.getLogger(__name__)

# src/fleet_copilot/config/settings.py -> project root is three levels above the package.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
CONFIG_PATH_ENV_VAR = "FLEET_COPILOT_CONFIG"


class _StrictModel(BaseModel):
    """Reject unknown keys so typos in settings.yaml fail loudly."""

    model_config = ConfigDict(extra="forbid")


class PathSettings(_StrictModel):
    raw_dataset: Path = Path("data/raw/device-telemetry-dataset.ndjson")
    fleet_db: Path = Path("data/processed/fleet.db")
    audit_db: Path = Path("data/audit/audit.db")
    quarantine_file: Path = Path("data/quarantine/rejected_records.ndjson")
    actions_db: Path = Path("data/actions/actions.db")
    checkpoint_db: Path = Path("data/checkpoints/checkpoints.db")
    log_dir: Path = Path("logs")
    reports_dir: Path = Path("reports")

    def resolve_from(self, root: Path) -> PathSettings:
        """Return a copy with every relative path anchored at `root`."""
        resolved = {name: value if value.is_absolute() else root / value for name, value in self}
        return PathSettings(**resolved)


class IngestionSettings(_StrictModel):
    max_reject_ratio: float = Field(default=0.05, ge=0.0, le=1.0)


class DeviceClassificationSettings(_StrictModel):
    desktop_models: list[str] = Field(default_factory=lambda: ["Mac mini"])


class LoggingSettings(_StrictModel):
    level: str = "INFO"
    log_file_name: str = "fleet_copilot.log"
    max_bytes: int = Field(default=5_000_000, gt=0)
    backup_count: int = Field(default=3, ge=0)


class ThresholdSettings(_StrictModel):
    """What counts as a problem. Every tool result states the values it used."""

    low_disk_free_pct: float = Field(default=15.0, gt=0, lt=100)
    critical_disk_free_pct: float = Field(default=5.0, gt=0, lt=100)
    high_memory_used_pct: float = Field(default=85.0, gt=0, le=100)
    battery_cycle_limit: int = Field(default=1000, gt=0)
    battery_cycle_warning_ratio: float = Field(default=0.8, gt=0, le=1)
    battery_capacity_drop_pct: float = Field(default=3.0, gt=0, lt=100)
    battery_service_conditions: list[str] = Field(
        default_factory=lambda: ["Service Recommended", "Replace Soon", "Replace Now"]
    )


class InsightSettings(_StrictModel):
    """How trends are judged across the time series."""

    consistent_share: float = Field(default=0.7, gt=0, le=1)
    intermittent_share: float = Field(default=0.3, gt=0, le=1)
    min_snapshots: int = Field(default=7, ge=2)


class ToolSettings(_StrictModel):
    max_rows: int = Field(default=50, gt=0)
    default_history_days: int = Field(default=30, gt=0)


LLMProvider = Literal["anthropic", "openai", "google"]


class LLMSettings(_StrictModel):
    """Language model used by the agent. Switch with `provider` + `model` (or CLI flags)."""

    provider: LLMProvider = "anthropic"
    model: str = "claude-opus-5-5"
    max_tokens: int = Field(default=16_000, gt=0)
    timeout_seconds: float = Field(default=120, gt=0)
    max_retries: int = Field(default=2, ge=0)
    # Claude only
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    server_side_fallback: bool = True  # retry a refused request on a fallback model
    # OpenAI / Gemini only (current Claude models reject sampling parameters)
    temperature: float | None = Field(default=None, ge=0, le=2)
    reasoning_effort: str | None = None  # passed through for reasoning models

    @model_validator(mode="after")
    def _model_matches_provider(self) -> LLMSettings:
        if self.provider != "anthropic" and self.model.lower().startswith("claude"):
            raise ValueError(
                f"llm.model is '{self.model}', a Claude model, but provider is "
                f"'{self.provider}'. Set llm.model (or --model) to a {self.provider} model."
            )
        return self


class AgentSettings(_StrictModel):
    max_tool_iterations: int = Field(default=8, ge=1)
    max_answer_retries: int = Field(default=1, ge=0)
    max_question_chars: int = Field(default=2000, gt=0)
    max_notification_chars: int = Field(default=500, gt=0)


class Settings(_StrictModel):
    paths: PathSettings = Field(default_factory=PathSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    device_classification: DeviceClassificationSettings = Field(
        default_factory=DeviceClassificationSettings
    )
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    insights: InsightSettings = Field(default_factory=InsightSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)


def load_settings(config_path: Path | None = None, project_root: Path = PROJECT_ROOT) -> Settings:
    """Load settings from `config_path`, the env override, or the default file."""
    path = config_path or Path(os.environ.get(CONFIG_PATH_ENV_VAR, DEFAULT_CONFIG_PATH))
    raw = _read_yaml(path)
    try:
        settings = Settings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid settings in {path}:\n{exc}") from exc
    settings.paths = settings.paths.resolve_from(project_root)
    return settings


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("Settings file %s not found; using built-in defaults.", path)
        return {}
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Settings file {path} is not valid YAML: {exc}") from exc
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ConfigError(f"Settings file {path} must contain a mapping at the top level.")
    return content


def with_llm_overrides(
    settings: Settings, provider: str | None = None, model: str | None = None
) -> Settings:
    """Settings with the LLM provider/model replaced, validated like the YAML file."""
    if provider is None and model is None:
        return settings
    llm = settings.llm.model_dump()
    if provider is not None:
        llm["provider"] = provider
        if model is None and provider != settings.llm.provider:
            llm["model"] = ""  # force an explicit model for the new provider
    if model is not None:
        llm["model"] = model
    if not llm["model"]:
        raise ConfigError(f"Choose a model for provider '{llm['provider']}' with --model.")
    try:
        validated = LLMSettings.model_validate(llm)
    except ValidationError as exc:
        raise ConfigError(f"Invalid LLM settings: {exc.errors()[0]['msg']}") from exc
    return settings.model_copy(update={"llm": validated})
