"""Build the configured LLM client: Claude (default), OpenAI GPT or Google Gemini.

All three go through LangChain chat models behind the same LLMClient
interface, so the graph, prompts, tools and guardrails do not change with the
provider. Provider-specific choices live only here.

Claude (current models):
- No `temperature`: sampling parameters are rejected, so reproducibility comes
  from deterministic tools, structured outputs and recorded responses.
- Reasoning depth via `output_config.effort`; thinking stays adaptive.
- Structured output in JSON-schema mode (forced tool choice is unsupported).
- Server-side fallback: a declined request is retried on a fallback model.

OpenAI / Gemini:
- Optional `temperature` (e.g. 0) and `reasoning_effort` pass-through.
- Installed as extras: pip install -e ".[openai]" / ".[google]".
"""

from __future__ import annotations

import logging
import os
from importlib import import_module
from typing import Any

from fleet_copilot.config.settings import LLMSettings
from fleet_copilot.exceptions import LLMUnavailableError
from fleet_copilot.llm.llm_client import LangChainLLMClient, LLMClient

logger = logging.getLogger(__name__)

SERVER_SIDE_FALLBACK_BETA = "server-side-fallback-2026-07-01"

# provider -> (API key env var, module, class, pip extra)
PROVIDERS: dict[str, tuple[str, str, str, str]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "langchain_anthropic", "ChatAnthropic", ""),
    "openai": ("OPENAI_API_KEY", "langchain_openai", "ChatOpenAI", "openai"),
    "google": ("GOOGLE_API_KEY", "langchain_google_genai", "ChatGoogleGenerativeAI", "google"),
}


def build_llm_client(settings: LLMSettings) -> LLMClient:
    if settings.provider not in PROVIDERS:
        raise LLMUnavailableError(f"Unsupported LLM provider '{settings.provider}'.")
    key_env, module_name, class_name, extra = PROVIDERS[settings.provider]
    if not os.environ.get(key_env):
        raise LLMUnavailableError(f"{key_env} is not set. Add it to .env (see .env.example).")
    try:
        chat_class = getattr(import_module(module_name), class_name)
    except ImportError as exc:
        raise LLMUnavailableError(
            f'The {settings.provider} integration is not installed: pip install -e ".[{extra}]"'
        ) from exc

    options = _PROVIDER_OPTIONS[settings.provider](settings)
    logger.info("LLM: %s / %s", settings.provider, settings.model)
    return LangChainLLMClient(chat_class(**options), structured_method="json_schema")


def _anthropic_options(settings: LLMSettings) -> dict[str, Any]:
    if settings.temperature is not None:
        logger.warning("llm.temperature is ignored for Claude: current models reject it.")
    options: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "max_retries": settings.max_retries,
        "default_request_timeout": settings.timeout_seconds,
        "output_config": {"effort": settings.effort},
    }
    if settings.server_side_fallback:
        options["betas"] = [SERVER_SIDE_FALLBACK_BETA]
        options["model_kwargs"] = {"fallbacks": "default"}
    return options


def _openai_options(settings: LLMSettings) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "max_retries": settings.max_retries,
        "timeout": settings.timeout_seconds,
    }
    if settings.temperature is not None:
        options["temperature"] = settings.temperature
    if settings.reasoning_effort:
        options["reasoning_effort"] = settings.reasoning_effort
    return options


def _google_options(settings: LLMSettings) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model": settings.model,
        "max_output_tokens": settings.max_tokens,
        "max_retries": settings.max_retries,
        "timeout": settings.timeout_seconds,
    }
    if settings.temperature is not None:
        options["temperature"] = settings.temperature
    if settings.reasoning_effort:
        options["reasoning_effort"] = settings.reasoning_effort
    return options


_PROVIDER_OPTIONS = {
    "anthropic": _anthropic_options,
    "openai": _openai_options,
    "google": _google_options,
}
