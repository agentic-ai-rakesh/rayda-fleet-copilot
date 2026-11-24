"""The two things the agent asks of a language model.

1. `structured`: return an instance of a Pydantic schema (plans, answers, proposals).
2. `call_tools`: one step of the tool-calling loop (an AIMessage, maybe with tool calls).

The graph depends only on this interface, so tests run on a scripted client
and the provider switch only adds new implementations.

Fallback behaviour (LangChainLLMClient):
- Rate limits, timeouts and 5xx are retried by the SDK (`llm.max_retries`).
- A refusal or blocked response (Claude stop_reason "refusal", OpenAI
  refusal/content_filter, Gemini safety finish reasons) raises LLMError instead
  of returning an empty answer. For Claude, `server_side_fallback` first
  retries the request on a fallback model.
- Any other failure raises LLMError; graph nodes turn it into a safe reply.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from fleet_copilot.exceptions import LLMError

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMClient(Protocol):
    def structured(
        self, messages: Sequence[BaseMessage], schema: type[SchemaT], *, purpose: str
    ) -> SchemaT: ...

    def call_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]], *, purpose: str
    ) -> AIMessage: ...


class LangChainLLMClient:
    """LLMClient backed by a LangChain chat model."""

    def __init__(self, chat_model: BaseChatModel, structured_method: str = "json_schema") -> None:
        self._model = chat_model
        self._structured_method = structured_method

    def structured(
        self, messages: Sequence[BaseMessage], schema: type[SchemaT], *, purpose: str
    ) -> SchemaT:
        runnable = self._model.with_structured_output(
            schema, method=self._structured_method, include_raw=True
        )
        started = time.perf_counter()
        try:
            output = runnable.invoke(list(messages))
        except Exception as exc:
            raise LLMError(f"{purpose}: model call failed ({type(exc).__name__}: {exc})") from exc

        raw: AIMessage | None = output.get("raw")
        self._log(purpose, started, raw)
        _raise_on_refusal(purpose, raw)
        parsed = output.get("parsed")
        if output.get("parsing_error") or not isinstance(parsed, schema):
            raise LLMError(f"{purpose}: response did not match {schema.__name__}.")
        return parsed

    def call_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]], *, purpose: str
    ) -> AIMessage:
        started = time.perf_counter()
        try:
            response = self._model.bind_tools(list(tools)).invoke(list(messages))
        except Exception as exc:
            raise LLMError(f"{purpose}: model call failed ({type(exc).__name__}: {exc})") from exc
        if not isinstance(response, AIMessage):
            raise LLMError(f"{purpose}: unexpected response type {type(response).__name__}.")
        self._log(purpose, started, response)
        _raise_on_refusal(purpose, response)
        return response

    def _log(self, purpose: str, started: float, message: AIMessage | None) -> None:
        usage = getattr(message, "usage_metadata", None) or {}
        meta = getattr(message, "response_metadata", None) or {}
        logger.info(
            "llm %s: %.0f ms, %s in / %s out tokens",
            purpose,
            (time.perf_counter() - started) * 1000,
            usage.get("input_tokens", "?"),
            usage.get("output_tokens", "?"),
            extra={
                "context": {
                    "purpose": purpose,
                    "model": meta.get("model") or meta.get("model_name"),
                    "stop_reason": meta.get("stop_reason"),
                    "usage": usage,
                }
            },
        )


# Finish reasons that mean the provider blocked or declined the response.
_BLOCKED_FINISH_REASONS = {
    "content_filter",  # OpenAI
    "SAFETY",
    "PROHIBITED_CONTENT",
    "BLOCKLIST",
    "SPII",
    "RECITATION",  # Gemini
}


def refusal_reason(message: AIMessage | None) -> str | None:
    """Why a response was declined, for any provider; None if it was not."""
    meta = getattr(message, "response_metadata", None) or {}
    if meta.get("stop_reason") == "refusal":  # Anthropic
        return str((meta.get("stop_details") or {}).get("category") or "refusal")
    refusal = (getattr(message, "additional_kwargs", None) or {}).get("refusal")
    if refusal:  # OpenAI structured-output refusal
        return "refusal"
    finish = meta.get("finish_reason")
    if isinstance(finish, str) and finish in _BLOCKED_FINISH_REASONS:
        return finish
    return None


def _raise_on_refusal(purpose: str, message: AIMessage | None) -> None:
    reason = refusal_reason(message)
    if reason:
        raise LLMError(f"{purpose}: the model declined the request ({reason}).")
