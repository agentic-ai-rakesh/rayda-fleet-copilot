"""Record and replay LLM responses, for reproducible agent evals.

- record: call the real model and save each response under a key derived from
  the exact request (purpose, model, output schema, messages, tool names).
- replay: serve saved responses only; a request never seen before raises
  ReplayMissError instead of calling the model.

Because tools are deterministic and "now" is the dataset's as-of time, the
same question produces the same requests, so a replayed run is exact and
costs nothing. Current Claude models do not accept temperature 0; this is how
the agent suite stays deterministic.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, message_to_dict, messages_from_dict

from fleet_copilot.exceptions import LLMError
from fleet_copilot.llm.llm_client import LLMClient, SchemaT
from fleet_copilot.utils.hash_utils import canonical_json, sha256_text

RecordingMode = Literal["record", "replay"]


class ReplayMissError(LLMError):
    """Replay mode was asked for a response that was never recorded."""


class RecordingLLMClient:
    def __init__(
        self,
        directory: Path,
        mode: RecordingMode,
        model_name: str,
        inner: LLMClient | None = None,
    ) -> None:
        if mode == "record" and inner is None:
            raise ValueError("record mode needs a real LLM client")
        self._dir = directory
        self._mode = mode
        self._model_name = model_name
        self._inner = inner
        self.hits = 0  # served from a recording
        self.misses = 0  # not recorded yet (sent to the model in record mode)
        self.replay_failures = 0  # not recorded, in replay mode: the run is incomplete
        directory.mkdir(parents=True, exist_ok=True)

    def structured(
        self, messages: Sequence[BaseMessage], schema: type[SchemaT], *, purpose: str
    ) -> SchemaT:
        key = self._key(purpose, messages, schema=schema.__name__)
        saved = self._load(key)
        if saved is not None:
            return schema.model_validate(saved["value"])
        result = self._require_inner(key, purpose).structured(messages, schema, purpose=purpose)
        self._save(key, purpose, {"kind": "structured", "value": result.model_dump(mode="json")})
        return result

    def call_tools(
        self, messages: Sequence[BaseMessage], tools: Sequence[dict[str, Any]], *, purpose: str
    ) -> AIMessage:
        names = sorted(t.get("name") or t["function"]["name"] for t in tools)
        key = self._key(purpose, messages, tools=names)
        saved = self._load(key)
        if saved is not None:
            (message,) = messages_from_dict([saved["value"]])
            return message
        result = self._require_inner(key, purpose).call_tools(messages, tools, purpose=purpose)
        self._save(key, purpose, {"kind": "ai_message", "value": message_to_dict(result)})
        return result

    # ----------------------------------------------------------------- helpers
    def _key(self, purpose: str, messages: Sequence[BaseMessage], **extra: Any) -> str:
        normalised = [
            {
                "type": m.type,
                "content": m.content,
                "tool_calls": [
                    {"name": c["name"], "args": c["args"], "id": c["id"]}
                    for c in getattr(m, "tool_calls", []) or []
                ],
                "tool_call_id": getattr(m, "tool_call_id", None),
            }
            for m in messages
        ]
        payload = {"purpose": purpose, "model": self._model_name, "messages": normalised, **extra}
        return sha256_text(canonical_json(payload))[:32]

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def _load(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        self.hits += 1
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, key: str, purpose: str, record: dict[str, Any]) -> None:
        record = {"purpose": purpose, "model": self._model_name, **record}
        self._path(key).write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")

    def _require_inner(self, key: str, purpose: str) -> LLMClient:
        self.misses += 1
        if self._mode == "replay" or self._inner is None:
            self.replay_failures += 1
            raise ReplayMissError(
                f"{purpose}: no recorded response ({key}); run evals with --mode record."
            )
        return self._inner
