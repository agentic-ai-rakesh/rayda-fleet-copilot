"""JSON-schema helpers for exposing Pydantic models as LLM tool definitions."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel


def inline_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Model JSON schema with every `$ref` inlined and `$defs`/titles removed."""
    schema = model.model_json_schema()
    definitions = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                merged = {
                    **copy.deepcopy(definitions[name]),
                    **{k: v for k, v in node.items() if k != "$ref"},
                }
                return resolve(merged)
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    resolved = resolve(schema)
    resolved.setdefault("properties", {})
    resolved["type"] = "object"
    return resolved
