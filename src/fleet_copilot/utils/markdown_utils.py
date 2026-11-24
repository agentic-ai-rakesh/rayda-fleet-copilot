"""Minimal Markdown builders for generated reports."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    def cell(value: object) -> str:
        text = "—" if value is None else str(value)
        return text.replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)
