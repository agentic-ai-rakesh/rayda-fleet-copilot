"""Logging setup: readable console output plus structured JSON log files.

Every log line carries the current run_id, so all lines from one command
(and, later, one agent conversation) can be found together. Pass structured
fields with `extra={"context": {...}}`; they land in the JSON file as-is.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fleet_copilot.config.settings import LoggingSettings

APP_LOGGER_NAME = "fleet_copilot"
_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s [%(run_id)s] %(name)s: %(message)s"

_current_run_id: ContextVar[str] = ContextVar("run_id", default="-")


def new_run_id(prefix: str = "run") -> str:
    """Create a short, unique id for one command run or agent session."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def set_run_id(run_id: str) -> None:
    _current_run_id.set(run_id)


def get_run_id() -> str:
    return _current_run_id.get()


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = get_run_id()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, ready for log tooling."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", "-"),
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if context:
            payload["context"] = context
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(settings: LoggingSettings, log_dir: Path) -> None:
    """Configure the app logger. Safe to call more than once."""
    logger = logging.getLogger(APP_LOGGER_NAME)
    logger.setLevel(settings.level.upper())
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    run_id_filter = _RunIdFilter()

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    console.addFilter(run_id_filter)
    logger.addHandler(console)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_dir / settings.log_file_name,
            maxBytes=settings.max_bytes,
            backupCount=settings.backup_count,
            encoding="utf-8",
        )
    except OSError as exc:
        # Fallback: keep running with console logging only.
        logger.warning("Could not open log file in %s (%s); logging to console only.", log_dir, exc)
        return
    file_handler.setFormatter(JsonFormatter())
    file_handler.addFilter(run_id_filter)
    logger.addHandler(file_handler)
