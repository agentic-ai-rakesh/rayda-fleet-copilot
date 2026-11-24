"""Application exception hierarchy.

Every error raised on purpose by the app derives from FleetCopilotError, so the
CLI can report it cleanly and let anything unexpected surface as a real bug.
"""


class FleetCopilotError(Exception):
    """Base class for expected, user-reportable errors."""


class ConfigError(FleetCopilotError):
    """The settings file exists but is invalid."""


class IngestionError(FleetCopilotError):
    """The dataset could not be loaded into the database."""


class DatabaseNotReadyError(FleetCopilotError):
    """The fleet database has not been built yet."""


class TenantNotFoundError(FleetCopilotError):
    """The requested company does not exist."""


class LLMError(FleetCopilotError):
    """The language model could not produce a usable response."""


class LLMUnavailableError(LLMError):
    """The language model is not configured or cannot be reached."""


class AuditError(FleetCopilotError):
    """An audit event could not be written. Callers must stop the operation."""


class AuditIntegrityError(AuditError):
    """The audit log hash chain does not verify (an entry was altered or removed)."""
