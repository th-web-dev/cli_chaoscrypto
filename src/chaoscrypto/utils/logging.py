from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

# Map string levels to logging constants
_LEVELS = {
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}

# Context variable so nested functions inherit the active command name
_current_command: ContextVar[str] = ContextVar("chaoscrypto_current_command", default="cli")


class _CommandFilter(logging.Filter):
    """Ensure every log record has a command label for prefix formatting."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        command = getattr(record, "command", None) or _current_command.get()
        record.command = command
        return True


def setup_logging(level: str = "WARNING") -> None:
    """
    Configure root logging once.

    Uses stderr, attaches a command-aware filter, and keeps the format concise.
    """
    if isinstance(level, str):
        numeric_level = _LEVELS.get(level.lower(), logging.WARNING)
    else:
        numeric_level = int(level)
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(command)s] %(levelname)s: %(message)s"))
        handler.addFilter(_CommandFilter())
        root.addHandler(handler)
    root.setLevel(numeric_level)
    # Keep third-party loggers quiet unless explicitly enabled
    logging.captureWarnings(True)


def set_command_context(command: str) -> None:
    """Tag subsequent log records with the active command name."""
    _current_command.set(command)


def resolve_log_level(verbose: bool, debug: bool) -> str:
    """Derive the configured level name from CLI flags."""
    if debug:
        return "DEBUG"
    if verbose:
        return "INFO"
    return "WARNING"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Convenience wrapper to keep imports centralized."""
    return logging.getLogger(name if name is not None else __name__)
