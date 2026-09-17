"""Structured JSON logging.

Every record is emitted as one JSON object so logs can be grepped or shipped to
a log aggregator. A redacting filter removes anything that looks like an API
credential before the record is written.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_SECRET_ENV_NAMES = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY")

_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret[_-]?key|apca[_-]?api[_-]?[a-z-]*)\s*[:=]\s*\S+"),
]

_RESERVED = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "thread", "threadName", "taskName",
}


class RedactingFilter(logging.Filter):
    """Strips live credential values and key-like patterns out of log records."""

    def __init__(self, extra_secrets: Iterable[str] = ()) -> None:
        super().__init__()
        secrets = {os.getenv(name, "") for name in _SECRET_ENV_NAMES}
        secrets.update(extra_secrets)
        self._secrets = sorted((s for s in secrets if s and len(s) >= 8), key=len, reverse=True)

    def _scrub(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "***REDACTED***")
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub(lambda m: f"{m.group(1)}=***REDACTED***", text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._scrub(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._scrub(str(v)) for k, v in record.args.items()}
            else:
                record.args = tuple(self._scrub(str(a)) for a in record.args)
        return True


class JsonFormatter(logging.Formatter):
    """Renders a log record as a single-line JSON document."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(log_dir: str, level: str = "INFO", component: str = "stockbot") -> logging.Logger:
    """Configure root logging once. Safe to call repeatedly."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in list(root.handlers):
        root.removeHandler(handler)

    redactor = RedactingFilter()
    formatter = JsonFormatter()

    # Logs go to stderr so that stdout carries only a command's JSON output and
    # stays pipeable into jq.
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    stream.addFilter(redactor)
    root.addHandler(stream)

    file_handler = logging.handlers.RotatingFileHandler(
        Path(log_dir) / f"{component}.log", maxBytes=10_000_000, backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(redactor)
    root.addHandler(file_handler)

    # Third-party noise.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    return logging.getLogger(component)
