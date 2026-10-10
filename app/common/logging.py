"""Structured JSON logging setup for the trading agent."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Merge extra structured data attached via `extra={"data": {...}}`
        if hasattr(record, "data") and isinstance(record.data, dict):
            payload["data"] = record.data
        if record.exc_info and record.exc_info[1]:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes structured JSON to stderr.

    Safe to call multiple times with the same *name*; handlers are added
    only once.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with structured JSON formatting."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JSONFormatter())
        root.addHandler(handler)
    root.setLevel(level)

