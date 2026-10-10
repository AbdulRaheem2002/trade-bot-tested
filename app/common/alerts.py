"""Alert and notification dispatcher – Rule / Phase 5.

Provides:
- AlertLevel: Enum of alert severities (INFO, WARNING, CRITICAL)
- Alert: Structured alert dataclass
- AlertManager: Dispatcher with listener callbacks, logging, and in-memory history.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from app.common.logging import get_logger

logger = get_logger(__name__)


class AlertLevel(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """Structured alert record."""

    level: AlertLevel
    title: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def format_log(self) -> str:
        return f"[{self.level.value}] {self.title}: {self.message}"


AlertHandler = Callable[[Alert], Any]


class AlertManager:
    """Manages system alerts and notifies registered handlers."""

    def __init__(self) -> None:
        self._handlers: list[AlertHandler] = []
        self._history: list[Alert] = []

    def add_handler(self, handler: AlertHandler) -> None:
        """Register an alert listener callback."""
        self._handlers.append(handler)

    def remove_handler(self, handler: AlertHandler) -> None:
        """Remove a previously registered alert listener."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def get_history(self) -> list[Alert]:
        """Return history of all dispatched alerts."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear the alert history."""
        self._history.clear()

    async def emit(
        self,
        level: AlertLevel | str,
        title: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> Alert:
        """Emit a structured alert to logs and all registered handlers."""
        lvl = AlertLevel(level) if isinstance(level, str) else level
        alert = Alert(
            level=lvl,
            title=title,
            message=message,
            data=data or {},
            timestamp=datetime.now(UTC),
        )

        self._history.append(alert)

        # Log according to severity
        if lvl == AlertLevel.CRITICAL:
            logger.critical(alert.format_log(), extra={"alert_data": alert.data})
        elif lvl == AlertLevel.WARNING:
            logger.warning(alert.format_log(), extra={"alert_data": alert.data})
        else:
            logger.info(alert.format_log(), extra={"alert_data": alert.data})

        # Notify handlers
        for handler in self._handlers:
            try:
                res = handler(alert)
                if hasattr(res, "__await__"):
                    await res
            except Exception as exc:
                logger.error("Error in alert handler %s: %s", handler, exc)

        return alert

    async def info(
        self, title: str, message: str, data: dict[str, Any] | None = None
    ) -> Alert:
        return await self.emit(AlertLevel.INFO, title, message, data)

    async def warning(
        self, title: str, message: str, data: dict[str, Any] | None = None
    ) -> Alert:
        return await self.emit(AlertLevel.WARNING, title, message, data)

    async def critical(
        self, title: str, message: str, data: dict[str, Any] | None = None
    ) -> Alert:
        return await self.emit(AlertLevel.CRITICAL, title, message, data)


# Global default instance
_default_alert_manager = AlertManager()


def get_alert_manager() -> AlertManager:
    return _default_alert_manager
