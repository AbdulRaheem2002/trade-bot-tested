"""Custom exception hierarchy for the trading agent."""

from __future__ import annotations


class TradingAgentError(Exception):
    """Base exception for all trading-agent errors."""


class ConfigError(TradingAgentError):
    """Raised when configuration is invalid or missing."""


class ValidationError(TradingAgentError):
    """Raised when a decision item fails validation."""


class SizingError(TradingAgentError):
    """Raised when position sizing cannot produce a valid order."""


class DataError(TradingAgentError):
    """Base exception for data layer errors."""


class ExchangeError(DataError):
    """Raised on exchange API failure (non-retryable or fatal)."""


class NetworkError(DataError):
    """Raised on network connection / timeout failures."""


class RateLimitError(DataError):
    """Raised when the exchange rate limit is exceeded."""


class StaleDataError(DataError):
    """Raised when market data is stale or older than expected."""


class ExecutionError(TradingAgentError):
    """Base exception for execution and broker errors."""


class InsufficientFundsError(ExecutionError):
    """Raised when an order requires more cash than currently available."""


class OrderRejectedError(ExecutionError):
    """Raised when an order is rejected by broker rules (min notional, precision, etc.)."""


class PositionNotFoundError(ExecutionError):
    """Raised when trying to close or modify a position that does not exist."""


class ReconciliationError(ExecutionError):
    """Raised when internal state mismatches external broker state."""


class LLMCallError(TradingAgentError):
    """Raised when an LLM call fails after all retries are exhausted."""

    def __init__(
        self,
        message: str,
        raw_response: str = "",
        attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.attempts = attempts


LLMCallFailure = LLMCallError


class OrchestratorError(TradingAgentError):
    """Base exception for orchestrator errors."""


class InstanceAlreadyRunningError(OrchestratorError):
    """Raised when another instance of the trading agent is already holding the lock."""

