"""Backtesting engine and deterministic strategies – Rule / Phase 7."""

from app.backtest.engine import BacktestEngine, BacktestResult
from app.backtest.strategy import (
    Breakout20Strategy,
    EMATrendStrategy,
    Strategy,
    get_strategy,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "Breakout20Strategy",
    "EMATrendStrategy",
    "Strategy",
    "get_strategy",
]
