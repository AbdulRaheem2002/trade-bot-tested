"""Shared fixtures for all test modules."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.common.enums import Action, ConvictionTier, Side
from app.common.models import (
    BreakerState,
    CycleState,
    DecisionItem,
    EquityPoint,
    MarketSnapshot,
    PortfolioState,
    Position,
)
from app.config.settings import TradingConfig

# ════════════════════════════════════════════════════════════════════════
# Timestamps
# ════════════════════════════════════════════════════════════════════════

CYCLE_TIME = datetime(2025, 6, 15, 4, 5, 0, tzinfo=UTC)
EXPECTED_CANDLE = datetime(2025, 6, 15, 0, 0, 0, tzinfo=UTC)


# ════════════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════════════


@pytest.fixture
def cfg() -> TradingConfig:
    """Default trading config."""
    return TradingConfig()


# ════════════════════════════════════════════════════════════════════════
# Market Snapshot
# ════════════════════════════════════════════════════════════════════════


def make_snapshot(
    symbol: str = "BTC/USDT",
    last_close: float = 70_000.0,
    ema200_4h: float = 68_000.0,
    atr14_4h: float = 1_000.0,
    spread_bps: float = 3.0,
    volume_24h: float = 100_000_000.0,
    candle_timestamp: datetime | None = None,
    current_price: float | None = None,
    lot_size: float = 0.001,
    min_notional: float = 10.0,
) -> MarketSnapshot:
    return MarketSnapshot(
        symbol=symbol,
        last_close=last_close,
        ema200_4h=ema200_4h,
        atr14_4h=atr14_4h,
        spread_bps=spread_bps,
        volume_24h=volume_24h,
        candle_timestamp=candle_timestamp or EXPECTED_CANDLE,
        current_price=current_price or last_close,
        lot_size=lot_size,
        min_notional=min_notional,
    )


# ════════════════════════════════════════════════════════════════════════
# Portfolio State
# ════════════════════════════════════════════════════════════════════════


def make_portfolio(
    equity: float = 10_000.0,
    cash: float = 10_000.0,
    positions: dict[str, Position] | None = None,
    peak_equity: float | None = None,
    equity_history: list[EquityPoint] | None = None,
) -> PortfolioState:
    return PortfolioState(
        equity=equity,
        cash=cash,
        positions=positions or {},
        peak_equity=peak_equity or equity,
        equity_history=equity_history or [],
    )


# ════════════════════════════════════════════════════════════════════════
# Cycle State
# ════════════════════════════════════════════════════════════════════════


def make_cycle_state(
    breaker_state: BreakerState | None = None,
    kill_switch: bool = False,
    symbols_closed: set[str] | None = None,
    new_opened: bool = False,
) -> CycleState:
    return CycleState(
        cycle_time=CYCLE_TIME,
        expected_candle_time=EXPECTED_CANDLE,
        symbols_closed_this_cycle=symbols_closed or set(),
        new_position_opened_this_cycle=new_opened,
        breaker_state=breaker_state or BreakerState(),
        kill_switch_active=kill_switch,
    )


# ════════════════════════════════════════════════════════════════════════
# Position
# ════════════════════════════════════════════════════════════════════════


def make_position(
    symbol: str = "BTC/USDT",
    side: Side = Side.LONG,
    entry_price: float = 65_000.0,
    quantity: float = 0.01,
    invalidation_price: float = 63_000.0,
    conviction: ConvictionTier = ConvictionTier.A,
    risk_amount: float = 20.0,
    opened_at: datetime | None = None,
) -> Position:
    return Position(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        quantity=quantity,
        invalidation_price=invalidation_price,
        conviction=conviction,
        risk_amount=risk_amount,
        opened_at=opened_at or CYCLE_TIME,
    )


# ════════════════════════════════════════════════════════════════════════
# Decision Item
# ════════════════════════════════════════════════════════════════════════


def make_decision_item(
    action: Action = Action.OPEN_LONG,
    symbol: str = "BTC/USDT",
    invalidation_price: float | None = 68_000.0,
    conviction: ConvictionTier | None = ConvictionTier.A,
    confidence: float | None = 0.8,
    thesis: str | None = "test thesis",
    rationale: str | None = "test rationale",
) -> DecisionItem:
    return DecisionItem(
        action=action,
        symbol=symbol,
        invalidation_price=invalidation_price,
        conviction=conviction,
        confidence=confidence,
        thesis=thesis,
        rationale=rationale,
    )
