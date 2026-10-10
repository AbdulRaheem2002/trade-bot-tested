"""Pydantic v2 domain models used across the trading agent.

These are pure data models with no imports from execution/ or data/.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.common.enums import (
    Action,
    ConvictionTier,
    RejectionReason,
    Side,
)

# ════════════════════════════════════════════════════════════════════════
# LLM Decision contract
# ════════════════════════════════════════════════════════════════════════


class DecisionItem(BaseModel):
    """A single decision proposed by the LLM (Reflex output)."""

    model_config = {"extra": "allow"}  # tolerate extra keys but ignore them

    action: Action
    symbol: str
    invalidation_price: float | None = None
    conviction: ConvictionTier | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    thesis: str | None = None
    invalidation: str | None = None  # textual description
    rationale: str | None = None


class Decision(BaseModel):
    """Top-level LLM response for one Reflex cycle."""

    model_config = {"extra": "allow"}

    reasoning: str = ""
    decisions: list[DecisionItem] = Field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════
# Portfolio / Position
# ════════════════════════════════════════════════════════════════════════


class Position(BaseModel):
    """An open position tracked by the portfolio service."""

    symbol: str
    side: Side
    entry_price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    invalidation_price: float = Field(gt=0)
    conviction: ConvictionTier
    risk_amount: float = Field(gt=0)  # 1 R in quote currency
    opened_at: datetime

    @property
    def notional(self) -> float:
        """Entry notional in quote currency."""
        return self.entry_price * self.quantity

    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized PnL in quote currency at *current_price*."""
        if self.side == Side.LONG:
            return (current_price - self.entry_price) * self.quantity
        return (self.entry_price - current_price) * self.quantity

    def unrealized_r(self, current_price: float) -> float:
        """Unrealized PnL expressed in R-multiples."""
        if self.risk_amount == 0:
            return 0.0
        return self.unrealized_pnl(current_price) / self.risk_amount


class EquityPoint(BaseModel):
    """A single equity snapshot for breaker calculations."""

    timestamp: datetime
    equity: float


class PortfolioState(BaseModel):
    """Immutable snapshot of the portfolio handed to the risk engine."""

    equity: float = Field(gt=0)
    cash: float = Field(ge=0)
    positions: dict[str, Position] = Field(default_factory=dict)
    peak_equity: float = Field(gt=0)
    equity_history: list[EquityPoint] = Field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════
# Market data snapshot
# ════════════════════════════════════════════════════════════════════════


class MarketSnapshot(BaseModel):
    """Symbol-level fields the validator and sizer need for one cycle."""

    symbol: str
    last_close: float = Field(gt=0)
    ema200_4h: float = Field(gt=0)
    atr14_4h: float = Field(gt=0)
    spread_bps: float = Field(ge=0)
    volume_24h: float = Field(ge=0)
    candle_timestamp: datetime
    current_price: float = Field(gt=0)
    lot_size: float = Field(gt=0, default=0.001)
    min_notional: float = Field(gt=0, default=10.0)
    candles_4h: list[dict[str, Any]] = Field(default_factory=list)
    candles_1d: list[dict[str, Any]] = Field(default_factory=list)
    rsi14_4h: float | None = None
    volume_sma20_4h: float | None = None
    ema50_1d: float | None = None
    ema200_1d: float | None = None


# ════════════════════════════════════════════════════════════════════════
# Breaker state
# ════════════════════════════════════════════════════════════════════════


class BreakerState(BaseModel):
    """Result of circuit-breaker evaluation."""

    daily_loss_blocked: bool = False
    weekly_loss_blocked: bool = False
    peak_drawdown_blocked: bool = False
    is_paused: bool = False  # latched by peak-drawdown until manual /resume

    @property
    def any_active(self) -> bool:
        return (
            self.daily_loss_blocked
            or self.weekly_loss_blocked
            or self.peak_drawdown_blocked
            or self.is_paused
        )


# ════════════════════════════════════════════════════════════════════════
# Cycle state (tracks intra-cycle bookkeeping)
# ════════════════════════════════════════════════════════════════════════


class CycleState(BaseModel):
    """Mutable state carried through one decision cycle."""

    cycle_time: datetime
    expected_candle_time: datetime
    symbols_closed_this_cycle: set[str] = Field(default_factory=set)
    new_position_opened_this_cycle: bool = False
    breaker_state: BreakerState = Field(default_factory=BreakerState)
    kill_switch_active: bool = False


# ════════════════════════════════════════════════════════════════════════
# Sized order & rejection (risk-engine output)
# ════════════════════════════════════════════════════════════════════════


class SizedOrder(BaseModel):
    """A fully-sized, approved order ready for execution."""

    symbol: str
    action: Action
    side: Side
    quantity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    invalidation_price: float = Field(gt=0)
    conviction: ConvictionTier
    confidence: float = Field(ge=0, le=1)
    notional: float = Field(gt=0)
    risk_amount: float = Field(gt=0)
    thesis: str | None = None
    rationale: str | None = None


class Rejection(BaseModel):
    """Record of a rejected decision item with machine- and human-readable reasons."""

    symbol: str
    action: Action
    reason: RejectionReason
    detail: str


# ════════════════════════════════════════════════════════════════════════
# Approved order (alias used by risk_engine for clarity)
# ════════════════════════════════════════════════════════════════════════

# For close orders, quantity comes from the existing position, not the sizer.


class ApprovedOrder(BaseModel):
    """An order approved by the full risk pipeline.

    For open/add this wraps a SizedOrder.  For close it carries the
    existing position's quantity.
    """

    symbol: str
    action: Action
    side: Side
    quantity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    invalidation_price: float = Field(default=0, ge=0)
    conviction: ConvictionTier | None = None
    confidence: float | None = None
    notional: float = Field(gt=0)
    risk_amount: float = Field(ge=0, default=0)
    thesis: str | None = None
    rationale: str | None = None


# ════════════════════════════════════════════════════════════════════════
# Execution & Trades
# ════════════════════════════════════════════════════════════════════════


class OrderFill(BaseModel):
    """An executed order fill returned by a broker."""

    fill_id: str
    client_order_id: str
    symbol: str
    side: Side
    price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    fee: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)
    timestamp: datetime


class ClosedTrade(BaseModel):
    """Record of a closed trade for performance tracking."""

    trade_id: str
    symbol: str
    side: Side
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    quantity: float = Field(gt=0)
    entry_time: datetime
    exit_time: datetime
    realized_pnl: float
    r_multiple: float
    fees_total: float = Field(ge=0)
    exit_reason: str = ""
