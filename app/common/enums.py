"""Enumerations used across the trading agent."""

from __future__ import annotations

from enum import StrEnum


class Action(StrEnum):
    """Decision actions the LLM may propose."""

    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    ADD = "add"
    CLOSE = "close"
    HOLD = "hold"


class Side(StrEnum):
    """Position side."""

    LONG = "long"
    SHORT = "short"


class ConvictionTier(StrEnum):
    """Conviction tiers that multiply risk-per-trade."""

    A = "A"
    B = "B"


class RejectionReason(StrEnum):
    """Machine-readable reason a decision was rejected."""

    # ── Rule 1 – parse / schema ──────────────────────────────────────
    PARSE_ERROR = "parse_error"
    SCHEMA_INVALID = "schema_invalid"
    EMPTY_DECISIONS = "empty_decisions"

    # ── Rule 3 – required fields for open/add ────────────────────────
    SYMBOL_NOT_IN_UNIVERSE = "symbol_not_in_universe"
    MISSING_INVALIDATION_PRICE = "missing_invalidation_price"
    MISSING_CONVICTION = "missing_conviction"
    MISSING_CONFIDENCE = "missing_confidence"

    # ── Rule 4 – invalidation geometry ───────────────────────────────
    INVALIDATION_WRONG_SIDE = "invalidation_wrong_side"
    INVALIDATION_TOO_TIGHT = "invalidation_too_tight"
    INVALIDATION_TOO_WIDE = "invalidation_too_wide"

    # ── Rule 5 – setup gates ─────────────────────────────────────────
    CLOSE_NOT_ABOVE_EMA200 = "close_not_above_ema200"
    CLOSE_NOT_BELOW_EMA200 = "close_not_below_ema200"
    SPREAD_TOO_WIDE = "spread_too_wide"
    VOLUME_TOO_LOW = "volume_too_low"
    STALE_DATA = "stale_data"

    # ── Rule 7 – one new position per cycle ──────────────────────────
    MAX_ONE_NEW_POSITION_PER_CYCLE = "max_one_new_position_per_cycle"

    # ── Rule 8 – no reopen ───────────────────────────────────────────
    SYMBOL_CLOSED_THIS_CYCLE = "symbol_closed_this_cycle"

    # ── Rule 9 – position limits ─────────────────────────────────────
    MAX_POSITIONS_REACHED = "max_positions_reached"

    # ── Rule 10 – add constraints ────────────────────────────────────
    ADD_UNREALIZED_R_TOO_LOW = "add_unrealized_r_too_low"
    ADD_STOP_NOT_AT_BREAKEVEN = "add_stop_not_at_breakeven"

    # ── Rule 11 – shorts ─────────────────────────────────────────────
    SHORTS_DISABLED = "shorts_disabled"
    SHORTS_EQUITY_TOO_LOW = "shorts_equity_too_low"

    # ── Rule 13 – breaker / kill switch ──────────────────────────────
    BREAKER_ACTIVE = "breaker_active"

    # ── Rule 14 – stop tightening only ───────────────────────────────
    INVALIDATION_LOOSENED = "invalidation_loosened"

    # ── Sizing rejections (rule 6) ───────────────────────────────────
    BELOW_MIN_NOTIONAL = "below_min_notional"

    # ── Position existence checks ────────────────────────────────────
    NO_OPEN_POSITION = "no_open_position"
    POSITION_ALREADY_EXISTS = "position_already_exists"


class BreakerKind(StrEnum):
    """Which circuit breaker fired."""

    DAILY_LOSS = "daily_loss"
    WEEKLY_LOSS = "weekly_loss"
    PEAK_DRAWDOWN = "peak_drawdown"


class TradingMode(StrEnum):
    """Operational mode."""

    PAPER = "paper"
    LIVE = "live"


class OrderStatus(StrEnum):
    """Lifecycle status of an order."""

    PENDING = "pending"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class AgentState(StrEnum):
    """Operational state of the trading agent."""

    RUNNING = "RUNNING"
    PAUSED_KILL = "PAUSED_KILL"
    PAUSED_BREAKER = "PAUSED_BREAKER"
    PAUSED_RECONCILIATION = "PAUSED_RECONCILIATION"
    STOPPED = "STOPPED"

