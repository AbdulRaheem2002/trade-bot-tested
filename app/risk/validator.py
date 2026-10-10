"""Decision validator – parse LLM output and validate each item against hard rules.

This module enforces rules 1-5, 8, 10, 11, 13, 14 from the master context.
No imports from execution/ or data/.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from app.common.enums import Action, RejectionReason, Side
from app.common.models import (
    CycleState,
    Decision,
    DecisionItem,
    MarketSnapshot,
    PortfolioState,
    Rejection,
)
from app.config.settings import TradingConfig

# ════════════════════════════════════════════════════════════════════════
# Response-level validation (Rule 1)
# ════════════════════════════════════════════════════════════════════════


def validate_response(raw: str | dict | None) -> Decision | None:
    """Parse and schema-validate the full LLM response.

    Returns a ``Decision`` on success, or ``None`` when the response is
    unparsable, schema-invalid, or empty (Rule 1 → treat as hold-all).
    """
    if raw is None:
        return None
    if isinstance(raw, Decision):
        return raw

    data: dict[str, Any]

    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    elif isinstance(raw, dict):
        data = raw
    else:
        return None

    if not isinstance(data, dict):
        return None

    try:
        decision = Decision.model_validate(data)
    except PydanticValidationError:
        return None

    # Empty decisions list → hold-all (Rule 1)
    if not decision.decisions:
        return None

    return decision


# ════════════════════════════════════════════════════════════════════════
# Item-level validation (Rules 3-5, 8, 10, 11, 13, 14)
# ════════════════════════════════════════════════════════════════════════

_OPEN_ACTIONS = frozenset({Action.OPEN_LONG, Action.OPEN_SHORT})
_OPEN_OR_ADD = frozenset({Action.OPEN_LONG, Action.OPEN_SHORT, Action.ADD})


def validate_item(
    item: DecisionItem,
    portfolio: PortfolioState,
    snapshot: MarketSnapshot | None,
    cycle_state: CycleState,
    config: TradingConfig,
) -> Rejection | None:
    """Validate a single decision item against hard rules.

    Returns ``None`` when the item is valid, or a ``Rejection`` explaining
    why it was rejected.
    """

    # ── Rule 13: breaker / kill switch → only close and hold ─────────
    if (
        cycle_state.breaker_state.any_active or cycle_state.kill_switch_active
    ) and item.action not in (Action.CLOSE, Action.HOLD):
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.BREAKER_ACTIVE,
            detail="Breaker or kill switch is active; only close and hold are accepted.",
        )

    # hold requires no further checks
    if item.action == Action.HOLD:
        return None

    # ── close: just needs an open position ───────────────────────────
    if item.action == Action.CLOSE:
        if item.symbol not in portfolio.positions:
            return Rejection(
                symbol=item.symbol,
                action=item.action,
                reason=RejectionReason.NO_OPEN_POSITION,
                detail=f"No open position for {item.symbol} to close.",
            )
        return None

    # From here on we handle open_long / open_short / add

    # ── Rule 3: required fields ──────────────────────────────────────
    if item.symbol not in config.universe:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.SYMBOL_NOT_IN_UNIVERSE,
            detail=f"{item.symbol} is not in the configured universe.",
        )

    if item.invalidation_price is None:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.MISSING_INVALIDATION_PRICE,
            detail="invalidation_price is required for open/add actions.",
        )

    if item.conviction is None:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.MISSING_CONVICTION,
            detail="conviction is required for open/add actions.",
        )

    if item.confidence is None:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.MISSING_CONFIDENCE,
            detail="confidence is required for open/add actions.",
        )

    # ── Rule 11: shorts ──────────────────────────────────────────────
    if item.action == Action.OPEN_SHORT:
        if not config.shorts_enabled:
            return Rejection(
                symbol=item.symbol,
                action=item.action,
                reason=RejectionReason.SHORTS_DISABLED,
                detail="Short selling is disabled in config.",
            )
        if portfolio.equity < config.min_equity_for_shorts:
            return Rejection(
                symbol=item.symbol,
                action=item.action,
                reason=RejectionReason.SHORTS_EQUITY_TOO_LOW,
                detail=(
                    f"Equity {portfolio.equity:.2f} < min_equity_for_shorts "
                    f"{config.min_equity_for_shorts:.2f}."
                ),
            )

    # ── Position existence checks ────────────────────────────────────
    if item.action in _OPEN_ACTIONS and item.symbol in portfolio.positions:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.POSITION_ALREADY_EXISTS,
            detail=f"A position for {item.symbol} is already open.",
        )

    if item.action == Action.ADD and item.symbol not in portfolio.positions:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.NO_OPEN_POSITION,
            detail=f"No open position for {item.symbol} to add to.",
        )

    # ── Rule 8: no reopening a symbol closed this cycle ──────────────
    if item.action in _OPEN_ACTIONS and item.symbol in cycle_state.symbols_closed_this_cycle:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.SYMBOL_CLOSED_THIS_CYCLE,
            detail=f"{item.symbol} was closed earlier this cycle; cannot reopen.",
        )

    # ── Snapshot required from here on ───────────────────────────────
    if snapshot is None:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.STALE_DATA,
            detail=f"No market snapshot available for {item.symbol}.",
        )

    # ── Rule 5: stale data ───────────────────────────────────────────
    if snapshot.candle_timestamp != cycle_state.expected_candle_time:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.STALE_DATA,
            detail=(
                f"Candle timestamp {snapshot.candle_timestamp} does not match "
                f"expected {cycle_state.expected_candle_time}."
            ),
        )

    # ── Rule 5: spread gate ──────────────────────────────────────────
    if snapshot.spread_bps > config.max_spread_bps:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.SPREAD_TOO_WIDE,
            detail=(f"Spread {snapshot.spread_bps:.1f} bps > max {config.max_spread_bps} bps."),
        )

    # ── Rule 5: volume gate ──────────────────────────────────────────
    if snapshot.volume_24h < config.min_24h_quote_volume:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.VOLUME_TOO_LOW,
            detail=(
                f"24h volume {snapshot.volume_24h:,.0f} < min {config.min_24h_quote_volume:,.0f}."
            ),
        )

    # ── Determine side ───────────────────────────────────────────────
    if item.action == Action.ADD:
        side = portfolio.positions[item.symbol].side
    elif item.action == Action.OPEN_LONG:
        side = Side.LONG
    else:
        side = Side.SHORT

    # ── Rule 5: EMA200 setup gate ────────────────────────────────────
    if side == Side.LONG and snapshot.last_close <= snapshot.ema200_4h:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.CLOSE_NOT_ABOVE_EMA200,
            detail=(
                f"Long setup gate: last close {snapshot.last_close:.4f} "
                f"<= EMA200 {snapshot.ema200_4h:.4f}."
            ),
        )
    if side == Side.SHORT and snapshot.last_close >= snapshot.ema200_4h:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.CLOSE_NOT_BELOW_EMA200,
            detail=(
                f"Short setup gate: last close {snapshot.last_close:.4f} "
                f">= EMA200 {snapshot.ema200_4h:.4f}."
            ),
        )

    # ── Rule 4: invalidation price correctly sided ───────────────────
    entry_estimate = snapshot.current_price
    inv = item.invalidation_price

    if side == Side.LONG and inv >= entry_estimate:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.INVALIDATION_WRONG_SIDE,
            detail=(
                f"Long invalidation {inv:.4f} must be below entry estimate {entry_estimate:.4f}."
            ),
        )
    if side == Side.SHORT and inv <= entry_estimate:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.INVALIDATION_WRONG_SIDE,
            detail=(
                f"Short invalidation {inv:.4f} must be above entry estimate {entry_estimate:.4f}."
            ),
        )

    # ── Rule 4: invalidation distance in ATR terms ───────────────────
    distance = abs(entry_estimate - inv)
    atr = snapshot.atr14_4h
    if atr <= 0:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.STALE_DATA,
            detail="ATR14 is zero or negative – data issue.",
        )

    distance_in_atr = distance / atr

    if distance_in_atr < config.min_stop_atr:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.INVALIDATION_TOO_TIGHT,
            detail=(f"Stop distance {distance_in_atr:.2f} ATR < min {config.min_stop_atr} ATR."),
        )
    if distance_in_atr > config.max_stop_atr:
        return Rejection(
            symbol=item.symbol,
            action=item.action,
            reason=RejectionReason.INVALIDATION_TOO_WIDE,
            detail=(f"Stop distance {distance_in_atr:.2f} ATR > max {config.max_stop_atr} ATR."),
        )

    # ── Rule 10: add constraints ─────────────────────────────────────
    if item.action == Action.ADD:
        pos = portfolio.positions[item.symbol]
        unrealised_r = pos.unrealized_r(snapshot.current_price)
        if unrealised_r < 1.0:
            return Rejection(
                symbol=item.symbol,
                action=item.action,
                reason=RejectionReason.ADD_UNREALIZED_R_TOO_LOW,
                detail=(f"Unrealized R = {unrealised_r:.2f} < +1.0 required for add."),
            )

        # Existing stop must be at breakeven or better
        if pos.side == Side.LONG and pos.invalidation_price < pos.entry_price:
            return Rejection(
                symbol=item.symbol,
                action=item.action,
                reason=RejectionReason.ADD_STOP_NOT_AT_BREAKEVEN,
                detail=(
                    f"Long stop {pos.invalidation_price:.4f} < entry "
                    f"{pos.entry_price:.4f}; must be at breakeven or better for add."
                ),
            )
        if pos.side == Side.SHORT and pos.invalidation_price > pos.entry_price:
            return Rejection(
                symbol=item.symbol,
                action=item.action,
                reason=RejectionReason.ADD_STOP_NOT_AT_BREAKEVEN,
                detail=(
                    f"Short stop {pos.invalidation_price:.4f} > entry "
                    f"{pos.entry_price:.4f}; must be at breakeven or better for add."
                ),
            )

    # ── Rule 14: invalidation may only be tightened, never loosened ──
    if item.action == Action.ADD:
        pos = portfolio.positions[item.symbol]
        if pos.side == Side.LONG and item.invalidation_price < pos.invalidation_price:
            return Rejection(
                symbol=item.symbol,
                action=item.action,
                reason=RejectionReason.INVALIDATION_LOOSENED,
                detail=(
                    f"New long stop {item.invalidation_price:.4f} < existing "
                    f"{pos.invalidation_price:.4f}; stops may only be tightened."
                ),
            )
        if pos.side == Side.SHORT and item.invalidation_price > pos.invalidation_price:
            return Rejection(
                symbol=item.symbol,
                action=item.action,
                reason=RejectionReason.INVALIDATION_LOOSENED,
                detail=(
                    f"New short stop {item.invalidation_price:.4f} > existing "
                    f"{pos.invalidation_price:.4f}; stops may only be tightened."
                ),
            )

    return None  # All checks passed
