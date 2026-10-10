"""Position sizer – implements Rule 6 exactly.

Pure function.  No imports from execution/ or data/.

The sizer computes quantity from the risk formula, then caps by:
1. max_single_position_pct (notional / equity)
2. remaining exposure room  (max_total_exposure_pct - current exposure)
3. available cash (after entry fee)
4. lot-size rounding (floor to lot_size step)
5. min-notional check (reject if below exchange minimum)
"""

from __future__ import annotations

import math

from app.common.enums import Action, ConvictionTier, RejectionReason, Side
from app.common.models import Rejection, SizedOrder
from app.config.settings import TradingConfig


def size_order(
    *,
    symbol: str,
    action: Action,
    side: Side,
    entry_price: float,
    invalidation_price: float,
    conviction: ConvictionTier,
    confidence: float,
    equity: float,
    total_exposure: float,
    existing_symbol_notional: float,
    cash: float,
    config: TradingConfig,
    lot_size: float = 0.001,
    min_notional: float = 10.0,
    thesis: str | None = None,
    rationale: str | None = None,
) -> SizedOrder | Rejection:
    """Size a single order per Rule 6.

    Returns a ``SizedOrder`` on success, or a ``Rejection`` if the
    capped quantity falls below the exchange minimum notional.
    """

    # ── 1. Risk amount ───────────────────────────────────────────────
    tier_mult = config.conviction_multipliers[conviction]
    risk_amount = equity * (config.risk_per_trade_pct / 100.0) * tier_mult

    # ── 2. Effective stop distance ───────────────────────────────────
    raw_distance = abs(entry_price - invalidation_price)
    slippage_buffer = entry_price * (config.slippage_bps / 10_000.0)
    fee_buffer = entry_price * (config.fees_taker_bps / 10_000.0) * 2  # entry + exit
    effective_distance = raw_distance + slippage_buffer + fee_buffer

    if effective_distance <= 0:
        return Rejection(
            symbol=symbol,
            action=action,
            reason=RejectionReason.BELOW_MIN_NOTIONAL,
            detail="Effective stop distance is zero or negative.",
        )

    # ── 3. Raw quantity from risk formula ────────────────────────────
    qty = risk_amount / effective_distance

    # ── 4. Cap: max single position pct ──────────────────────────────
    max_single_notional = equity * (config.max_single_position_pct / 100.0)
    # For add, the combined notional must stay under the cap
    allowed_notional = max_single_notional - existing_symbol_notional
    if allowed_notional <= 0:
        return Rejection(
            symbol=symbol,
            action=action,
            reason=RejectionReason.BELOW_MIN_NOTIONAL,
            detail=(
                f"Existing symbol notional {existing_symbol_notional:.2f} already "
                f"at/above single-position cap {max_single_notional:.2f}."
            ),
        )
    max_qty_single = allowed_notional / entry_price
    qty = min(qty, max_qty_single)

    # ── 5. Cap: remaining total exposure room ────────────────────────
    max_total_notional = equity * (config.max_total_exposure_pct / 100.0)
    exposure_room = max_total_notional - total_exposure
    if exposure_room <= 0:
        return Rejection(
            symbol=symbol,
            action=action,
            reason=RejectionReason.BELOW_MIN_NOTIONAL,
            detail=(
                f"Total exposure {total_exposure:.2f} already at/above "
                f"cap {max_total_notional:.2f}."
            ),
        )
    max_qty_exposure = exposure_room / entry_price
    qty = min(qty, max_qty_exposure)

    # ── 6. Cap: available cash after entry fee ───────────────────────
    entry_fee_mult = 1 + config.fees_taker_bps / 10_000.0
    if cash <= 0:
        return Rejection(
            symbol=symbol,
            action=action,
            reason=RejectionReason.BELOW_MIN_NOTIONAL,
            detail="No cash available.",
        )
    max_qty_cash = cash / (entry_price * entry_fee_mult)
    qty = min(qty, max_qty_cash)

    # ── 7. Lot-size rounding (floor) ─────────────────────────────────
    if lot_size > 0:
        qty = math.floor(qty / lot_size) * lot_size
        # Avoid floating-point dust
        decimals = _lot_decimals(lot_size)
        qty = round(qty, decimals)

    # ── 8. Min-notional check ────────────────────────────────────────
    notional = qty * entry_price
    if qty <= 0 or notional < min_notional:
        return Rejection(
            symbol=symbol,
            action=action,
            reason=RejectionReason.BELOW_MIN_NOTIONAL,
            detail=(
                f"Sized notional {notional:.2f} < min notional {min_notional:.2f} "
                f"(qty={qty}, entry={entry_price:.4f})."
            ),
        )

    # ── 9. Compute actual risk amount for this sized qty ─────────────
    actual_risk = qty * effective_distance

    return SizedOrder(
        symbol=symbol,
        action=action,
        side=side,
        quantity=qty,
        entry_price=entry_price,
        invalidation_price=invalidation_price,
        conviction=conviction,
        confidence=confidence,
        notional=notional,
        risk_amount=actual_risk,
        thesis=thesis,
        rationale=rationale,
    )


def _lot_decimals(lot_size: float) -> int:
    """Return the number of decimal places implied by *lot_size*."""
    if lot_size >= 1:
        return 0
    s = f"{lot_size:.10f}".rstrip("0")
    if "." not in s:
        return 0
    return len(s.split(".")[1])
