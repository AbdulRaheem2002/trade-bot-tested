"""Risk engine – single entry point for the decision pipeline.

Orchestrates validator → sizer → breakers and enforces Rules 7, 9, 12.
No imports from execution/ or data/.
"""

from __future__ import annotations

from app.common.enums import Action, ConvictionTier, RejectionReason, Side
from app.common.logging import get_logger
from app.common.models import (
    ApprovedOrder,
    CycleState,
    DecisionItem,
    MarketSnapshot,
    PortfolioState,
    Rejection,
    SizedOrder,
)
from app.config.settings import TradingConfig
from app.risk.sizer import size_order
from app.risk.validator import validate_item, validate_response

logger = get_logger(__name__)

_OPEN_ACTIONS = frozenset({Action.OPEN_LONG, Action.OPEN_SHORT})


def process_decisions(
    decision_raw: str | dict | None,
    portfolio: PortfolioState,
    snapshots: dict[str, MarketSnapshot],
    cycle_state: CycleState,
    config: TradingConfig,
) -> tuple[list[ApprovedOrder], list[Rejection]]:
    """Run the full risk pipeline for one decision cycle.

    Returns ``(approved_orders, rejections)``.

    1. Parse the LLM response (Rule 1).
    2. Process close decisions first so Rule 8 bookkeeping is up-to-date.
    3. Validate each remaining item (Rules 3-5, 8, 10, 11, 13, 14).
    4. Among valid opens, select the best one (Rule 7: highest tier, then
       first listed; at most 1 new position per cycle).
    5. Check max-position limit (Rule 9) before sizing.
    6. Size approved open/add items (Rule 6).
    """
    approved: list[ApprovedOrder] = []
    rejections: list[Rejection] = []

    # ── Step 1: Parse (Rule 1) ───────────────────────────────────────
    parsed = validate_response(decision_raw)
    if parsed is None:
        logger.info(
            "Decision treated as hold-all",
            extra={"data": {"reason": "parse_failure_or_empty"}},
        )
        return approved, rejections

    # ── Step 2: Separate closes from everything else ─────────────────
    close_items: list[DecisionItem] = []
    other_items: list[DecisionItem] = []
    for item in parsed.decisions:
        if item.action == Action.CLOSE:
            close_items.append(item)
        else:
            other_items.append(item)

    # ── Step 3: Process closes first ─────────────────────────────────
    for item in close_items:
        snapshot = snapshots.get(item.symbol)
        rej = validate_item(item, portfolio, snapshot, cycle_state, config)
        if rej is not None:
            rejections.append(rej)
            _log_rejection(rej)
            continue

        pos = portfolio.positions[item.symbol]
        order = ApprovedOrder(
            symbol=item.symbol,
            action=Action.CLOSE,
            side=pos.side,
            quantity=pos.quantity,
            entry_price=snapshot.current_price if snapshot else pos.entry_price,
            invalidation_price=pos.invalidation_price,
            conviction=pos.conviction,
            confidence=None,
            notional=pos.quantity * (snapshot.current_price if snapshot else pos.entry_price),
            risk_amount=0,
            thesis=item.thesis,
            rationale=item.rationale,
        )
        approved.append(order)
        cycle_state.symbols_closed_this_cycle.add(item.symbol)
        _log_approved(order)

    # ── Step 4: Validate non-close items ─────────────────────────────
    valid_opens: list[tuple[int, DecisionItem]] = []  # (original index, item)
    valid_adds: list[DecisionItem] = []

    for idx, item in enumerate(other_items):
        if item.action == Action.HOLD:
            continue  # hold = no action required

        snapshot = snapshots.get(item.symbol)
        rej = validate_item(item, portfolio, snapshot, cycle_state, config)
        if rej is not None:
            rejections.append(rej)
            _log_rejection(rej)
            continue

        if item.action in _OPEN_ACTIONS:
            valid_opens.append((idx, item))
        elif item.action == Action.ADD:
            valid_adds.append(item)

    # ── Step 5: Rule 7 – at most 1 new position per cycle ───────────
    selected_open: DecisionItem | None = None

    if valid_opens:
        if cycle_state.new_position_opened_this_cycle:
            # All opens rejected – already opened one this cycle
            for _, item in valid_opens:
                rej = Rejection(
                    symbol=item.symbol,
                    action=item.action,
                    reason=RejectionReason.MAX_ONE_NEW_POSITION_PER_CYCLE,
                    detail="A new position was already opened this cycle.",
                )
                rejections.append(rej)
                _log_rejection(rej)
        else:
            # Sort: highest tier first (A < B lexicographically, but A is better)
            # tier A → sort key 0, tier B → sort key 1; stable sort preserves
            # original order for ties (first listed wins).
            valid_opens.sort(
                key=lambda t: 0 if t[1].conviction == ConvictionTier.A else 1,
            )
            selected_open = valid_opens[0][1]

            # Reject the rest
            for _, item in valid_opens[1:]:
                rej = Rejection(
                    symbol=item.symbol,
                    action=item.action,
                    reason=RejectionReason.MAX_ONE_NEW_POSITION_PER_CYCLE,
                    detail="Only one new position per cycle; a higher-priority open was selected.",
                )
                rejections.append(rej)
                _log_rejection(rej)

    # ── Step 6: Size the selected open ───────────────────────────────
    if selected_open is not None:
        # Rule 9: max positions check
        current_position_count = len(portfolio.positions) - len(
            # Subtract positions we're closing this cycle
            cycle_state.symbols_closed_this_cycle & set(portfolio.positions.keys())
        )
        if current_position_count >= config.max_open_positions:
            rej = Rejection(
                symbol=selected_open.symbol,
                action=selected_open.action,
                reason=RejectionReason.MAX_POSITIONS_REACHED,
                detail=(
                    f"Already at {current_position_count} positions "
                    f"(max {config.max_open_positions})."
                ),
            )
            rejections.append(rej)
            _log_rejection(rej)
        else:
            snapshot = snapshots[selected_open.symbol]
            side = Side.LONG if selected_open.action == Action.OPEN_LONG else Side.SHORT
            total_exp = _total_exposure(portfolio, cycle_state)
            result = size_order(
                symbol=selected_open.symbol,
                action=selected_open.action,
                side=side,
                entry_price=snapshot.current_price,
                invalidation_price=selected_open.invalidation_price,  # type: ignore[arg-type]
                conviction=selected_open.conviction,  # type: ignore[arg-type]
                confidence=selected_open.confidence,  # type: ignore[arg-type]
                equity=portfolio.equity,
                total_exposure=total_exp,
                existing_symbol_notional=0.0,
                cash=portfolio.cash,
                config=config,
                lot_size=snapshot.lot_size,
                min_notional=snapshot.min_notional,
                thesis=selected_open.thesis,
                rationale=selected_open.rationale,
            )
            if isinstance(result, Rejection):
                rejections.append(result)
                _log_rejection(result)
            else:
                order = _sized_to_approved(result)
                approved.append(order)
                cycle_state.new_position_opened_this_cycle = True
                _log_approved(order)

    # ── Step 7: Size valid adds ──────────────────────────────────────
    for item in valid_adds:
        pos = portfolio.positions[item.symbol]
        snapshot = snapshots[item.symbol]
        total_exp = _total_exposure(portfolio, cycle_state)
        result = size_order(
            symbol=item.symbol,
            action=Action.ADD,
            side=pos.side,
            entry_price=snapshot.current_price,
            invalidation_price=item.invalidation_price,  # type: ignore[arg-type]
            conviction=item.conviction,  # type: ignore[arg-type]
            confidence=item.confidence,  # type: ignore[arg-type]
            equity=portfolio.equity,
            total_exposure=total_exp,
            existing_symbol_notional=pos.notional,
            cash=portfolio.cash,
            config=config,
            lot_size=snapshot.lot_size,
            min_notional=snapshot.min_notional,
            thesis=item.thesis,
            rationale=item.rationale,
        )
        if isinstance(result, Rejection):
            rejections.append(result)
            _log_rejection(result)
        else:
            order = _sized_to_approved(result)
            approved.append(order)
            _log_approved(order)

    return approved, rejections


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════


def _total_exposure(portfolio: PortfolioState, cycle_state: CycleState) -> float:
    """Sum notionals of positions that aren't being closed this cycle."""
    return sum(
        pos.notional
        for sym, pos in portfolio.positions.items()
        if sym not in cycle_state.symbols_closed_this_cycle
    )


def _sized_to_approved(sized: SizedOrder) -> ApprovedOrder:
    return ApprovedOrder(
        symbol=sized.symbol,
        action=sized.action,
        side=sized.side,
        quantity=sized.quantity,
        entry_price=sized.entry_price,
        invalidation_price=sized.invalidation_price,
        conviction=sized.conviction,
        confidence=sized.confidence,
        notional=sized.notional,
        risk_amount=sized.risk_amount,
        thesis=sized.thesis,
        rationale=sized.rationale,
    )


def _log_rejection(rej: Rejection) -> None:
    logger.info(
        "Decision rejected",
        extra={
            "data": {
                "symbol": rej.symbol,
                "action": rej.action.value,
                "reason": rej.reason.value,
                "detail": rej.detail,
            }
        },
    )


def _log_approved(order: ApprovedOrder) -> None:
    logger.info(
        "Order approved",
        extra={
            "data": {
                "symbol": order.symbol,
                "action": order.action.value,
                "side": order.side.value,
                "quantity": order.quantity,
                "entry_price": order.entry_price,
                "notional": order.notional,
            }
        },
    )
