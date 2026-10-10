"""PortfolioService single-writer state engine and persistence – Rule / Phase 3.

All state mutations (orders, fills, position changes, snapshots) are strictly serialized
through an asyncio.Queue to guarantee thread-safe / coroutine-safe single-writer semantics.
Provides fast, non-blocking read-only snapshots to consumers.
Supports database persistence, restart recovery, and broker state reconciliation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.common.enums import ConvictionTier, Side
from app.common.errors import ReconciliationError
from app.common.logging import get_logger
from app.common.models import (
    ClosedTrade,
    Decision,
    EquityPoint,
    OrderFill,
    PortfolioState,
    Position,
    Rejection,
)
from app.config.settings import TradingConfig
from app.execution.base import Broker
from app.portfolio.db import (
    DecisionModel,
    EquitySnapshotModel,
    FillModel,
    OrderModel,
    PositionModel,
    RejectionModel,
    SystemStateModel,
    TradeModel,
    init_portfolio_db,
)

logger = get_logger(__name__)


@dataclass
class _MutationCommand:
    """Internal message placed on the single-writer queue."""

    action: str
    payload: dict[str, Any]
    future: asyncio.Future[Any]


@dataclass
class ReconciliationReport:
    """Result of state comparison between PortfolioService and Broker."""

    is_matched: bool
    mismatches: list[str]


class PortfolioService:
    """Single-writer Portfolio Service managing positions, equity, and database persistence."""

    def __init__(
        self,
        engine: Engine | str,
        broker: Broker,
        config: TradingConfig | None = None,
        initial_cash: float | None = None,
    ) -> None:
        self.engine = init_portfolio_db(engine)
        self.session_factory = sessionmaker(bind=self.engine)
        self.broker = broker
        self.config = config or TradingConfig()

        init_c = initial_cash if initial_cash is not None else float(self.config.starting_equity)
        self.cash: float = init_c
        self.peak_equity: float = init_c
        self.mode: str = self.config.mode.value
        self.kill_switch: bool = False
        self.agent_state: str = "RUNNING"
        self.last_cycle_id: str | None = None

        # Internal state
        self._positions: dict[str, Position] = {}
        self._equity_history: list[EquityPoint] = []
        self._entry_fees: dict[str, float] = {}

        # Single-writer serialization queue
        self._queue: asyncio.Queue[_MutationCommand | None] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._is_running: bool = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background single-writer worker."""
        if self._is_running:
            return
        self._queue = asyncio.Queue()
        self._is_running = True
        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="portfolio_service_worker"
        )
        logger.info("PortfolioService worker started")

    async def stop(self) -> None:
        """Stop worker cleanly, waiting for remaining mutations to drain."""
        if not self._is_running:
            return
        self._is_running = False
        if self._queue is not None:
            await self._queue.put(None)  # Sentinel to exit loop
        if self._worker_task:
            await self._worker_task
            self._worker_task = None
        logger.info("PortfolioService worker stopped")

    async def __aenter__(self) -> PortfolioService:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    # ── Worker Loop ──────────────────────────────────────────────────────────

    async def _worker_loop(self) -> None:
        """Sequentially process mutation commands one by one."""
        while True:
            cmd = await self._queue.get()
            if cmd is None:
                self._queue.task_done()
                break

            try:
                result = self._execute_mutation(cmd.action, cmd.payload)
                if not cmd.future.done():
                    cmd.future.set_result(result)
            except Exception as exc:
                logger.error("Error executing mutation %s: %s", cmd.action, exc)
                if not cmd.future.done():
                    cmd.future.set_exception(exc)
            finally:
                self._queue.task_done()

    async def _dispatch_mutation(self, mutation_type: str, **kwargs: Any) -> Any:
        """Submit a mutation to the single-writer queue and await its completion."""
        if not self._is_running:
            return self._execute_mutation(mutation_type, kwargs)

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        cmd = _MutationCommand(action=mutation_type, payload=kwargs, future=future)
        await self._queue.put(cmd)
        return await future

    # ── Mutation Handlers ────────────────────────────────────────────────────

    def _execute_mutation(self, action: str, payload: dict[str, Any]) -> Any:
        """Execute state update and persist to database synchronously inside worker."""
        with self.session_factory() as session:
            result = None
            if action == "RECORD_ORDER":
                result = self._handle_record_order(session, payload)
            elif action == "APPLY_OPEN_FILL":
                result = self._handle_apply_open_fill(session, payload)
            elif action == "APPLY_CLOSE_FILL":
                result = self._handle_apply_close_fill(session, payload)
            elif action == "RECORD_EQUITY_SNAPSHOT":
                result = self._handle_record_equity_snapshot(session, payload)
            elif action == "RECORD_DECISION":
                result = self._handle_record_decision(session, payload)
            elif action == "RECORD_REJECTIONS":
                result = self._handle_record_rejections(session, payload)
            elif action == "SET_CASH":
                result = self._handle_set_cash(session, payload)
            elif action == "SYNC_STATE":
                result = self._handle_sync_state(session, payload)
            else:
                raise ValueError(f"Unknown mutation action: {action}")

            session.commit()
            return result

    def _handle_record_order(self, session: Session, payload: dict[str, Any]) -> OrderModel:
        order = OrderModel(
            client_order_id=payload["client_order_id"],
            symbol=payload["symbol"],
            side=payload["side"],
            action=payload["action"],
            quantity=payload["quantity"],
            price=payload.get("price"),
            status=payload.get("status", "filled"),
            created_at=payload.get("created_at") or datetime.now(UTC),
        )
        session.merge(order)
        return order

    def _handle_apply_open_fill(self, session: Session, payload: dict[str, Any]) -> Position:
        fill: OrderFill = payload["fill"]
        invalidation_price: float = payload["invalidation_price"]
        conviction: ConvictionTier = payload["conviction"]
        risk_amount: float = payload["risk_amount"]

        # Record fill in database
        fill_record = FillModel(
            fill_id=fill.fill_id,
            client_order_id=fill.client_order_id,
            symbol=fill.symbol,
            side=fill.side.value,
            price=fill.price,
            quantity=fill.quantity,
            fee=fill.fee,
            slippage_bps=fill.slippage_bps,
            timestamp=fill.timestamp,
        )
        session.merge(fill_record)

        # Deduct cash
        cost = (fill.price * fill.quantity) + fill.fee
        self.cash -= cost

        if fill.symbol in self._positions:
            existing = self._positions[fill.symbol]
            new_qty = existing.quantity + fill.quantity
            avg_entry = (
                (existing.entry_price * existing.quantity) + (fill.price * fill.quantity)
            ) / new_qty
            new_risk = existing.risk_amount + risk_amount
            pos = Position(
                symbol=fill.symbol,
                side=fill.side,
                entry_price=avg_entry,
                quantity=new_qty,
                invalidation_price=existing.invalidation_price,
                conviction=existing.conviction,
                risk_amount=new_risk,
                opened_at=existing.opened_at,
            )
            self._entry_fees[fill.symbol] = self._entry_fees.get(fill.symbol, 0.0) + fill.fee
        else:
            pos = Position(
                symbol=fill.symbol,
                side=fill.side,
                entry_price=fill.price,
                quantity=fill.quantity,
                invalidation_price=invalidation_price,
                conviction=conviction,
                risk_amount=risk_amount,
                opened_at=fill.timestamp,
            )
            self._entry_fees[fill.symbol] = fill.fee

        self._positions[fill.symbol] = pos

        # Update positions table
        pos_record = PositionModel(
            symbol=pos.symbol,
            side=pos.side.value,
            entry_price=pos.entry_price,
            quantity=pos.quantity,
            invalidation_price=pos.invalidation_price,
            conviction=pos.conviction.value,
            risk_amount=pos.risk_amount,
            opened_at=pos.opened_at,
            status="OPEN",
        )
        session.merge(pos_record)

        # Update system state cash
        self._persist_system_state(session)
        return pos

    def _handle_apply_close_fill(
        self, session: Session, payload: dict[str, Any]
    ) -> ClosedTrade | None:
        fill: OrderFill = payload["fill"]
        reason: str = payload.get("reason", "")

        if fill.symbol not in self._positions:
            return None

        pos = self._positions[fill.symbol]

        # Record fill
        fill_record = FillModel(
            fill_id=fill.fill_id,
            client_order_id=fill.client_order_id,
            symbol=fill.symbol,
            side=fill.side.value,
            price=fill.price,
            quantity=fill.quantity,
            fee=fill.fee,
            slippage_bps=fill.slippage_bps,
            timestamp=fill.timestamp,
        )
        session.merge(fill_record)

        entry_fee = self._entry_fees.pop(fill.symbol, 0.0)
        total_fees = entry_fee + fill.fee

        # PnL computation
        if pos.side == Side.LONG:
            gross_proceeds = fill.price * pos.quantity
            net_proceeds = gross_proceeds - fill.fee
            entry_cost = pos.entry_price * pos.quantity
            realized_pnl = (gross_proceeds - entry_cost) - total_fees
            self.cash += net_proceeds
        else:
            entry_collateral = pos.entry_price * pos.quantity
            realized_pnl = ((pos.entry_price - fill.price) * pos.quantity) - total_fees
            self.cash += entry_collateral + (pos.entry_price - fill.price) * pos.quantity - fill.fee

        r_multiple = (realized_pnl / pos.risk_amount) if pos.risk_amount > 0 else 0.0

        closed_trade = ClosedTrade(
            trade_id=f"trade_{fill.client_order_id}",
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=fill.price,
            quantity=pos.quantity,
            entry_time=pos.opened_at,
            exit_time=fill.timestamp,
            realized_pnl=realized_pnl,
            r_multiple=r_multiple,
            fees_total=total_fees,
            exit_reason=reason,
        )

        # Remove from active positions in memory and DB
        del self._positions[pos.symbol]
        session.execute(delete(PositionModel).where(PositionModel.symbol == pos.symbol))

        # Insert Trade record
        trade_record = TradeModel(
            trade_id=closed_trade.trade_id,
            symbol=closed_trade.symbol,
            side=closed_trade.side.value,
            entry_price=closed_trade.entry_price,
            exit_price=closed_trade.exit_price,
            quantity=closed_trade.quantity,
            entry_time=closed_trade.entry_time,
            exit_time=closed_trade.exit_time,
            realized_pnl=closed_trade.realized_pnl,
            r_multiple=closed_trade.r_multiple,
            fees_total=closed_trade.fees_total,
            exit_reason=closed_trade.exit_reason,
        )
        session.merge(trade_record)

        self._persist_system_state(session)
        return closed_trade

    def _handle_record_equity_snapshot(
        self, session: Session, payload: dict[str, Any]
    ) -> EquityPoint:
        prices: dict[str, float] = payload.get("prices", {})
        ts: datetime = payload.get("timestamp") or datetime.now(UTC)

        equity = self._calculate_equity(prices)
        exposure = self._calculate_exposure(prices)

        self.peak_equity = max(self.peak_equity, equity)
        pt = EquityPoint(timestamp=ts, equity=equity)
        self._equity_history.append(pt)

        # Save to DB
        snap = EquitySnapshotModel(
            timestamp=ts,
            equity=equity,
            cash=self.cash,
            exposure=exposure,
            realized_pnl=payload.get("realized_pnl", 0.0),
            unrealized_pnl=payload.get("unrealized_pnl", 0.0),
        )
        session.add(snap)
        self._persist_system_state(session)
        return pt

    def _handle_record_decision(self, session: Session, payload: dict[str, Any]) -> None:
        dec = DecisionModel(
            cycle_id=payload["cycle_id"],
            timestamp=payload.get("timestamp") or datetime.now(UTC),
            raw_response=payload["raw_response"],
            parsed_json=payload["parsed_json"],
            reasoning=payload.get("reasoning"),
        )
        session.add(dec)

    def _handle_record_rejections(self, session: Session, payload: dict[str, Any]) -> None:
        rejections: list[Rejection] = payload["rejections"]
        cid: str = payload["cycle_id"]
        ts: datetime = payload.get("timestamp") or datetime.now(UTC)
        for r in rejections:
            rec = RejectionModel(
                cycle_id=cid,
                timestamp=ts,
                symbol=r.symbol,
                action=r.action.value,
                reason=r.reason.value,
                detail=r.detail,
            )
            session.add(rec)

    def _handle_set_cash(self, session: Session, payload: dict[str, Any]) -> float:
        self.cash = float(payload["cash"])
        self._persist_system_state(session)
        return self.cash

    def _handle_sync_state(self, session: Session, payload: dict[str, Any]) -> None:
        for k, v in payload.items():
            setattr(self, k, v)
        self._persist_system_state(session)

    def _persist_system_state(self, session: Session) -> None:
        """Write current system state to system_state table."""
        row = SystemStateModel(
            id=1,
            mode=self.mode,
            breaker_state_json="{}",
            kill_switch=self.kill_switch,
            last_cycle_id=self.last_cycle_id,
            agent_state=self.agent_state,
            cash=self.cash,
            peak_equity=self.peak_equity,
            updated_at=datetime.now(UTC),
        )
        session.merge(row)

    # ── Calculation Helpers ──────────────────────────────────────────────────

    def _calculate_equity(self, prices: dict[str, float] | None = None) -> float:
        prices = prices or {}
        pos_val = 0.0
        for sym, pos in self._positions.items():
            price = prices.get(sym, pos.entry_price)
            if pos.side == Side.LONG:
                pos_val += pos.quantity * price
            else:
                # Short value: collateral + unrealized pnl
                pos_val += (pos.quantity * pos.entry_price) + pos.unrealized_pnl(price)
        return self.cash + pos_val

    def _calculate_exposure(self, prices: dict[str, float] | None = None) -> float:
        prices = prices or {}
        return sum(
            pos.quantity * prices.get(sym, pos.entry_price) for sym, pos in self._positions.items()
        )

    # ── Public Async API ─────────────────────────────────────────────────────

    async def record_order(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        action: str,
        quantity: float,
        price: float | None = None,
        status: str = "filled",
    ) -> Any:
        """Record an order synchronously through queue."""
        return await self._dispatch_mutation(
            "RECORD_ORDER",
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            action=action,
            quantity=quantity,
            price=price,
            status=status,
        )

    async def apply_open_fill(
        self,
        fill: OrderFill,
        invalidation_price: float,
        conviction: ConvictionTier,
        risk_amount: float,
        action: str = "open",
    ) -> Position:
        """Apply an order fill opening or adding to a position."""
        return await self._dispatch_mutation(
            "APPLY_OPEN_FILL",
            fill=fill,
            invalidation_price=invalidation_price,
            conviction=conviction,
            risk_amount=risk_amount,
            action=action,
        )

    async def apply_close_fill(
        self,
        fill: OrderFill,
        reason: str = "",
    ) -> ClosedTrade | None:
        """Apply an order fill closing a position."""
        return await self._dispatch_mutation(
            "APPLY_CLOSE_FILL",
            fill=fill,
            reason=reason,
        )

    async def record_equity_snapshot(
        self,
        prices: dict[str, float] | None = None,
        timestamp: datetime | None = None,
        realized_pnl: float = 0.0,
        unrealized_pnl: float = 0.0,
    ) -> EquityPoint:
        """Record an equity point."""
        return await self._dispatch_mutation(
            "RECORD_EQUITY_SNAPSHOT",
            prices=prices or {},
            timestamp=timestamp,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
        )

    async def record_decisions(
        self,
        cycle_id: str,
        raw_response: str,
        parsed_decision: Decision,
        timestamp: datetime | None = None,
    ) -> None:
        """Save raw and parsed LLM decisions."""
        await self._dispatch_mutation(
            "RECORD_DECISION",
            cycle_id=cycle_id,
            timestamp=timestamp,
            raw_response=raw_response,
            parsed_json=parsed_decision.model_dump_json(),
            reasoning=parsed_decision.reasoning,
        )

    async def record_rejections(
        self,
        cycle_id: str,
        rejections: list[Rejection],
        timestamp: datetime | None = None,
    ) -> None:
        """Save rejections."""
        if not rejections:
            return
        await self._dispatch_mutation(
            "RECORD_REJECTIONS",
            cycle_id=cycle_id,
            rejections=rejections,
            timestamp=timestamp,
        )

    async def set_cash(self, cash: float) -> float:
        """Update cash level."""
        return await self._dispatch_mutation("SET_CASH", cash=cash)

    async def set_agent_state(self, state: str) -> None:
        """Update agent operational state (RUNNING, PAUSED_KILL, etc.)."""
        self.agent_state = state
        await self._dispatch_mutation("SYNC_STATE", agent_state=state)

    # ── Read-Only Views ──────────────────────────────────────────────────────

    def get_state(self, prices: dict[str, float] | None = None) -> PortfolioState:
        """Return an immutable snapshot of current PortfolioState for the Risk Engine."""
        prices = prices or {}
        eq = self._calculate_equity(prices)
        return PortfolioState(
            equity=eq,
            cash=self.cash,
            positions=dict(self._positions),
            peak_equity=self.peak_equity,
            equity_history=list(self._equity_history),
        )

    def compute_equity(self, prices: dict[str, float] | None = None) -> float:
        """Compute current equity using provided or entry prices."""
        return self._calculate_equity(prices)

    def compute_exposure(self, prices: dict[str, float] | None = None) -> float:
        """Compute total position notional exposure."""
        return self._calculate_exposure(prices)

    def compute_unrealized_r(self, symbol: str, current_price: float) -> float:
        """Compute unrealized R multiple for an open position."""
        pos = self._positions.get(symbol)
        if not pos:
            return 0.0
        return pos.unrealized_r(current_price)

    # ── Persistence & Restart Recovery ───────────────────────────────────────

    @classmethod
    def load_from_db(
        cls,
        engine: Engine | str,
        broker: Broker,
        config: TradingConfig | None = None,
    ) -> PortfolioService:
        """Reconstruct PortfolioService state from database tables."""
        service = cls(engine=engine, broker=broker, config=config)

        with service.session_factory() as session:
            # 1. System state
            state_row = session.scalar(select(SystemStateModel).where(SystemStateModel.id == 1))
            if state_row:
                service.cash = float(state_row.cash)
                service.peak_equity = float(state_row.peak_equity)
                service.mode = state_row.mode
                service.kill_switch = bool(state_row.kill_switch)
                service.last_cycle_id = state_row.last_cycle_id
                service.agent_state = getattr(state_row, "agent_state", "RUNNING") or "RUNNING"
            else:
                # Save initial system state
                service._persist_system_state(session)
                session.commit()

            # 2. Open positions
            pos_rows = session.scalars(
                select(PositionModel).where(PositionModel.status == "OPEN")
            ).all()
            for p in pos_rows:
                opened_at = p.opened_at
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=UTC)
                pos = Position(
                    symbol=p.symbol,
                    side=Side(p.side),
                    entry_price=p.entry_price,
                    quantity=p.quantity,
                    invalidation_price=p.invalidation_price,
                    conviction=ConvictionTier(p.conviction),
                    risk_amount=p.risk_amount,
                    opened_at=opened_at,
                )
                service._positions[p.symbol] = pos

            # 3. Equity history
            history_rows = session.scalars(
                select(EquitySnapshotModel).order_by(EquitySnapshotModel.timestamp.asc())
            ).all()
            service._equity_history = [
                EquityPoint(
                    timestamp=(
                        h.timestamp.replace(tzinfo=UTC)
                        if h.timestamp.tzinfo is None
                        else h.timestamp
                    ),
                    equity=h.equity,
                )
                for h in history_rows
            ]

        logger.info(
            "PortfolioService recovered from DB: cash=%.2f, positions=%d, history_points=%d",
            service.cash,
            len(service._positions),
            len(service._equity_history),
        )
        return service

    async def reconcile(
        self, strict: bool = True, tolerance_qty: float = 1e-5
    ) -> ReconciliationReport:
        """Compare internal state with external broker state and identify discrepancies."""
        broker_positions = await self.broker.get_open_positions()
        broker_balances = await self.broker.get_balances()

        mismatches: list[str] = []

        # 1. Compare open positions symbols
        internal_symbols = set(self._positions.keys())
        broker_symbols = set(broker_positions.keys())

        missing_in_broker = internal_symbols - broker_symbols
        for s in missing_in_broker:
            mismatches.append(f"Position {s} exists in PortfolioService but missing in Broker")

        missing_in_internal = broker_symbols - internal_symbols
        for s in missing_in_internal:
            mismatches.append(f"Position {s} exists in Broker but missing in PortfolioService")

        # 2. Compare quantities & sides of overlapping positions
        for s in internal_symbols & broker_symbols:
            ipos = self._positions[s]
            bpos = broker_positions[s]
            if ipos.side != bpos.side:
                mismatches.append(
                    f"Position {s} side mismatch: internal {ipos.side} vs broker {bpos.side}"
                )
            if abs(ipos.quantity - bpos.quantity) > tolerance_qty:
                mismatches.append(
                    f"Position {s} quantity mismatch: internal {ipos.quantity} vs broker {bpos.quantity}"
                )

        # 3. Compare cash
        broker_cash = broker_balances.get("cash")
        if broker_cash is not None and abs(self.cash - broker_cash) > 1.0:
            mismatches.append(
                f"Cash mismatch: internal ${self.cash:.2f} vs broker ${broker_cash:.2f}"
            )

        # 4. Compare open orders with internal state
        broker_orders = await self.broker.get_open_orders()
        # In live execution, verify that resting orders match expected internal stop orders
        if hasattr(self.broker, "_stop_orders"):
            tracked_stops = getattr(self.broker, "_stop_orders", {})
            broker_order_ids = {str(o.get("id")) for o in broker_orders if o.get("id")}

            # Check that every tracked stop order exists on exchange
            for sym, stop_info in tracked_stops.items():
                oid = str(stop_info.get("order_id"))
                if oid and oid not in broker_order_ids:
                    mismatches.append(
                        f"Expected stop order {oid} for {sym} missing from exchange open orders"
                    )

            # Check for unknown / rogue orders on the exchange
            tracked_order_ids = {str(info.get("order_id")) for info in tracked_stops.values() if info.get("order_id")}
            for o in broker_orders:
                oid = str(o.get("id"))
                if oid not in tracked_order_ids:
                    mismatches.append(
                        f"Unrecognized open order {oid} ({o.get('symbol')} {o.get('type')}) on exchange"
                    )

        is_matched = len(mismatches) == 0
        report = ReconciliationReport(is_matched=is_matched, mismatches=mismatches)

        if not is_matched:
            logger.warning("Reconciliation found discrepancies: %s", mismatches)
            if strict:
                raise ReconciliationError(
                    f"Broker reconciliation failed with {len(mismatches)} discrepancies: "
                    + "; ".join(mismatches)
                )

        return report
