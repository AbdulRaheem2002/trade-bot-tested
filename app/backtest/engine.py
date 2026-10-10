"""Deterministic Backtesting Engine – Rule / Phase 7.

Features:
- Chronological candle replay with strict zero-look-ahead snapshot building.
- Re-uses the EXACT live risk pipeline:
    `process_decisions()` → `validator` → `sizer` → `breakers`.
- Re-uses the EXACT `PaperBroker` execution model (slippage and taker fees).
- Intra-candle invalidation monitor on high/lows:
    * Standard stop: filled at invalidation price + slippage/fees.
    * Gap-through-stop: filled at candle OPEN + slippage/fees.
- Walk-forward train/test splitting (In-Sample vs Out-of-Sample) with loud overfitting warnings.
- Equal-weight buy-and-hold universe benchmark.
- Markdown report generation and performance metrics computation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.backtest.strategy import Strategy, get_strategy
from app.common.enums import Action, Side
from app.common.logging import get_logger
from app.common.models import (
    CycleState,
    Decision,
    EquityPoint,
    MarketSnapshot,
    PortfolioState,
    Rejection,
)
from app.config.loader import load_config
from app.config.settings import TradingConfig
from app.data.snapshot import build_snapshot
from app.data.store import CandleModel, CandleStore, get_default_store
from app.execution.paper_broker import PaperBroker
from app.portfolio.metrics import PerformanceMetrics, compute_metrics
from app.risk.breakers import compute_breaker_state
from app.risk.risk_engine import process_decisions

logger = get_logger(__name__)


@dataclass
class BacktestTrade:
    """Record of a closed trade during backtesting."""

    trade_id: str
    symbol: str
    side: Side
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    realized_pnl: float
    fees_total: float
    r_multiple: float
    exit_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "realized_pnl": self.realized_pnl,
            "fees_total": self.fees_total,
            "r_multiple": self.r_multiple,
            "exit_reason": self.exit_reason,
        }


@dataclass
class EquitySnapshotPoint:
    """Historical equity point for performance charting."""

    timestamp: datetime
    equity: float
    cash: float
    exposure: float
    realized_pnl: float
    unrealized_pnl: float


@dataclass
class BacktestResult:
    """Comprehensive output of a backtest run."""

    strategy_name: str
    symbols: list[str]
    start_time: datetime
    end_time: datetime
    initial_capital: float
    final_equity: float
    metrics: PerformanceMetrics
    trades: list[BacktestTrade]
    equity_curve: list[EquitySnapshotPoint]
    benchmark_return_pct: float
    benchmark_max_drawdown_pct: float
    rejections: list[Rejection]
    # Walk-forward splits (if split was used)
    is_metrics: PerformanceMetrics | None = None
    oos_metrics: PerformanceMetrics | None = None
    is_trades_count: int = 0
    oos_trades_count: int = 0
    walk_forward_warning: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy_name,
            "symbols": self.symbols,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "metrics": self.metrics.to_dict(),
            "benchmark_return_pct": self.benchmark_return_pct,
            "benchmark_max_drawdown_pct": self.benchmark_max_drawdown_pct,
            "total_trades": len(self.trades),
        }


class BacktestEngine:
    """Deterministic, zero-lookahead backtesting engine."""

    def __init__(
        self,
        strategy: Strategy | str,
        config: TradingConfig | None = None,
        initial_cash: float | None = None,
        store: CandleStore | None = None,
    ) -> None:
        self.config = config or load_config()
        if isinstance(strategy, str):
            self.strategy = get_strategy(strategy)
        else:
            self.strategy = strategy

        self.initial_cash = initial_cash or self.config.starting_equity
        self.store = store or get_default_store()
        self.broker = PaperBroker(
            initial_cash=self.initial_cash,
            slippage_bps=self.config.slippage_bps,
            fees_taker_bps=self.config.fees_taker_bps,
            shorts_enabled=self.config.shorts_enabled,
        )

        # Bookkeeping
        self.trades: list[BacktestTrade] = []
        self.equity_curve: list[EquitySnapshotPoint] = []
        self.rejections: list[Rejection] = []
        self.breaker_paused = False

    # ── Replay Candle Extraction ─────────────────────────────────────────────

    def get_replay_timestamps(
        self,
        symbols: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[int]:
        """Fetch distinct 4h candle open timestamps common across requested universe."""
        with self.store.Session() as session:
            stmt = (
                select(CandleModel.open_time)
                .where(CandleModel.symbol.in_(symbols))
                .where(CandleModel.timeframe == "4h")
                .distinct()
                .order_by(CandleModel.open_time.asc())
            )
            if start_time:
                stmt = stmt.where(CandleModel.open_time >= int(start_time.timestamp() * 1000))
            if end_time:
                stmt = stmt.where(CandleModel.open_time <= int(end_time.timestamp() * 1000))

            timestamps = session.scalars(stmt).all()
        return timestamps

    # ── Benchmark Computation ────────────────────────────────────────────────

    def compute_buy_and_hold_benchmark(
        self,
        symbols: list[str],
        timestamps: list[int],
    ) -> tuple[float, float]:
        """Compute equal-weight buy-and-hold basket return and max drawdown."""
        if not timestamps:
            return 0.0, 0.0

        baskets: list[float] = []
        initial_prices: dict[str, float] = {}

        for sym in symbols:
            df = self.store.get_candles(sym, "4h")
            if not df.empty:
                # Find closest price to first timestamp
                row = df[df["open_time"] >= timestamps[0]]
                if not row.empty:
                    initial_prices[sym] = float(row.iloc[0]["open"])

        if not initial_prices:
            return 0.0, 0.0

        for ts in timestamps:
            returns = []
            for sym, init_p in initial_prices.items():
                df = self.store.get_candles(sym, "4h")
                match = df[df["open_time"] == ts]
                if not match.empty and init_p > 0:
                    curr_p = float(match.iloc[0]["close"])
                    returns.append(curr_p / init_p)
            if returns:
                baskets.append(sum(returns) / len(returns))

        if not baskets:
            return 0.0, 0.0

        final_ret_pct = (baskets[-1] - 1.0) * 100.0

        # Max drawdown of basket
        peak = baskets[0]
        max_dd = 0.0
        for val in baskets:
            if val > peak:
                peak = val
            dd = ((peak - val) / peak) * 100.0 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        return final_ret_pct, max_dd

    # ── Invalidation Evaluation ──────────────────────────────────────────────

    async def _evaluate_intra_candle_invalidations(
        self,
        candle_ts_ms: int,
        cycle_time: datetime,
    ) -> None:
        """Evaluate open positions against current candle extremes (high/low).

        Rules:
        - Long: touched if candle low <= invalidation_price.
          * Gap-through-stop: if candle open < invalidation_price, fills at open.
          * Standard stop: fills at invalidation_price.
        - Short: touched if candle high >= invalidation_price.
          * Gap-through-stop: if candle open > invalidation_price, fills at open.
          * Standard stop: fills at invalidation_price.
        """
        open_positions = list(self.broker._positions.values())

        for pos in open_positions:
            sym = pos.symbol
            with self.store.Session() as session:
                stmt = (
                    select(CandleModel)
                    .where(CandleModel.symbol == sym)
                    .where(CandleModel.timeframe == "4h")
                    .where(CandleModel.open_time == candle_ts_ms)
                )
                candle = session.scalar(stmt)

            if candle is None:
                continue

            c_open = float(candle.open)
            c_high = float(candle.high)
            c_low = float(candle.low)

            is_stopped = False
            base_exit_price = pos.invalidation_price

            if pos.side == Side.LONG:
                if c_low <= pos.invalidation_price:
                    is_stopped = True
                    # Check gap-through-stop at open
                    if c_open < pos.invalidation_price:
                        base_exit_price = c_open
                    else:
                        base_exit_price = pos.invalidation_price

            elif pos.side == Side.SHORT and c_high >= pos.invalidation_price:
                is_stopped = True
                # Check gap-through-stop at open
                if c_open > pos.invalidation_price:
                    base_exit_price = c_open
                else:
                    base_exit_price = pos.invalidation_price

            if is_stopped:
                # Set price and close via broker fill model
                self.broker.set_market_price(sym, base_exit_price)
                fill = await self.broker.close_position(
                    symbol=sym,
                    reason="invalidation",
                    timestamp=cycle_time,
                )
                if fill:
                    r_mult = (
                        (self.broker.realized_pnl / pos.risk_amount)
                        if pos.risk_amount > 0
                        else 0.0
                    )
                    trade = BacktestTrade(
                        trade_id=f"bt_{uuid.uuid4().hex[:8]}",
                        symbol=sym,
                        side=pos.side,
                        entry_time=pos.opened_at,
                        exit_time=cycle_time,
                        entry_price=pos.entry_price,
                        exit_price=fill.price,
                        quantity=fill.quantity,
                        realized_pnl=(
                            (fill.price - pos.entry_price) * fill.quantity - fill.fee
                            if pos.side == Side.LONG
                            else (pos.entry_price - fill.price) * fill.quantity - fill.fee
                        ),
                        fees_total=fill.fee,
                        r_multiple=r_mult,
                        exit_reason="invalidation",
                    )
                    self.trades.append(trade)

    # ── Snapshot Construction with Zero Look-Ahead ───────────────────────────

    async def _build_cycle_snapshot(
        self,
        symbol: str,
        cycle_time: datetime,
    ) -> MarketSnapshot | None:
        """Build MarketSnapshot with strict zero look-ahead using the Phase 2 builder."""
        try:
            return await build_snapshot(
                symbol=symbol,
                store=self.store,
                client=None,
                config=self.config,
                now=cycle_time,
            )
        except Exception as exc:
            logger.debug("Could not build snapshot for %s at %s: %s", symbol, cycle_time, exc)
    def _get_portfolio_state(self, current_prices: dict[str, float]) -> PortfolioState:
        """Derive validated PortfolioState from PaperBroker state and current prices."""
        positions = dict(self.broker._positions)
        unrealized = 0.0
        for sym, pos in positions.items():
            px = current_prices.get(sym, pos.entry_price)
            unrealized += pos.unrealized_pnl(px)

        long_value = sum(
            pos.quantity * current_prices.get(sym, pos.entry_price)
            for sym, pos in positions.items()
            if pos.side == Side.LONG
        )
        short_value = sum(
            pos.quantity * pos.entry_price
            + pos.unrealized_pnl(current_prices.get(sym, pos.entry_price))
            for sym, pos in positions.items()
            if pos.side == Side.SHORT
        )
        equity = self.broker.cash + long_value + short_value
        peak = max(equity, getattr(self, "_peak_equity", self.initial_cash))
        self._peak_equity = peak
        self._last_unrealized_pnl = unrealized

        return PortfolioState(
            cash=self.broker.cash,
            equity=equity,
            peak_equity=peak,
            positions=positions,
            equity_history=[
                EquityPoint(timestamp=p.timestamp, equity=p.equity)
                for p in self.equity_curve
            ],
        )

    # ── Main Backtest Runner ─────────────────────────────────────────────────

    async def run(
        self,
        symbols: list[str] | None = None,
        days: int = 365,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        train_test_split: float = 0.7,
    ) -> BacktestResult:
        """Run backtest across requested symbols and time horizon.

        Args:
            symbols: Universe of symbols to trade (default: config.universe).
            days: Lookback days if start_time not provided.
            start_time: Explicit start timestamp.
            end_time: Explicit end timestamp (default: now).
            train_test_split: Walk-forward in-sample fraction (0.0 < split <= 1.0).
        """
        universe = symbols or self.config.universe
        end_dt = end_time or datetime.now(UTC)
        start_dt = start_time or (end_dt - timedelta(days=days))

        # Reset engine state
        self.trades.clear()
        self.equity_curve.clear()
        self.rejections.clear()
        self.breaker_paused = False
        self.broker = PaperBroker(
            initial_cash=self.initial_cash,
            slippage_bps=self.config.slippage_bps,
            fees_taker_bps=self.config.fees_taker_bps,
            shorts_enabled=self.config.shorts_enabled,
        )

        # 1. Fetch chronological candle open timestamps
        timestamps = self.get_replay_timestamps(universe, start_dt, end_dt)
        if not timestamps:
            # Fall back to all available candles in store if requested slice is empty
            timestamps = self.get_replay_timestamps(universe)

        if not timestamps:
            raise ValueError(
                f"No 4h candles found in store for symbols {universe}. Please backfill or load candles first."
            )

        # We require at least 55 candles of initial history for technical indicators (EMA50/EMA200/ATR)
        history_warmup = min(55, len(timestamps) - 1)
        active_timestamps = timestamps[history_warmup:]
        if not active_timestamps:
            raise ValueError(
                f"Insufficient historical candles ({len(timestamps)}) to compute indicators."
            )

        # Walk-forward split determination
        walk_forward_warning = False
        split_idx = int(len(active_timestamps) * train_test_split)

        if train_test_split >= 1.0 or split_idx >= len(active_timestamps):
            walk_forward_warning = True
            print("\n" + "=" * 80)
            print("⚠️  [WALK-FORWARD OVERFITTING WARNING]")
            print("Strategy is evaluated solely IN-SAMPLE without out-of-sample testing!")
            print("To ensure strategy robustness and eliminate curve-fitting bias,")
            print("always validate across an out-of-sample evaluation period (e.g. 70/30 split).")
            print("=" * 80 + "\n", flush=True)

        is_split_ts = active_timestamps[split_idx] if not walk_forward_warning else None
        is_trades: list[BacktestTrade] = []
        oos_trades: list[BacktestTrade] = []
        is_snapshots: list[EquitySnapshotPoint] = []
        oos_snapshots: list[EquitySnapshotPoint] = []

        # 2. Step chronologically through candles
        for ts_ms in active_timestamps:
            candle_dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
            # Candle close occurs 4h after candle open
            cycle_time = candle_dt + timedelta(hours=4) + timedelta(seconds=self.config.candle_close_delay_seconds)

            # Step A: Evaluate intra-candle stops on high/lows
            await self._evaluate_intra_candle_invalidations(ts_ms, cycle_time)

            # Step B: Build zero look-ahead snapshots for all universe symbols
            snapshots: dict[str, MarketSnapshot] = {}
            for sym in universe:
                snap = await self._build_cycle_snapshot(sym, cycle_time)
                if snap is not None:
                    snapshots[sym] = snap
                    # Update broker market price to latest close
                    self.broker.set_market_price(sym, snap.current_price)
                    self.broker.set_market_rules(
                        sym,
                        lot_size=snap.lot_size,
                        min_notional=snap.min_notional,
                    )

            if not snapshots:
                continue

            # Step C: Evaluate strategy to collect candidate decisions
            current_prices = {s: sn.current_price for s, sn in snapshots.items()}
            port_state = self._get_portfolio_state(current_prices)

            combined_decisions = []
            for sym in universe:
                if sym in snapshots:
                    dec = self.strategy.on_candle_close(snapshots[sym], port_state)
                    combined_decisions.extend(dec.decisions)

            cycle_decision = Decision(
                reasoning=f"Backtest cycle {cycle_time.isoformat()}",
                decisions=combined_decisions,
            )

            # Step D: Circuit breaker evaluation
            breaker_state = compute_breaker_state(
                current_equity=port_state.equity,
                peak_equity=port_state.peak_equity,
                equity_history=[
                    type("Point", (), {"timestamp": p.timestamp, "equity": p.equity})()
                    for p in self.equity_curve
                ],
                config=self.config,
                now=cycle_time,
                previously_paused=self.breaker_paused,
            )
            if breaker_state.is_paused:
                self.breaker_paused = True

            cycle_state = CycleState(
                cycle_time=cycle_time,
                expected_candle_time=candle_dt,
                breaker_state=breaker_state,
            )

            # Step E: Risk Engine Pipeline (Same validator, sizer, and rules)
            approved_orders, cycle_rejections = process_decisions(
                decision_raw=cycle_decision,
                portfolio=port_state,
                snapshots=snapshots,
                cycle_state=cycle_state,
                config=self.config,
            )
            self.rejections.extend(cycle_rejections)

            # Step F: Execute Approved Orders through PaperBroker fill model
            for order in approved_orders:
                if order.action == Action.CLOSE:
                    pos = port_state.positions.get(order.symbol)
                    fill = await self.broker.close_position(
                        symbol=order.symbol,
                        reason="strategy_close",
                        timestamp=cycle_time,
                    )
                    if fill and pos:
                        trade = BacktestTrade(
                            trade_id=f"bt_{uuid.uuid4().hex[:8]}",
                            symbol=order.symbol,
                            side=pos.side,
                            entry_time=pos.opened_at,
                            exit_time=cycle_time,
                            entry_price=pos.entry_price,
                            exit_price=fill.price,
                            quantity=fill.quantity,
                            realized_pnl=(
                                (fill.price - pos.entry_price) * fill.quantity - fill.fee
                                if pos.side == Side.LONG
                                else (pos.entry_price - fill.price) * fill.quantity - fill.fee
                            ),
                            fees_total=fill.fee,
                            r_multiple=(
                                (self.broker.realized_pnl / pos.risk_amount)
                                if pos.risk_amount > 0
                                else 0.0
                            ),
                            exit_reason="strategy_close",
                        )
                        self.trades.append(trade)

                elif order.action in (Action.OPEN_LONG, Action.OPEN_SHORT):
                    try:
                        await self.broker.place_market_order(
                            symbol=order.symbol,
                            side=order.side,
                            qty=order.quantity,
                            client_order_id=f"bt_open_{uuid.uuid4().hex[:8]}",
                            invalidation_price=order.invalidation_price,
                            conviction=order.conviction,
                            risk_amount=order.risk_amount,
                            timestamp=cycle_time,
                        )
                    except Exception as exc:
                        logger.debug("Backtest order rejected by broker: %s", exc)

            # Step G: Record Equity Snapshot
            current_port = self._get_portfolio_state(current_prices)
            total_notional = sum(
                p.quantity * current_prices.get(p.symbol, p.entry_price)
                for p in current_port.positions.values()
            )
            exposure_pct = (
                (total_notional / current_port.equity * 100.0) if current_port.equity > 0 else 0.0
            )

            point = EquitySnapshotPoint(
                timestamp=cycle_time,
                equity=round(current_port.equity, 2),
                cash=round(current_port.cash, 2),
                exposure=round(exposure_pct, 2),
                realized_pnl=round(self.broker.realized_pnl, 2),
                unrealized_pnl=round(getattr(self, "_last_unrealized_pnl", 0.0), 2),
            )
            self.equity_curve.append(point)

            # Walk-forward segment tracking
            if is_split_ts is not None:
                if ts_ms < is_split_ts:
                    is_snapshots.append(point)
                else:
                    oos_snapshots.append(point)

        # Segment trades by exit time
        if is_split_ts is not None:
            split_dt = datetime.fromtimestamp(is_split_ts / 1000, tz=UTC)
            for t in self.trades:
                if t.exit_time < split_dt:
                    is_trades.append(t)
                else:
                    oos_trades.append(t)

        # 3. Compute Metrics
        overall_metrics = compute_metrics(
            trades=[type("T", (), t.to_dict())() for t in self.trades],
            snapshots=[type("S", (), {"equity": p.equity})() for p in self.equity_curve],
        )

        is_metrics = None
        oos_metrics = None
        if is_snapshots and not walk_forward_warning:
            is_metrics = compute_metrics(
                trades=[type("T", (), t.to_dict())() for t in is_trades],
                snapshots=[type("S", (), {"equity": p.equity})() for p in is_snapshots],
            )
        if oos_snapshots and not walk_forward_warning:
            oos_metrics = compute_metrics(
                trades=[type("T", (), t.to_dict())() for t in oos_trades],
                snapshots=[type("S", (), {"equity": p.equity})() for p in oos_snapshots],
            )

        # 4. Compute Benchmark
        bm_ret, bm_dd = self.compute_buy_and_hold_benchmark(universe, active_timestamps)

        first_dt = datetime.fromtimestamp(active_timestamps[0] / 1000, tz=UTC)
        last_dt = datetime.fromtimestamp(active_timestamps[-1] / 1000, tz=UTC)

        return BacktestResult(
            strategy_name=self.strategy.name,
            symbols=universe,
            start_time=first_dt,
            end_time=last_dt,
            initial_capital=self.initial_cash,
            final_equity=self.equity_curve[-1].equity if self.equity_curve else self.initial_cash,
            metrics=overall_metrics,
            trades=self.trades,
            equity_curve=self.equity_curve,
            benchmark_return_pct=bm_ret,
            benchmark_max_drawdown_pct=bm_dd,
            rejections=self.rejections,
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            is_trades_count=len(is_trades),
            oos_trades_count=len(oos_trades),
            walk_forward_warning=walk_forward_warning,
        )


def generate_markdown_report(result: BacktestResult, output_path: Path | str | None = None) -> str:
    """Format backtest result into a comprehensive Markdown report file."""
    m = result.metrics
    ret_diff = m.total_return_pct - result.benchmark_return_pct

    lines = [
        f"# Backtest Performance Report: {result.strategy_name.upper()}",
        "",
        f"**Universe:** {', '.join(result.symbols)}  ",
        f"**Evaluation Period:** {result.start_time.strftime('%Y-%m-%d')} to {result.end_time.strftime('%Y-%m-%d')}  ",
        f"**Initial Capital:** ${result.initial_capital:,.2f}  ",
        f"**Final Equity:** ${result.final_equity:,.2f}  ",
        "",
    ]

    if result.walk_forward_warning:
        lines.extend([
            "> [!WARNING]",
            "> **WALK-FORWARD OVERFITTING ALERT**: This strategy was evaluated solely on in-sample data.",
            "> Deterministic strategies must be validated on out-of-sample data before passing the go-live gate.",
            "",
        ])

    lines.extend([
        "## 1. Executive Summary & Benchmark Comparison",
        "",
        "| Metric | Strategy Performance | Buy & Hold Benchmark | Relative Alpha |",
        "|---|---|---|---|",
        f"| **Total Return** | **{m.total_return_pct:+.2f}%** | {result.benchmark_return_pct:+.2f}% | {ret_diff:+.2f}% |",
        f"| **Max Drawdown** | **{m.max_drawdown_pct:.2f}%** | {result.benchmark_max_drawdown_pct:.2f}% | {result.benchmark_max_drawdown_pct - m.max_drawdown_pct:+.2f}% |",
        f"| **Sharpe Ratio** | **{m.sharpe_ratio:.2f}** | — | — |",
        f"| **Sortino Ratio** | **{m.sortino_ratio:.2f}** | — | — |",
        f"| **Total Closed Trades** | **{m.number_of_trades}** | — | — |",
        f"| **Win Rate** | **{m.win_rate_pct:.1f}%** | — | — |",
        f"| **Profit Factor** | **{m.profit_factor:.2f}** | — | — |",
        f"| **Total Fees Paid** | **${m.total_fees_paid:,.2f}** | — | — |",
        "",
    ])

    # Walk-forward section
    if result.is_metrics and result.oos_metrics:
        is_m = result.is_metrics
        oos_m = result.oos_metrics
        lines.extend([
            "## 2. Walk-Forward Analysis (Train / Test Split)",
            "",
            "| Metric | In-Sample (IS - 70%) | Out-of-Sample (OOS - 30%) | Robustness Ratio |",
            "|---|---|---|---|",
            f"| **Return %** | {is_m.total_return_pct:+.2f}% | {oos_m.total_return_pct:+.2f}% | {((oos_m.total_return_pct / is_m.total_return_pct) if is_m.total_return_pct != 0 else 0):.2f} |",
            f"| **Win Rate %** | {is_m.win_rate_pct:.1f}% | {oos_m.win_rate_pct:.1f}% | — |",
            f"| **Profit Factor** | {is_m.profit_factor:.2f} | {oos_m.profit_factor:.2f} | — |",
            f"| **Sharpe Ratio** | {is_m.sharpe_ratio:.2f} | {oos_m.sharpe_ratio:.2f} | — |",
            f"| **Max Drawdown %** | {is_m.max_drawdown_pct:.2f}% | {oos_m.max_drawdown_pct:.2f}% | — |",
            f"| **Trades Count** | {result.is_trades_count} | {result.oos_trades_count} | — |",
            "",
        ])

    lines.extend([
        "## 3. Trade Quality & Sizing Metrics",
        "",
        f"- **Average R-Multiple:** `{m.average_r:+.2f} R`",
        f"- **Expectancy (R):** `{m.expectancy_r:+.2f} R` per trade",
        f"- **Expectancy (USD):** `${m.expectancy_pnl:+,.2f}` per trade",
        f"- **Gross Profit:** `${m.gross_profit:,.2f}` | **Gross Loss:** `${m.gross_loss:,.2f}`",
        f"- **Exposure Time:** `{m.exposure_time_pct:.1f}%` of total backtest horizon",
        "",
        "## 4. Closed Trades Sample",
        "",
        "| Time | Symbol | Side | Entry | Exit | Qty | Realized PnL | R | Reason |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    if not result.trades:
        lines.append("| — | — | — | — | — | — | No trades executed | — | — |")
    else:
        for t in result.trades[-15:]:  # Show last 15 trades
            lines.append(
                f"| {t.exit_time.strftime('%Y-%m-%d %H:%M')} | {t.symbol} | {t.side.value} | "
                f"${t.entry_price:,.2f} | ${t.exit_price:,.2f} | {t.quantity:.4f} | "
                f"${t.realized_pnl:+,.2f} | {t.r_multiple:+.2f} R | `{t.exit_reason}` |"
            )

    report_text = "\n".join(lines) + "\n"

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(report_text, encoding="utf-8")
        logger.info("Backtest Markdown report written to %s", out_file)

    return report_text
