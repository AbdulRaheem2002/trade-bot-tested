"""Demo script: simulates paper trading on recorded candles – Rule / Phase 3.

Demonstrates:
1. Replaying recorded 4h candles from the local candle store (or built-in sequence)
2. Placing market orders via PaperBroker with slippage and taker fees
3. Synchronizing fills and positions through PortfolioService's single-writer queue
4. Persisting orders, fills, trades, and equity snapshots to SQLite
5. Computing and displaying performance metrics
6. Exporting and printing the CSV trade journal
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.common.enums import ConvictionTier, Side  # noqa: E402
from app.data.store import CandleStore  # noqa: E402
from app.execution.paper_broker import PaperBroker  # noqa: E402
from app.portfolio.db import TradeModel  # noqa: E402
from app.portfolio.metrics import compute_metrics_from_db, export_trade_journal_csv  # noqa: E402
from app.portfolio.service import PortfolioService  # noqa: E402


async def run_demo() -> None:
    print("=" * 70)
    print("      AUTONOMOUS TRADING AGENT - PHASE 3 PAPER TRADING DEMO")
    print("=" * 70)

    db_path = _PROJECT_ROOT / "trading_agent.db"
    db_url = f"sqlite:///{db_path}"

    # 1. Load recorded candles from SQLite store
    store = CandleStore(db_url)
    symbol = "BTC/USDT"
    candles_df = store.get_candles(symbol, "4h")

    if len(candles_df) >= 10:
        print(f"Loaded {len(candles_df)} recorded {symbol} 4h candles from {db_path.name}")
        recent_candles = candles_df.tail(10).to_dict("records")
    else:
        print("Using built-in recorded candle sequence...")
        base_time = int(datetime(2026, 6, 1, 0, 0, tzinfo=UTC).timestamp() * 1000)
        recent_candles = [
            {"open_time": base_time + (i * 14_400_000), "close": p}
            for i, p in enumerate(
                [65000.0, 65400.0, 66100.0, 65800.0, 66500.0, 67200.0, 66900.0, 67800.0]
            )
        ]

    # 2. Initialize Paper Broker and Portfolio Service
    initial_cash = 10_000.0
    broker = PaperBroker(
        initial_cash=initial_cash,
        slippage_bps=5.0,  # 0.05% slippage against trader
        fees_taker_bps=10.0,  # 0.10% taker fee
    )

    async with PortfolioService(db_url, broker, initial_cash=initial_cash) as service:
        print(
            f"\nInitial State: Cash = ${service.cash:,.2f} | Peak Equity = ${service.peak_equity:,.2f}"
        )

        # ── Trade 1: Open LONG at candle 0 ───────────────────────────────────
        c0 = recent_candles[0]
        price_0 = float(c0["close"])
        broker.set_market_price(symbol, price_0)

        dt_0 = datetime.fromtimestamp(c0["open_time"] / 1000, tz=UTC)
        print(f"\n[Candle 0 | {dt_0.strftime('%Y-%m-%d %H:%M UTC')}] BTC Close: ${price_0:,.2f}")
        print("--> Decision: OPEN_LONG (Conviction A, 0.05 BTC)")

        # Record order
        client_id_1 = f"demo_order_1_{int(dt_0.timestamp())}"
        await service.record_order(
            client_order_id=client_id_1,
            symbol=symbol,
            side="long",
            action="open_long",
            quantity=0.05,
            price=price_0,
        )

        # Place market order on broker
        fill_1 = await broker.place_market_order(
            symbol=symbol,
            side=Side.LONG,
            qty=0.05,
            client_order_id=client_id_1,
            invalidation_price=price_0 * 0.96,
            conviction=ConvictionTier.A,
            risk_amount=price_0 * 0.04 * 0.05,
            timestamp=dt_0,
        )
        await service.apply_open_fill(
            fill=fill_1,
            invalidation_price=price_0 * 0.96,
            conviction=ConvictionTier.A,
            risk_amount=price_0 * 0.04 * 0.05,
        )
        await broker.set_protective_stop(symbol, price_0 * 0.96)
        print(f"    Filled: {fill_1.quantity} BTC @ ${fill_1.price:,.2f} (Fee: ${fill_1.fee:.2f})")
        print(f"    Cash remaining: ${service.cash:,.2f}")

        # Snapshot at candle 0
        await service.record_equity_snapshot(prices={symbol: price_0}, timestamp=dt_0)

        # ── Intermediate candles ─────────────────────────────────────────────
        for idx in range(1, 4):
            c = recent_candles[idx]
            p = float(c["close"])
            dt = datetime.fromtimestamp(c["open_time"] / 1000, tz=UTC)
            broker.set_market_price(symbol, p)
            unrealized_r = service.compute_unrealized_r(symbol, p)
            eq = service.compute_equity({symbol: p})
            await service.record_equity_snapshot(prices={symbol: p}, timestamp=dt)
            print(
                f"[Candle {idx} | {dt.strftime('%Y-%m-%d %H:%M UTC')}] Price: ${p:,.2f} | Equity: ${eq:,.2f} | Unrealized R: {unrealized_r:+.2f}R"
            )

        # ── Close Trade 1 at candle 4 ────────────────────────────────────────
        c4 = recent_candles[4]
        price_4 = float(c4["close"])
        dt_4 = datetime.fromtimestamp(c4["open_time"] / 1000, tz=UTC)
        broker.set_market_price(symbol, price_4)
        print(f"\n[Candle 4 | {dt_4.strftime('%Y-%m-%d %H:%M UTC')}] BTC Close: ${price_4:,.2f}")
        print("--> Decision: CLOSE (Take profit / Signal exit)")

        close_fill_1 = await broker.close_position(symbol, reason="signal_exit", timestamp=dt_4)
        if close_fill_1:
            closed_trade_1 = await service.apply_close_fill(close_fill_1, reason="signal_exit")
            if closed_trade_1:
                print(f"    Exit Filled: {close_fill_1.quantity} BTC @ ${close_fill_1.price:,.2f}")
                print(
                    f"    Realized PnL: ${closed_trade_1.realized_pnl:+.2f} | R-Multiple: {closed_trade_1.r_multiple:+.2f}R"
                )
        await service.record_equity_snapshot(prices={symbol: price_4}, timestamp=dt_4)

        # ── Reconcile state with broker ──────────────────────────────────────
        report = await service.reconcile(strict=True)
        print(
            f"\nReconciliation check: {'PASSED (Zero Discrepancies)' if report.is_matched else 'FAILED'}"
        )

        # ── Compute and Print Metrics ────────────────────────────────────────
        metrics = compute_metrics_from_db(service.engine)

        print("\n" + "=" * 70)
        print("                     PERFORMANCE METRICS SUMMARY")
        print("=" * 70)
        print(
            f"  Total Trades:       {metrics.number_of_trades:<10} | Win Rate:          {metrics.win_rate_pct:.1f}%"
        )
        print(
            f"  Gross Profit:       ${metrics.gross_profit:<9.2f} | Gross Loss:        ${metrics.gross_loss:.2f}"
        )
        print(
            f"  Profit Factor:      {metrics.profit_factor:<10.2f} | Average R:         {metrics.average_r:+.2f}R"
        )
        print(
            f"  Expectancy:         {metrics.expectancy_r:<+9.2f}R | Expectancy ($):    ${metrics.expectancy_pnl:+.2f}"
        )
        print(
            f"  Total Return:       {metrics.total_return_pct:<+9.2f}% | Max Drawdown:      {metrics.max_drawdown_pct:.2f}%"
        )
        print(
            f"  Sharpe Ratio:       {metrics.sharpe_ratio:<10.2f} | Sortino Ratio:     {metrics.sortino_ratio:.2f}"
        )
        print(
            f"  Total Fees Paid:    ${metrics.total_fees_paid:<9.2f} | Exposure Time:     {metrics.exposure_time_pct:.1f}%"
        )
        print("=" * 70)

        # ── Trade Journal CSV Export ─────────────────────────────────────────
        journal_dir = _PROJECT_ROOT / "logs"
        journal_path = journal_dir / "trade_journal.csv"
        with Session(service.engine) as sess:
            db_trades = sess.scalars(select(TradeModel)).all()
            csv_content = export_trade_journal_csv(db_trades, output_path=journal_path)

        print(f"\nTrade Journal Exported to: {journal_path}")
        print("\n--- Trade Journal CSV Content ---")
        print(csv_content.strip())
        print("---------------------------------")
        print("\nPhase 3 Paper Trading Demo finished successfully.")


if __name__ == "__main__":
    asyncio.run(run_demo())
