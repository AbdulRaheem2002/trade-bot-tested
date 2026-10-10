"""Fast On-Demand Trading Cycle Runner – Option B Fast Trading.

Evaluates all active universe tokens against live Binance market data,
invokes Gemini AI, validates through the mathematical risk engine,
and executes orders immediately without waiting for scheduled candle boundaries.

Usage:
    python scripts/fast_cycle.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.brain.brain import run_brain  # noqa: E402
from app.common.llm_client import LLMClient  # noqa: E402
from app.common.models import BreakerState, CycleState, MarketSnapshot  # noqa: E402
from app.config.loader import load_config  # noqa: E402
from app.data.exchange_client import ExchangeClient  # noqa: E402
from app.data.snapshot import build_snapshot  # noqa: E402
from app.data.store import CandleStore  # noqa: E402
from app.execution.paper_broker import PaperBroker  # noqa: E402
from app.portfolio.service import PortfolioService  # noqa: E402
from app.reflex.reflex import run_reflex  # noqa: E402


async def run_fast_cycle() -> None:
    print("\n" + "=" * 75)
    print("      AUTONOMOUS TRADING AGENT - FAST ON-DEMAND TRADING CYCLE      ")
    print("=" * 75)

    config = load_config()
    db_path = _PROJECT_ROOT / "trading_agent.db"
    db_url = f"sqlite:///{db_path}"

    print(f"[*] Active Universe ({len(config.universe)} tokens): {', '.join(config.universe)}")
    print(f"[*] Max Open Positions: {config.max_open_positions} | Risk per Trade: {config.risk_per_trade_pct}%")
    print("[*] Connecting to Binance live exchange...\n")

    store = CandleStore(db_url)
    broker = PaperBroker(initial_cash=config.starting_equity)
    exchange_client = ExchangeClient(exchange_id=config.exchange_id)
    llm_client = LLMClient(timeout_s=config.decision_llm_timeout_s, max_retries=config.llm_max_retries)
    portfolio_service = PortfolioService(engine=db_url, broker=broker, config=config)
    await portfolio_service.start()

    try:
        now_dt = datetime.now(UTC)
        cycle_id = f"fast_{now_dt.strftime('%Y%m%d_%H%M%S')}"

        # 1. Sync live candles and current prices for all universe tokens
        print("[*] Step 1: Syncing live candles & order books from Binance...")
        for sym in config.universe:
            try:
                # Update both 4h and 1d candles
                await store.update_to_latest(sym, "4h", client=exchange_client)
                await store.update_to_latest(sym, "1d", client=exchange_client)
            except Exception as exc:
                print(f"    [!] Notice for {sym}: {exc}")

        # 2. Build live technical snapshots
        print("\n[*] Step 2: Building live indicator snapshots (EMA, ATR, RSI, Spreads)...")
        snapshots: dict[str, MarketSnapshot] = {}
        for sym in config.universe:
            try:
                snap = await build_snapshot(sym, store=store, client=exchange_client, config=config, now=now_dt)
                snapshots[sym] = snap
                print(
                    f"    [+] {sym:<10} Price: ${snap.current_price:,.2f} | EMA200: ${snap.ema200_4h:,.2f} | ATR14: ${snap.atr14_4h:,.2f} | Spread: {snap.spread_bps:.1f} bps"
                )
            except Exception as exc:
                print(f"    [-] Skipped {sym} (insufficient candles yet): {exc}")

        if not snapshots:
            print("[!] Could not build snapshots for any symbol. Exiting.")
            return

        # 3. Check / Run Brain daily macro report if needed
        print("\n[*] Step 3: Verifying Daily Brain Market Regime...")
        try:
            brain_report = await run_brain(
                client=llm_client,
                store=store,
                portfolio_service=portfolio_service,
                engine=db_url,
                config=config,
            )
            print(f"    [+] Brain Macro Regime: {brain_report.market_regime.value.upper()}")
        except Exception as exc:
            print(f"    [!] Brain report notice: {exc}")

        # 4. Run Reflex 4-hour decision cycle with AI
        print(f"\n[*] Step 4: Running Reflex Tactical AI Engine ({os.environ.get('LLM_MODEL_NAME', 'Gemini')})...")
        cycle_state = CycleState(
            cycle_time=now_dt,
            expected_candle_time=now_dt,
            breaker_state=BreakerState(),
            kill_switch_active=False,
        )

        approved_orders, rejections = await run_reflex(
            cycle_id=cycle_id,
            client=llm_client,
            portfolio_service=portfolio_service,
            store=store,
            exchange_client=exchange_client,
            config=config,
            engine=db_url,
            cycle_state=cycle_state,
        )

        print("\n" + "=" * 75)
        print("                         CYCLE EXECUTION RESULTS                         ")
        print("=" * 75)

        if rejections:
            print(f"[!] Risk Engine Filtered / Rejected ({len(rejections)}):")
            for r in rejections:
                print(f"    - {r.symbol}: {r.reason.value.upper()} ({r.detail})")

        if approved_orders:
            print(f"[+] Approved & Executed Orders ({len(approved_orders)}):")
            for order in approved_orders:
                print(
                    f"    - {order.action.value.upper()} {order.symbol} | Qty: {order.quantity:.4f} | Stop: ${order.protective_stop_price:,.2f}"
                )
                # Fill paper order
                fill = await broker.submit_order(order)
                if fill:
                    print(
                        f"      --> FILLED: {fill.quantity:.4f} @ ${fill.price:,.4f} (Fee: ${fill.fee:,.2f})"
                    )
                    if order.action.value.startswith("open"):
                        await portfolio_service.apply_open_fill(
                            fill,
                            invalidation_price=order.protective_stop_price,
                            thesis="Fast On-Demand Market Setup",
                            target_risk_amount=order.target_risk_amount,
                        )
                    elif order.action.value == "close":
                        await portfolio_service.apply_close_fill(fill, reason="signal_exit")
        else:
            print("[-] No orders approved this cycle (AI opted to HOLD or risk conditions not met).")

        # 5. Current Portfolio Status
        port = portfolio_service.get_state()
        print("\n" + "=" * 75)
        print(f"[*] Portfolio Status: Equity = ${port.equity:,.2f} | Cash = ${port.cash:,.2f} | Open Positions = {len(port.positions)}")
        for sym, pos in port.positions.items():
            print(f"    * {sym} ({pos.side.value.upper()}): {pos.quantity:.4f} units @ Entry: ${pos.entry_price:,.2f} | Stop: ${pos.invalidation_price:,.2f}")
        print("=" * 75 + "\n")

    finally:
        await portfolio_service.stop()
        await exchange_client.close()


def main() -> None:
    asyncio.run(run_fast_cycle())


if __name__ == "__main__":
    main()
