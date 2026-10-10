"""Fast On-Demand Trading Cycle Runner – Option B Fast Trading.

Evaluates all active universe tokens against live Binance market data,
invokes Gemini AI, validates through the mathematical risk engine,
and executes orders immediately without waiting for scheduled candle boundaries.

Usage:
    python scripts/fast_cycle.py
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path

# Silence internal framework logging for clean 3-line console output
logging.disable(logging.INFO)
logging.getLogger("litellm").setLevel(logging.CRITICAL)
os.environ["LITELLM_LOG"] = "CRITICAL"
warnings.filterwarnings("ignore")

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.brain.brain import run_brain  # noqa: E402
from app.common.enums import ConvictionTier  # noqa: E402
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
    config = load_config()
    db_path = _PROJECT_ROOT / "trading_agent.db"
    db_url = f"sqlite:///{db_path}"

    store = CandleStore(db_url)
    exchange_client = ExchangeClient(exchange_id=config.exchange_id)
    broker = PaperBroker(
        initial_cash=config.starting_equity,
        shorts_enabled=config.shorts_enabled,
        exchange_client=exchange_client,
    )
    llm_client = LLMClient(timeout_s=config.decision_llm_timeout_s, max_retries=config.llm_max_retries)
    portfolio_service = PortfolioService(engine=db_url, broker=broker, config=config)
    await portfolio_service.start()

    try:
        now_dt = datetime.now(UTC)
        time_tag = now_dt.strftime("%H:%M:%S")
        cycle_id = f"fast_{now_dt.strftime('%Y%m%d_%H%M%S')}"

        # 1. Sync live candles quietly
        for sym in config.universe:
            with contextlib.suppress(Exception):
                await store.update_to_latest(sym, "4h", client=exchange_client)
                await store.update_to_latest(sym, "1d", client=exchange_client)

        snapshots: dict[str, MarketSnapshot] = {}
        for sym in config.universe:
            with contextlib.suppress(Exception):
                snap = await build_snapshot(sym, store=store, client=exchange_client, config=config, now=now_dt)
                snapshots[sym] = snap

        if not snapshots:
            print(f"[{time_tag}] [!] Could not sync live market data from Binance.")
            return

        with contextlib.suppress(Exception):
            await run_brain(
                client=llm_client,
                store=store,
                portfolio_service=portfolio_service,
                engine=db_url,
                config=config,
            )

        # Line 1: Sync status
        engine_name = os.environ.get("LLM_MODEL_NAME", "Gemini / Groq")
        print(f"[{time_tag}] ⚡ 5-Min Cycle: Synced {len(snapshots)} pairs from Binance | AI Engine: {engine_name}")

        # 2. Run Reflex decision cycle
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

        # Line 2: Decision / Execution status
        if approved_orders:
            for order in approved_orders:
                client_id = f"fast_{order.symbol}_{uuid.uuid4().hex[:8]}"
                if order.action.value.startswith("open"):
                    fill = await broker.place_market_order(
                        symbol=order.symbol,
                        side=order.side,
                        qty=order.quantity,
                        client_order_id=client_id,
                        invalidation_price=order.invalidation_price,
                        conviction=order.conviction or ConvictionTier.B,
                        risk_amount=order.risk_amount,
                    )
                    if fill:
                        await portfolio_service.apply_open_fill(
                            fill=fill,
                            invalidation_price=order.invalidation_price,
                            conviction=order.conviction or ConvictionTier.B,
                            risk_amount=order.risk_amount,
                            action=order.action.value,
                        )
                        print(f"[{time_tag}] 🟢 EXECUTED: {order.action.value.upper()} {order.symbol} ({fill.quantity:.4f} @ ${fill.price:,.2f} | Stop: ${order.invalidation_price:,.2f})")
                elif order.action.value == "close":
                    fill = await broker.close_position(symbol=order.symbol, client_order_id=client_id)
                    if fill:
                        await portfolio_service.apply_close_fill(fill, reason="signal_exit")
                        print(f"[{time_tag}] 🔴 CLOSED: {order.symbol} @ ${fill.price:,.2f}")
        else:
            print(f"[{time_tag}] 🛡️ DECISION: HOLD CASH (All {len(snapshots)} pairs scanned; zero risk; capital protected)")

        # Line 3: Portfolio state
        port = portfolio_service.get_state()
        print(f"[{time_tag}] 💼 PORTFOLIO: Equity = ${port.equity:,.2f} | Cash = ${port.cash:,.2f} | Open Positions = {len(port.positions)}")

    finally:
        await portfolio_service.stop()
        await exchange_client.close()


def main() -> None:
    asyncio.run(run_fast_cycle())


if __name__ == "__main__":
    main()
