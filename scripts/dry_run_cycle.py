"""Dry run script: runs one full Reflex cycle on stored candles using MockLLM – Rule / Phase 4.

Usage:
    python scripts/dry_run_cycle.py
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

from app.common.enums import Action, ConvictionTier  # noqa: E402
from app.common.llm_client import MockLLM  # noqa: E402
from app.common.models import CycleState, Decision, DecisionItem, MarketSnapshot  # noqa: E402
from app.config.loader import load_config  # noqa: E402
from app.data.snapshot import build_snapshot  # noqa: E402
from app.data.store import CandleStore  # noqa: E402
from app.execution.paper_broker import PaperBroker  # noqa: E402
from app.portfolio.service import PortfolioService  # noqa: E402
from app.reflex.reflex import run_reflex  # noqa: E402


async def main() -> None:
    print("=" * 70)
    print("      AUTONOMOUS TRADING AGENT - REFLEX DRY RUN CYCLE")
    print("=" * 70)

    config = load_config()
    db_path = _PROJECT_ROOT / "trading_agent.db"
    db_url = f"sqlite:///{db_path}"

    store = CandleStore(db_url)
    broker = PaperBroker(initial_cash=config.starting_equity)
    now_dt = datetime.now(UTC)
    cycle_id = f"dry_run_{int(now_dt.timestamp())}"

    # 1. Build snapshots for universe
    snapshots: dict[str, MarketSnapshot] = {}
    for sym in config.universe:
        try:
            snap = await build_snapshot(sym, store=store, now=now_dt)
            snapshots[sym] = snap
            print(
                f"[Snapshot {sym}] Close: ${snap.last_close:,.2f} | EMA200: ${snap.ema200_4h:,.2f} | ATR14: ${snap.atr14_4h:,.2f}"
            )
        except Exception:
            # Fallback snapshot if candles not in DB
            snap = MarketSnapshot(
                symbol=sym,
                last_close=65_000.0 if "BTC" in sym else (3_200.0 if "ETH" in sym else 150.0),
                ema200_4h=60_000.0 if "BTC" in sym else (3_000.0 if "ETH" in sym else 140.0),
                atr14_4h=1_200.0 if "BTC" in sym else (80.0 if "ETH" in sym else 6.0),
                spread_bps=2.5,
                volume_24h=100_000_000.0,
                candle_timestamp=now_dt,
                current_price=65_000.0 if "BTC" in sym else (3_200.0 if "ETH" in sym else 150.0),
                lot_size=0.001,
                min_notional=10.0,
            )
            snapshots[sym] = snap
            print(
                f"[Fallback {sym}] Close: ${snap.last_close:,.2f} | EMA200: ${snap.ema200_4h:,.2f} | ATR14: ${snap.atr14_4h:,.2f}"
            )

    btc_snap = snapshots.get("BTC/USDT")
    btc_price = btc_snap.last_close if btc_snap else 65_000.0
    btc_atr = btc_snap.atr14_4h if btc_snap else 1_200.0

    # 2. Configure MockLLM with a representative Decision response:
    #    - 1 valid open_long on BTC/USDT (2 ATR stop distance)
    #    - 1 invalid decision on ETH/USDT (stop above entry for long -> triggers rejection demonstration)
    #    - 1 hold decision on SOL/USDT
    scripted_decision = Decision(
        reasoning="BTC exhibits strong momentum above 4h EMA200; ETH setup has invalid geometry; SOL on watch.",
        decisions=[
            DecisionItem(
                action=Action.OPEN_LONG,
                symbol="BTC/USDT",
                invalidation_price=btc_price - (2.0 * btc_atr),
                conviction=ConvictionTier.A,
                confidence=0.88,
                thesis="Clean breakout from daily consolidation range",
                rationale="Above 4h EMA200 with expanding volume",
            ),
            DecisionItem(
                action=Action.OPEN_LONG,
                symbol="ETH/USDT",
                invalidation_price=4_000.0,  # Invalid: above current price for long!
                conviction=ConvictionTier.B,
                confidence=0.60,
                thesis="Trend following",
                rationale="Testing invalidation check",
            ),
            DecisionItem(
                action=Action.HOLD,
                symbol="SOL/USDT",
            ),
        ],
    )

    mock_llm = MockLLM(canned_responses=[scripted_decision])

    # 3. Execute Reflex cycle
    async with PortfolioService(db_url, broker, config=config) as p_service:
        cycle_state = CycleState(cycle_time=now_dt, expected_candle_time=now_dt)

        print(f"\nExecuting Reflex Cycle: {cycle_id}")
        approved_orders, rejections = await run_reflex(
            cycle_id=cycle_id,
            client=mock_llm,
            portfolio_service=p_service,
            config=config,
            engine=p_service.engine,
            cycle_state=cycle_state,
            snapshots=snapshots,
        )

        # 4. Print Decision proposals, Rejections, and Sized Orders
        print("\n" + "=" * 70)
        print("                        LLM DECISION PROPOSALS")
        print("=" * 70)
        print(f"Reasoning: {scripted_decision.reasoning}")
        for idx, d in enumerate(scripted_decision.decisions):
            print(
                f"  [{idx + 1}] {d.action.value.upper():<10} {d.symbol:<10} Stop: {d.invalidation_price} | Tier: {d.conviction} | Conf: {d.confidence}"
            )

        print("\n" + "=" * 70)
        print("                        REJECTIONS AUDIT LOG")
        print("=" * 70)
        if rejections:
            for r in rejections:
                print(
                    f"  - Symbol: {r.symbol:<10} Action: {r.action.value:<10} Reason: {r.reason.value}"
                )
                print(f"    Detail: {r.detail}")
        else:
            print("  No rejections.")

        print("\n" + "=" * 70)
        print("                        APPROVED & SIZED ORDERS")
        print("=" * 70)
        if approved_orders:
            for order in approved_orders:
                print(f"  - Symbol:       {order.symbol}")
                print(
                    f"    Action:       {order.action.value.upper()} ({order.side.value.upper()})"
                )
                print(f"    Quantity:     {order.quantity:.4f}")
                print(f"    Entry Price:  ${order.entry_price:,.2f}")
                print(f"    Invalidation: ${order.invalidation_price:,.2f}")
                print(f"    Notional:     ${order.notional:,.2f}")
                print(
                    f"    Risk Budget:  ${order.risk_amount:,.2f} (Tier {order.conviction.value})"
                )
                print(f"    Thesis:       {order.thesis}")
        else:
            print("  No approved orders.")
        print("=" * 70)
        print("\nReflex dry run cycle completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
