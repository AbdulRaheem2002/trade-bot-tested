"""Main entrypoint for autonomous trading agent CLI – Phase 5 & Phase 9.

Usage:
    python -m app.main --mode paper
    python -m app.main --mode paper --replay data/sample_replay
    python -m app.main --mode live (requires all 6 Go-Live gating checks)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any

from app.common.enums import TradingMode
from app.common.logging import get_logger, setup_logging
from app.config.loader import load_config
from app.orchestrator import Orchestrator

logger = get_logger("app.main")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Autonomous Medium-Term Crypto Portfolio Trading Agent",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (paper or live). Note: live requires strict 6-stage gating verification.",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Path to directory containing historical candles for multi-day replay simulation.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml file.",
    )
    return parser.parse_args(args)


async def main_async(args: argparse.Namespace) -> int:
    """Async main routine."""
    # Load configuration
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)

    # Override mode in config
    config.mode = TradingMode(args.mode)

    broker = None
    exchange_client = None

    if config.mode == TradingMode.LIVE:
        from app.data.exchange_client import ExchangeClient
        from app.execution.live_broker import LiveBroker
        from app.execution.live_gate import LiveGatingError, verify_live_trading_gates

        api_key = os.environ.get("EXCHANGE_API_KEY", "")
        secret = os.environ.get("EXCHANGE_SECRET", "")

        exchange_client = ExchangeClient(
            exchange_id=config.exchange_id,
            options={"apiKey": api_key, "secret": secret},
        )

        try:
            await verify_live_trading_gates(
                mode=config.mode,
                config=config,
                api_key=api_key,
                secret=secret,
                exchange=exchange_client._exchange,
            )
        except LiveGatingError as exc:
            logger.error("Live trading gate check failed: %s", exc)
            print(f"Error: {exc}", file=sys.stderr)
            await exchange_client.close()
            return 1

        broker = LiveBroker(
            exchange_client=exchange_client,
            config=config,
            max_capital=config.live.max_capital,
        )

    llm_client = None
    if args.replay and not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        from app.common.enums import Action, ConvictionTier
        from app.common.llm_client import MockLLM
        from app.common.models import Decision, DecisionItem

        btc_open = Decision(
            reasoning="BTC/USDT breaking out above 4h EMA200 with solid volume.",
            decisions=[
                DecisionItem(
                    action=Action.OPEN_LONG,
                    symbol="BTC/USDT",
                    invalidation_price=63_000.0,
                    conviction=ConvictionTier.A,
                    confidence=0.88,
                    thesis="Clean breakout from daily consolidation range",
                    rationale="Trend continuation above 4h EMA200",
                )
            ],
        )
        eth_open = Decision(
            reasoning="ETH/USDT showing relative strength.",
            decisions=[
                DecisionItem(
                    action=Action.OPEN_LONG,
                    symbol="ETH/USDT",
                    invalidation_price=3_040.0,
                    conviction=ConvictionTier.B,
                    confidence=0.75,
                    thesis="Support hold above 4h EMA200",
                    rationale="Bounce off key moving average",
                )
            ],
        )
        canned = [btc_open] + [Decision(reasoning="Holding positions", decisions=[])] * 15 + [eth_open]
        llm_client = MockLLM(canned_responses=canned)

    orchestrator = Orchestrator(
        config=config,
        broker=broker,
        exchange_client=exchange_client,
        llm_client=llm_client,
    )

    # Replay mode execution
    if args.replay:
        replay_path = Path(args.replay)
        if not replay_path.exists():
            print(f"Error: Replay directory not found: {replay_path}", file=sys.stderr)
            return 1

        logger.info("Executing simulated multi-day replay from %s...", replay_path)
        try:
            await orchestrator.portfolio_service.start()
            await orchestrator.run_replay(replay_path)
            await orchestrator.portfolio_service.stop()
            return 0
        except Exception as exc:
            logger.exception("Replay failed with error: %s", exc)
            return 1

    # Normal real-time paper operation with signal handling
    stop_event = asyncio.Event()

    def _on_shutdown_signal(*_: Any) -> None:
        logger.info("Received termination signal. Triggering graceful shutdown...")
        stop_event.set()

    # Register OS signal handlers
    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _on_shutdown_signal)
            except NotImplementedError:
                signal.signal(sig, _on_shutdown_signal)
    else:
        # Windows ProactorEventLoop signal handling
        signal.signal(signal.SIGINT, _on_shutdown_signal)
        signal.signal(signal.SIGTERM, _on_shutdown_signal)

    try:
        await orchestrator.start()
        logger.info("Trading agent is running. Press Ctrl+C to terminate.")
        await stop_event.wait()
    except Exception as exc:
        logger.exception("Fatal error in orchestrator: %s", exc)
        return 1
    finally:
        await orchestrator.stop()

    return 0


def main() -> None:
    """Synchronous entry point."""
    setup_logging()
    args = parse_args()
    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
