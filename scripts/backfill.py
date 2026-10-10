"""CLI script to backfill OHLCV candles for the universe – Rule / Phase 2.

Usage:
    python scripts/backfill.py [--symbols BTC/USDT,ETH/USDT] [--timeframes 4h,1d] [--days 365]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path so scripts can run standalone
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config.loader import load_config  # noqa: E402
from app.data.exchange_client import ExchangeClient  # noqa: E402
from app.data.store import CandleStore  # noqa: E402


async def run_backfill(
    symbols: list[str],
    timeframes: list[str],
    days: int,
    exchange_id: str,
    db_url: str,
) -> None:
    print(f"Connecting to exchange: {exchange_id}...")
    store = CandleStore(db_url=db_url)
    client = ExchangeClient(exchange_id=exchange_id)

    try:
        for symbol in symbols:
            print("\n==================================================")
            print(f"Symbol: {symbol}")
            print("==================================================")
            for tf in timeframes:
                print(f"--> Backfilling {tf} candles for {days} days...")
                upserted = await store.backfill(
                    symbol=symbol,
                    timeframe=tf,
                    days=days,
                    client=client,
                )
                total_df = store.get_candles(symbol, tf)
                total_candles = len(total_df)
                gaps = store.detect_gaps(symbol, tf)

                print(f"    - Saved this run: {upserted} candles")
                print(f"    - Total closed in DB: {total_candles} candles")
                print(f"    - Gaps detected: {len(gaps)}")
                if gaps:
                    print(f"    - Gap details: {gaps[:5]} (showing first {min(len(gaps), 5)})")
    finally:
        await client.close()
        print("\nBackfill complete. Exchange connection closed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill OHLCV data into SQLite candle store.")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols (defaults to config.yaml universe, e.g. BTC/USDT,ETH/USDT,SOL/USDT)",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="4h,1d",
        help="Comma-separated timeframes (default: 4h,1d)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days to backfill (default: 365)",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default=None,
        help="Exchange ID (defaults to config.yaml exchange_id)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="sqlite:///trading_agent.db",
        help="Database URL (default: sqlite:///trading_agent.db)",
    )

    args = parser.parse_args()

    cfg = load_config()
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else cfg.universe
    timeframes = [tf.strip() for tf in args.timeframes.split(",")]
    exchange_id = args.exchange or cfg.exchange_id

    asyncio.run(
        run_backfill(
            symbols=symbols,
            timeframes=timeframes,
            days=args.days,
            exchange_id=exchange_id,
            db_url=args.db,
        )
    )


if __name__ == "__main__":
    main()
