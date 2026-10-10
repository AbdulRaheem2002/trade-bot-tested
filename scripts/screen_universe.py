"""CLI tool to autonomously screen top-opportunity Binance spot tokens – Option B.

Usage:
    # Preview top 10 candidates
    python scripts/screen_universe.py

    # Preview top 15 candidates with custom minimum volume
    python scripts/screen_universe.py --top 15 --min-volume 25000000

    # Screen and immediately apply to config.yaml
    python scripts/screen_universe.py --top 10 --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.data.exchange_client import ExchangeClient  # noqa: E402
from app.data.screener import UniverseScreener, update_config_universe  # noqa: E402


async def run_screener(top_n: int, min_volume: float, apply: bool) -> None:
    print("\n" + "=" * 75)
    print("      AUTONOMOUS UNIVERSE SCREENER (BINANCE SPOT MARKET SCAN)      ")
    print("=" * 75)
    print(f"[*] Minimum 24h Volume Gate: ${min_volume:,.0f}")
    print(f"[*] Target Candidates:        Top {top_n} tokens")
    print("[*] Fetching live exchange tickers from Binance...\n")

    client = ExchangeClient(exchange_id="binance")
    try:
        screener = UniverseScreener(min_quote_volume=min_volume)
        candidates = await screener.scan_market(client, top_n=top_n)

        if not candidates:
            print("[!] No tokens met the screening criteria. Try lowering --min-volume.")
            return

        header = f"{'Rank':<5} {'Symbol':<12} {'Price ($)':<12} {'24h Vol ($)':<16} {'24h Change':<12} {'Spread':<10} {'Score':<8}"
        print(header)
        print("-" * len(header))

        for idx, c in enumerate(candidates, 1):
            change_sign = "+" if c["change_24h_pct"] > 0 else ""
            change_str = f"{change_sign}{c['change_24h_pct']:.2f}%"
            vol_str = f"${c['quote_volume_24h']:,.0f}"
            price_str = f"${c['price']:,.4f}" if c["price"] < 1 else f"${c['price']:,.2f}"

            print(
                f"{idx:<5} {c['symbol']:<12} {price_str:<12} {vol_str:<16} {change_str:<12} {c['spread_bps']:>4.1f} bps  {c['score']:<8.1f}"
            )

        selected_symbols = [c["symbol"] for c in candidates]

        print("=" * 75)
        print(f"[+] Screened Universe: {', '.join(selected_symbols)}")

        if apply:
            cfg_path = _PROJECT_ROOT / "config.yaml"
            update_config_universe(selected_symbols, config_path=cfg_path)
            print(f"[+] Successfully applied top {len(selected_symbols)} tokens to {cfg_path.name}!")
        else:
            print("[i] To automatically save these tokens to config.yaml, re-run with '--apply'.")
        print("=" * 75 + "\n")

    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Crypto Universe Screener")
    parser.add_argument("--top", type=int, default=10, help="Number of top candidates to screen (default: 10)")
    parser.add_argument(
        "--min-volume",
        type=float,
        default=30_000_000.0,
        help="Minimum 24h quote volume in USDT (default: 30000000)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply screened tokens directly to config.yaml",
    )
    args = parser.parse_args()

    asyncio.run(run_screener(top_n=args.top, min_volume=args.min_volume, apply=args.apply))


if __name__ == "__main__":
    main()
