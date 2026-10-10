"""CLI Tool: Analyze Any Coin & Generate Live Trade Signal (LONG / SHORT).

Usage:
    python scripts/analyze_coin.py DOGE
    python scripts/analyze_coin.py SOL
    python scripts/analyze_coin.py BTC
    python scripts/analyze_coin.py PEPE
    python scripts/analyze_coin.py ETH/USDT
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

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

from app.brain.analyzer import CoinAnalyzer  # noqa: E402


def format_num(val: float) -> str:
    """Format price nicely based on magnitude."""
    if val >= 1000:
        return f"${val:,.2f}"
    if val >= 1:
        return f"${val:,.4f}"
    return f"${val:,.6f}"


async def run_analysis(symbol_input: str, engine: str = "auto") -> None:
    print("\n" + "=" * 76)
    print("        AUTONOMOUS AI COIN ANALYZER & LIVE SIGNAL GENERATOR        ")
    print("=" * 76)
    print(f"[*] Target Asset:   {symbol_input.upper()}")
    print("[*] Exchange:       Binance Spot (Real-Time)")
    print(f"[*] AI Engine:      {engine.upper()} (Auto-failover enabled)")
    print("[*] Fetching live candlesticks, order book & computing indicators...")

    analyzer = CoinAnalyzer(model=engine)
    try:
        sig = await analyzer.analyze_symbol(symbol_input)

        # Signal Header Badge
        badge = sig.signal.upper()
        if "LONG" in badge:
            badge_str = f"🟢 [ SIGNAL: STRONG LONG ] ({sig.confidence_pct}% Confidence | {sig.conviction})"
        elif "SHORT" in badge:
            badge_str = f"🔴 [ SIGNAL: STRONG SHORT ] ({sig.confidence_pct}% Confidence | {sig.conviction})"
        else:
            badge_str = f"⚪ [ SIGNAL: WAIT / NEUTRAL ] ({sig.confidence_pct}% Confidence | {sig.conviction})"

        print("\n" + "=" * 76)
        print(f"  {sig.symbol} @ {format_num(sig.current_price)}")
        print(f"  {badge_str}")
        print(f"  🏆 Selected Strategy: {sig.selected_strategy}")
        print(f"  📜 Attribution:       {sig.strategy_attribution}")
        print(f"  ⚡ Model:              {sig.model_used} | Latency: {sig.latency_s}s")
        print("=" * 76)

        if sig.strategy_scores:
            print("\n[+] 4-STRATEGY EVALUATION SCORECARD:")
            for s_name, s_score in sig.strategy_scores.items():
                marker = "⭐ [WINNER]" if s_name in sig.selected_strategy or sig.selected_strategy in s_name else "  "
                print(f"  {marker} • {s_name:<34} : {s_score}")

        if sig.winning_edge:
            print("\n[+] WINNING STRATEGY EDGE & SELECTION RATIONALE:")
            print(f"  \"{sig.winning_edge}\"")

        print("\n[+] TRADE SETUP & RISK PARAMETERS:")
        print(f"  • Recommended Entry:     {sig.entry_zone}")
        print(f"  • Stop-Loss (Invalid):   {format_num(sig.stop_loss)}")
        print(f"  • Take-Profit Target 1:  {format_num(sig.take_profit_1)}")
        print(f"  • Take-Profit Target 2:  {format_num(sig.take_profit_2)}")
        print(f"  • Risk / Reward Ratio:   {sig.risk_reward_ratio}")

        print("\n[+] TECHNICAL STRUCTURE & METRICS:")
        ind = sig.indicators
        print(f"  • Macro Trend:           {sig.macro_trend}")
        print(f"  • RSI (14, 4h):          {ind.get('rsi14_4h')} ({sig.rsi_status})")
        print(f"  • 4h EMA 20 / 50 / 200:  {format_num(ind.get('ema20_4h', 0))} | {format_num(ind.get('ema50_4h', 0))} | {format_num(ind.get('ema200_4h', 0))}")
        print(f"  • Volatility ATR (14):   {format_num(ind.get('atr14_4h', 0))}")
        print(f"  • 4h Volume Ratio:       {ind.get('volume_ratio', 1.0)}x 20-period SMA")
        print(f"  • 24h Volume / Spread:   ${ind.get('volume_24h_usd', 0):,.0f} USDT | {ind.get('spread_bps', 0)} bps")

        if sig.key_support or sig.key_resistance:
            supp_str = ", ".join(format_num(s) for s in sig.key_support) or "N/A"
            res_str = ", ".join(format_num(r) for r in sig.key_resistance) or "N/A"
            print(f"  • Key Support Levels:    {supp_str}")
            print(f"  • Key Resistance Levels: {res_str}")

        print("\n[+] AI ANALYSIS THESIS:")
        print(f"  \"{sig.thesis}\"")

        print("\n[!] INVALIDATION CONDITION:")
        print(f"  \"{sig.invalidation_condition}\"")

        print("=" * 76 + "\n")

    except Exception as exc:
        print(f"\n[!] Error analyzing {symbol_input}: {exc}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live AI Coin Analysis & Trade Signal Generator")
    parser.add_argument("symbol", nargs="?", default="DOGE", help="Coin symbol (e.g., DOGE, SOL, BTC, PEPE)")
    parser.add_argument(
        "--engine",
        "-e",
        default="auto",
        choices=["auto", "groq", "gemini"],
        help="LLM inference engine: auto (failover), groq (sub-second), or gemini",
    )
    args = parser.parse_args()

    asyncio.run(run_analysis(args.symbol, engine=args.engine))


if __name__ == "__main__":
    main()
