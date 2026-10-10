"""CLI Runner for deterministic strategy backtesting – Rule / Phase 7.

Usage:
    python -m app.backtest.run --strategy ema_trend --symbols BTC/USDT --days 365
    python -m app.backtest.run --strategy breakout_20 --symbols BTC/USDT,ETH/USDT --split 0.7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.backtest.engine import BacktestEngine, generate_markdown_report
from app.backtest.strategy import get_strategy
from app.config.loader import load_config
from app.data.store import CandleStore, get_default_store


def _ensure_sample_candles_if_empty(store: CandleStore, symbols: list[str]) -> None:
    """Ensure candle store has sufficient candle data for backtesting."""
    replay_dir = Path("data/sample_replay")

    # Check if store already has candles
    missing_symbols = []
    for sym in symbols:
        df = store.get_candles(sym, "4h")
        if len(df) < 60:
            missing_symbols.append(sym)

    if not missing_symbols:
        return

    # Check if sample_replay has json files
    if replay_dir.exists():
        for sym in missing_symbols:
            sanitized = sym.replace("/", "_")
            f_4h = replay_dir / f"{sanitized}_4h.json"
            f_1d = replay_dir / f"{sanitized}_1d.json"

            if f_4h.exists():
                try:
                    data = json.loads(f_4h.read_text(encoding="utf-8"))
                    store.upsert_candles(sym, "4h", data["candles"])
                except Exception:
                    pass

            if f_1d.exists():
                try:
                    data = json.loads(f_1d.read_text(encoding="utf-8"))
                    store.upsert_candles(sym, "1d", data["candles"])
                except Exception:
                    pass

    # If still missing, generate synthetic sample replay candles
    for sym in missing_symbols:
        df = store.get_candles(sym, "4h")
        if len(df) < 60:
            base_time = datetime(2025, 6, 1, 0, 0, tzinfo=UTC)
            base_price = 60_000.0 if "BTC" in sym else (3_000.0 if "ETH" in sym else 140.0)

            # Generate 1d candles
            candles_1d: list[list[float]] = []
            p = base_price
            for d in range(120):
                dt = base_time + timedelta(days=d)
                ts = int(dt.timestamp() * 1000)
                change = 0.003 * ((d % 7) - 2)
                p = round(p * (1.0 + change), 2)
                candles_1d.append([ts, p, round(p * 1.02, 2), round(p * 0.98, 2), p, 50_000_000.0])
            store.upsert_candles(sym, "1d", candles_1d)

            # Generate 4h candles
            candles_4h: list[list[float]] = []
            p4 = base_price
            start_4h = base_time + timedelta(days=30)
            for i in range(250):
                dt4 = start_4h + timedelta(hours=i * 4)
                ts4 = int(dt4.timestamp() * 1000)
                # Trend upwards with pullbacks
                delta = 0.004 if (i % 20 < 14) else -0.003
                p4 = round(p4 * (1.0 + delta), 2)
                candles_4h.append([ts4, p4, round(p4 * 1.015, 2), round(p4 * 0.985, 2), p4, 15_000_000.0])
            store.upsert_candles(sym, "4h", candles_4h)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deterministic strategy backtesting engine (Phase 7)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ema_trend",
        choices=["ema_trend", "breakout_20"],
        help="Strategy name (ema_trend or breakout_20)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTC/USDT",
        help="Comma-separated list of symbols (default: BTC/USDT)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of historical days to backtest (default: 365)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.7,
        help="Walk-forward in-sample fraction (default: 0.7)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        help="Initial backtest capital in quote currency (default: 10000.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the markdown report (optional)",
    )
    return parser.parse_args()


async def run_backtest_cli(args: argparse.Namespace) -> int:
    cfg = load_config()
    store = get_default_store()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = cfg.universe

    _ensure_sample_candles_if_empty(store, symbols)

    strategy = get_strategy(args.strategy)
    engine = BacktestEngine(
        strategy=strategy,
        config=cfg,
        initial_cash=args.capital,
        store=store,
    )

    print("\n" + "=" * 80)
    print("      AUTONOMOUS TRADING AGENT – DETERMINISTIC BACKTESTER (PHASE 7)")
    print("=" * 80)
    print(f" Strategy:       {args.strategy.upper()}")
    print(f" Universe:       {', '.join(symbols)}")
    print(f" Horizon:        Last {args.days} days")
    print(f" Initial Equity: ${args.capital:,.2f}")
    print(f" Train/Test:     {int(args.split * 100)}% In-Sample / {int((1.0 - args.split) * 100)}% Out-of-Sample")
    print("=" * 80 + "\n", flush=True)

    result = await engine.run(
        symbols=symbols,
        days=args.days,
        train_test_split=args.split,
    )

    # Format Markdown report
    report_path = args.output or f"backtest_report_{args.strategy}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.md"
    generate_markdown_report(result, output_path=report_path)

    # Print summary to stdout
    m = result.metrics
    alpha = m.total_return_pct - result.benchmark_return_pct
    print("--------------------------------------------------------------------------------")
    print("                     BACKTEST PERFORMANCE SUMMARY REPORT                        ")
    print("--------------------------------------------------------------------------------")
    print(f" Strategy Return:         {m.total_return_pct:>8.2f}%    | Final Equity: ${result.final_equity:,.2f}")
    print(f" Buy & Hold Benchmark:    {result.benchmark_return_pct:>8.2f}%    | Relative Alpha: {alpha:>+.2f}%")
    print(f" Strategy Max Drawdown:   {m.max_drawdown_pct:>8.2f}%    | Benchmark Max DD: {result.benchmark_max_drawdown_pct:.2f}%")
    print(f" Sharpe Ratio:            {m.sharpe_ratio:>8.2f}     | Sortino Ratio: {m.sortino_ratio:.2f}")
    print(f" Total Closed Trades:     {m.number_of_trades:>8d}     | Win Rate: {m.win_rate_pct:.1f}%")
    print(f" Profit Factor:           {m.profit_factor:>8.2f}     | Average R: {m.average_r:+.2f} R")
    print(f" Total Fees Paid:       ${m.total_fees_paid:>8.2f}     | Rejections: {len(result.rejections)}")
    print("--------------------------------------------------------------------------------")

    if result.is_metrics and result.oos_metrics:
        print(" WALK-FORWARD (TRAIN / TEST) BREAKDOWN:")
        print(f"  • In-Sample  (70%): Return={result.is_metrics.total_return_pct:+.2f}%, Sharpe={result.is_metrics.sharpe_ratio:.2f}, Trades={result.is_trades_count}")
        print(f"  • Out-Sample (30%): Return={result.oos_metrics.total_return_pct:+.2f}%, Sharpe={result.oos_metrics.sharpe_ratio:.2f}, Trades={result.oos_trades_count}")
        print("--------------------------------------------------------------------------------")

    print(f"\n[+] Full Markdown report saved to: {report_path}\n")
    return 0


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(run_backtest_cli(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
