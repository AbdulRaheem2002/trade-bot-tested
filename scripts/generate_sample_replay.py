"""Generate realistic sample replay data for simulated multi-day runs – Rule / Phase 5."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path


def generate_replay_candles() -> None:
    target_dir = Path("data/sample_replay")
    target_dir.mkdir(parents=True, exist_ok=True)

    base_time = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)

    # 1. Generate 1d candles (45 days)
    for sym, base_price, vol_base in [
        ("BTC/USDT", 60_000.0, 100_000_000.0),
        ("ETH/USDT", 3_000.0, 80_000_000.0),
        ("SOL/USDT", 140.0, 60_000_000.0),
    ]:
        candles_1d: list[list[float]] = []
        curr_p = base_price
        for day in range(45):
            dt = base_time + timedelta(days=day)
            ts_ms = int(dt.timestamp() * 1000)
            o = curr_p
            # Slight upward trend
            change = (day * 0.003) + ((day % 3 - 1) * 0.008)
            c = round(o * (1.0 + change), 2)
            h = round(max(o, c) * 1.015, 2)
            low_val = round(min(o, c) * 0.985, 2)
            curr_p = c
            v = vol_base * (1.0 + (day % 5) * 0.1)
            candles_1d.append([ts_ms, o, h, low_val, c, v])

        fn_1d = target_dir / f"{sym.replace('/', '_')}_1d.json"
        fn_1d.write_text(
            json.dumps({"symbol": sym, "timeframe": "1d", "candles": candles_1d}, indent=2)
        )

    # 2. Generate 4h candles (90 candles = 15 days)
    # Starts at day 30 so 1d history is fully populated
    start_4h = base_time + timedelta(days=30)
    for sym, base_p, vol_base in [
        ("BTC/USDT", 65_000.0, 25_000_000.0),
        ("ETH/USDT", 3_200.0, 20_000_000.0),
        ("SOL/USDT", 150.0, 15_000_000.0),
    ]:
        candles_4h: list[list[float]] = []
        curr_p = base_p
        for i in range(90):
            dt = start_4h + timedelta(hours=i * 4)
            ts_ms = int(dt.timestamp() * 1000)
            o = curr_p
            # Periodic wave with trend
            if i < 40:
                step = 0.002
            elif i < 65:
                # Upward trend above EMA200
                step = 0.004
            elif i < 75:
                # Retracement (triggers invalidation for tested positions)
                step = -0.008
            else:
                step = 0.003

            c = round(o * (1.0 + step), 2)
            h = round(max(o, c) * 1.008, 2)
            low_val = round(min(o, c) * 0.992, 2)
            curr_p = c
            v = vol_base * (1.0 + (i % 6) * 0.1)
            candles_4h.append([ts_ms, o, h, low_val, c, v])

        fn_4h = target_dir / f"{sym.replace('/', '_')}_4h.json"
        fn_4h.write_text(
            json.dumps({"symbol": sym, "timeframe": "4h", "candles": candles_4h}, indent=2)
        )

    print(f"Sample replay candles generated successfully in {target_dir}")


if __name__ == "__main__":
    generate_replay_candles()
