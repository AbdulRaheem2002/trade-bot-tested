"""Market snapshot builder with multi-timeframe alignment – Rule / Phase 2.

Features:
- Constructs a validated `MarketSnapshot` for a given symbol
- Includes last N closed 4h candles (default 60) and 1d candles (default 30) in compact format
- Strict data freshness check against expected closed candle time (raises StaleDataError if older)
- Zero look-ahead guarantee for multi-timeframe alignment: 1d candles are only considered
  after they have fully closed (e.g., at 00:00 UTC the following day)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.common.errors import StaleDataError
from app.common.logging import get_logger
from app.common.models import MarketSnapshot
from app.common.time_utils import get_latest_closed_candle_time
from app.data.indicators import add_1d_indicators, add_4h_indicators
from app.data.store import CandleStore, get_default_store

if TYPE_CHECKING:
    from app.config.settings import TradingConfig
    from app.data.exchange_client import ExchangeClient

logger = get_logger(__name__)


def _df_to_compact_candles(df: Any, n: int) -> list[dict[str, Any]]:
    """Convert last *n* rows of OHLCV dataframe into compact dicts."""
    if df.empty:
        return []
    tail = df.tail(n)
    return [
        {
            "t": int(row["open_time"]),
            "o": float(row["open"]),
            "h": float(row["high"]),
            "l": float(row["low"]),
            "c": float(row["close"]),
            "v": float(row["volume"]),
        }
        for _, row in tail.iterrows()
    ]


async def build_snapshot(
    symbol: str,
    store: CandleStore | None = None,
    client: ExchangeClient | None = None,
    config: TradingConfig | None = None,
    now: datetime | None = None,
    n_4h: int = 60,
    n_1d: int = 30,
) -> MarketSnapshot:
    """Build a validated MarketSnapshot for *symbol* with zero look-ahead."""
    store = store or get_default_store()
    now_dt = now or datetime.now(UTC)
    now_dt = now_dt.replace(tzinfo=UTC) if now_dt.tzinfo is None else now_dt.astimezone(UTC)

    now_ms = int(now_dt.timestamp() * 1000)

    # ── 1. Fetch 4h closed candles ───────────────────────────────────
    df_4h = store.get_candles(symbol, "4h", now_ms=now_ms)
    if df_4h.empty:
        raise StaleDataError(f"No closed 4h candles found in store for {symbol}")

    latest_4h = df_4h.iloc[-1]
    latest_open_ms = int(latest_4h["open_time"])
    latest_dt = datetime.fromtimestamp(latest_open_ms / 1000, tz=UTC)

    # ── 2. Data freshness check ──────────────────────────────────────
    expected_closed_time = get_latest_closed_candle_time(now_dt, "4h")
    expected_closed_ms = int(expected_closed_time.timestamp() * 1000)

    if latest_open_ms < expected_closed_ms:
        logger.warning(
            "Snapshot for %s is stale: latest closed is %s, expected %s",
            symbol,
            latest_dt.isoformat(),
            expected_closed_time.isoformat(),
        )
        raise StaleDataError(
            f"Market snapshot for {symbol} is stale: latest closed 4h candle is {latest_dt.isoformat()}, "
            f"expected {expected_closed_time.isoformat()}."
        )

    # ── 3. Compute 4h indicators ─────────────────────────────────────
    df_4h_ind = add_4h_indicators(df_4h)
    last_row_4h = df_4h_ind.iloc[-1]

    ema200_4h = float(last_row_4h["ema200_4h"])
    atr14_4h = float(last_row_4h["atr14_4h"]) if not pd_isna(last_row_4h["atr14_4h"]) else 0.0
    rsi14_4h = float(last_row_4h["rsi14_4h"]) if not pd_isna(last_row_4h["rsi14_4h"]) else None
    vol_sma20_4h = (
        float(last_row_4h["volume_sma20_4h"])
        if not pd_isna(last_row_4h["volume_sma20_4h"])
        else None
    )
    last_close = float(last_row_4h["close"])

    # ── 4. Fetch 1d closed candles (zero look-ahead) ─────────────────
    # Store guarantees only candles where open_time + 1d <= now_ms are returned
    df_1d = store.get_candles(symbol, "1d", now_ms=now_ms)
    ema50_1d: float | None = None
    ema200_1d: float | None = None

    if not df_1d.empty:
        df_1d_ind = add_1d_indicators(df_1d)
        last_row_1d = df_1d_ind.iloc[-1]
        if not pd_isna(last_row_1d["ema50_1d"]):
            ema50_1d = float(last_row_1d["ema50_1d"])
        if not pd_isna(last_row_1d["ema200_1d"]):
            ema200_1d = float(last_row_1d["ema200_1d"])

    # ── 5. Market metrics (spread, ticker volume, lot size) ──────────
    current_price = last_close
    spread_bps = 3.0  # safe default
    volume_24h = 100_000_000.0  # safe default
    lot_size = 0.001
    min_notional = 10.0

    if client is not None:
        try:
            ticker = await client.fetch_ticker(symbol)
            if ticker.get("last"):
                current_price = float(ticker["last"])
            if ticker.get("quote_volume"):
                volume_24h = float(ticker["quote_volume"])
        except Exception as exc:
            logger.warning("Could not fetch ticker for %s: %s", symbol, exc)

        try:
            bid, ask = await client.fetch_order_book_top(symbol)
            mid = (bid + ask) / 2.0
            if mid > 0:
                spread_bps = ((ask - bid) / mid) * 10_000.0
        except Exception as exc:
            logger.warning("Could not fetch order book for %s: %s", symbol, exc)

        try:
            market_info = await client.fetch_market(symbol)
            lot_size = float(market_info.get("lot_size", lot_size))
            min_notional = float(market_info.get("min_notional", min_notional))
        except Exception as exc:
            logger.warning("Could not fetch market limits for %s: %s", symbol, exc)

    compact_4h = _df_to_compact_candles(df_4h, n_4h)
    compact_1d = _df_to_compact_candles(df_1d, n_1d)

    return MarketSnapshot(
        symbol=symbol,
        last_close=last_close,
        ema200_4h=ema200_4h,
        atr14_4h=atr14_4h,
        spread_bps=spread_bps,
        volume_24h=volume_24h,
        candle_timestamp=latest_dt,
        current_price=current_price,
        lot_size=lot_size,
        min_notional=min_notional,
        candles_4h=compact_4h,
        candles_1d=compact_1d,
        rsi14_4h=rsi14_4h,
        volume_sma20_4h=vol_sma20_4h,
        ema50_1d=ema50_1d,
        ema200_1d=ema200_1d,
    )


def pd_isna(val: Any) -> bool:
    """Helper to check if a value is nan or None."""
    import math

    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (ValueError, TypeError):
        return False
