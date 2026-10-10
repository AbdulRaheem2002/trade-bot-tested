"""Tests for MarketSnapshot builder and zero look-ahead multi-timeframe alignment – Phase 2."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from app.common.errors import StaleDataError
from app.data.exchange_client import ExchangeClient
from app.data.snapshot import build_snapshot
from app.data.store import CandleStore, timeframe_to_ms


@pytest.fixture
def snapshot_store() -> CandleStore:
    return CandleStore(db_url="sqlite:///:memory:")


class TestSnapshotFreshness:
    """Validate data freshness rejection."""

    @pytest.mark.asyncio
    async def test_stale_snapshot_rejected(self, snapshot_store: CandleStore):
        # Suppose now is 2025-06-15 08:05:00 UTC
        # The expected latest closed 4h candle opened at 2025-06-15 04:00:00 UTC (and closed at 08:00)
        now = datetime(2025, 6, 15, 8, 5, 0, tzinfo=UTC)
        tf_4h = "4h"

        # Stored candle opened at 2025-06-15 00:00:00 UTC (one cycle older than expected 04:00)
        stale_open_ms = int(datetime(2025, 6, 15, 0, 0, 0, tzinfo=UTC).timestamp() * 1000)
        snapshot_store.upsert_candles(
            "BTC/USDT",
            tf_4h,
            [[stale_open_ms, 70_000.0, 71_000.0, 69_000.0, 70_500.0, 1000.0]],
            now_ms=int(now.timestamp() * 1000),
        )

        with pytest.raises(StaleDataError, match="stale"):
            await build_snapshot("BTC/USDT", store=snapshot_store, now=now)

    @pytest.mark.asyncio
    async def test_fresh_snapshot_accepted(self, snapshot_store: CandleStore):
        now = datetime(2025, 6, 15, 8, 5, 0, tzinfo=UTC)
        tf_4h = "4h"
        tf_4h_ms = timeframe_to_ms(tf_4h)

        # Expected latest closed 4h candle is at 2025-06-15 04:00:00 UTC
        expected_open_ms = int(datetime(2025, 6, 15, 4, 0, 0, tzinfo=UTC).timestamp() * 1000)

        # Store 205 closed 4h candles ending at expected_open_ms
        candles = []
        for i in range(205):
            t = expected_open_ms - ((204 - i) * tf_4h_ms)
            candles.append([t, 60_000.0 + i, 61_000.0 + i, 59_000.0 + i, 60_500.0 + i, 1000.0])

        snapshot_store.upsert_candles(
            "BTC/USDT",
            tf_4h,
            candles,
            now_ms=int(now.timestamp() * 1000),
        )

        snapshot = await build_snapshot("BTC/USDT", store=snapshot_store, now=now)
        assert snapshot.symbol == "BTC/USDT"
        assert snapshot.candle_timestamp == datetime(2025, 6, 15, 4, 0, 0, tzinfo=UTC)
        assert snapshot.last_close > 0
        assert snapshot.ema200_4h > 0
        assert snapshot.atr14_4h > 0
        assert len(snapshot.candles_4h) == 60


class TestMultiTimeframeZeroLookahead:
    """Validate that 1d values NEVER leak unclosed/future day data."""

    @pytest.mark.asyncio
    async def test_zero_lookahead_alignment(self, snapshot_store: CandleStore):
        # Time: 2025-06-15 12:00:00 UTC (noon of June 15)
        # Expected closed 4h candle: opened 2025-06-15 08:00:00 UTC
        # For 1d candles:
        # Candle 1d for June 15 opened 2025-06-15 00:00:00 UTC, but only closes at 2025-06-16 00:00:00 UTC!
        # Thus, on June 15 at 12:00 UTC, the latest CLOSED 1d candle is June 14!
        now = datetime(2025, 6, 15, 12, 0, 0, tzinfo=UTC)
        now_ms = int(now.timestamp() * 1000)

        # 1. Populate 4h candles
        expected_4h_open_ms = int(datetime(2025, 6, 15, 8, 0, 0, tzinfo=UTC).timestamp() * 1000)
        tf_4h_ms = timeframe_to_ms("4h")
        candles_4h = [
            [expected_4h_open_ms - (i * tf_4h_ms), 50_000.0, 51_000.0, 49_000.0, 50_500.0, 1000.0]
            for i in range(210)
        ]
        candles_4h.reverse()
        snapshot_store.upsert_candles("BTC/USDT", "4h", candles_4h, now_ms=now_ms)

        # 2. Populate 1d candles:
        # 1d candle for June 14 (closed at June 15 00:00)
        june_14_ms = int(datetime(2025, 6, 14, 0, 0, 0, tzinfo=UTC).timestamp() * 1000)
        # 1d candle for June 15 (closes June 16 00:00 -> IN PROGRESS at now!)
        june_15_ms = int(datetime(2025, 6, 15, 0, 0, 0, tzinfo=UTC).timestamp() * 1000)
        # 1d candle for June 16 (future)
        june_16_ms = int(datetime(2025, 6, 16, 0, 0, 0, tzinfo=UTC).timestamp() * 1000)

        tf_1d_ms = timeframe_to_ms("1d")
        historical_1d = [
            [
                june_14_ms - (i * tf_1d_ms),
                40_000.0 + i,
                41_000.0 + i,
                39_000.0 + i,
                40_500.0 + i,
                5000.0,
            ]
            for i in range(100)
        ]
        historical_1d.reverse()
        snapshot_store.upsert_candles("BTC/USDT", "1d", historical_1d, now_ms=now_ms)

        # Attempt to insert June 15 (in progress) and June 16 (future)
        snapshot_store.upsert_candles(
            "BTC/USDT",
            "1d",
            [
                [june_15_ms, 999_999.0, 999_999.0, 999_999.0, 999_999.0, 100_000.0],
                [june_16_ms, 999_999.0, 999_999.0, 999_999.0, 999_999.0, 100_000.0],
            ],
            now_ms=now_ms,
        )

        snapshot = await build_snapshot("BTC/USDT", store=snapshot_store, now=now)

        # Verify that 1d candles in the snapshot do NOT contain June 15 or June 16
        for c in snapshot.candles_1d:
            assert c["t"] <= june_14_ms
            assert c["c"] < 900_000.0  # unpolluted by future price spike

        # Indicator values computed on 1d must NOT reflect the future 999,999 price
        if snapshot.ema50_1d is not None:
            assert snapshot.ema50_1d < 100_000.0


class TestSnapshotWithMockClient:
    """Validate integration with ExchangeClient for live ticker, spread, and limits."""

    @pytest.mark.asyncio
    async def test_snapshot_fetches_client_metadata(self, snapshot_store: CandleStore):
        now = datetime(2025, 6, 15, 8, 5, 0, tzinfo=UTC)
        now_ms = int(now.timestamp() * 1000)
        expected_4h_open_ms = int(datetime(2025, 6, 15, 4, 0, 0, tzinfo=UTC).timestamp() * 1000)
        tf_4h_ms = timeframe_to_ms("4h")

        candles = [
            [expected_4h_open_ms - (i * tf_4h_ms), 60_000.0, 61_000.0, 59_000.0, 60_500.0, 1000.0]
            for i in range(205)
        ]
        candles.reverse()
        snapshot_store.upsert_candles("BTC/USDT", "4h", candles, now_ms=now_ms)

        mock_client = AsyncMock(spec=ExchangeClient)
        mock_client.fetch_ticker.return_value = {
            "symbol": "BTC/USDT",
            "last": 60_800.0,
            "quote_volume": 150_000_000.0,
        }
        mock_client.fetch_order_book_top.return_value = (60_790.0, 60_810.0)
        mock_client.fetch_market.return_value = {
            "symbol": "BTC/USDT",
            "lot_size": 0.0001,
            "min_notional": 5.0,
        }

        snap = await build_snapshot("BTC/USDT", store=snapshot_store, client=mock_client, now=now)
        assert snap.current_price == 60_800.0
        assert snap.volume_24h == 150_000_000.0
        # spread: (60810 - 60790) / 60800 * 10000 = 20 / 60800 * 10000 = ~3.289 bps
        assert snap.spread_bps == pytest.approx(3.28947, rel=1e-3)
        assert snap.lot_size == 0.0001
        assert snap.min_notional == 5.0
