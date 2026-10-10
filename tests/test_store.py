"""Tests for SQLite candle store – Phase 2.

Covers:
- In-progress candle exclusion (Rule 2)
- Idempotent upserts
- Gap detection (middle, start, end, none)
- Gap repair using a mocked exchange client
- Backfill and update_to_latest using mocked exchange client
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.data.exchange_client import ExchangeClient
from app.data.store import CandleStore, timeframe_to_ms


@pytest.fixture
def mem_store() -> CandleStore:
    """In-memory SQLite candle store for fast isolated testing."""
    return CandleStore(db_url="sqlite:///:memory:")


class TestStoreInProgressCandleExclusion:
    """Validate that in-progress candles are NEVER stored or returned."""

    def test_in_progress_candle_excluded_from_upsert(self, mem_store: CandleStore):
        # Suppose now is 10:00 (epoch ms: 36,000,000)
        now_ms = 36_000_000
        tf = "4h"  # 14,400,000 ms

        # Candle 1: 00:00 to 04:00 (open: 0, closes: 14,400,000 <= now_ms) -> CLOSED
        c1 = [0, 100.0, 105.0, 99.0, 104.0, 1000.0]
        # Candle 2: 04:00 to 08:00 (open: 14,400,000, closes: 28,800,000 <= now_ms) -> CLOSED
        c2 = [14_400_000, 104.0, 108.0, 103.0, 107.0, 1200.0]
        # Candle 3: 08:00 to 12:00 (open: 28,800,000, closes: 43,200,000 > now_ms) -> IN PROGRESS!
        c3 = [28_800_000, 107.0, 110.0, 106.0, 109.0, 500.0]

        upserted = mem_store.upsert_candles("BTC/USDT", tf, [c1, c2, c3], now_ms=now_ms)
        assert upserted == 2

        df = mem_store.get_candles("BTC/USDT", tf, now_ms=now_ms)
        assert len(df) == 2
        assert 28_800_000 not in df["open_time"].to_list()

    def test_get_candles_filters_out_any_future_timestamp(self, mem_store: CandleStore):
        # Force-inserting a candle that becomes in-progress later
        # e.g., if now_ms is checked during retrieval
        c1 = [0, 100.0, 105.0, 99.0, 104.0, 1000.0]
        c2 = [14_400_000, 104.0, 108.0, 103.0, 107.0, 1200.0]
        mem_store.upsert_candles("BTC/USDT", "4h", [c1, c2], now_ms=100_000_000)

        # At an earlier now_ms = 20_000_000, candle 2 (closes at 28.8M) is in-progress
        df = mem_store.get_candles("BTC/USDT", "4h", now_ms=20_000_000)
        assert len(df) == 1
        assert df.iloc[0]["open_time"] == 0


class TestStoreIdempotentUpsert:
    """Validate idempotent inserts and updates."""

    def test_duplicate_upsert_leaves_count_unchanged(self, mem_store: CandleStore):
        tf = "1h"
        tf_ms = timeframe_to_ms(tf)
        now_ms = 50_000_000

        c1 = [0, 100.0, 105.0, 95.0, 102.0, 500.0]
        c2 = [tf_ms, 102.0, 106.0, 101.0, 104.0, 600.0]

        res1 = mem_store.upsert_candles("ETH/USDT", tf, [c1, c2], now_ms=now_ms)
        assert res1 == 2
        assert len(mem_store.get_candles("ETH/USDT", tf, now_ms=now_ms)) == 2

        # Upsert same candles with updated close price
        c1_updated = [0, 100.0, 105.0, 95.0, 103.5, 500.0]
        res2 = mem_store.upsert_candles("ETH/USDT", tf, [c1_updated, c2], now_ms=now_ms)
        assert res2 == 2

        df = mem_store.get_candles("ETH/USDT", tf, now_ms=now_ms)
        assert len(df) == 2
        assert df.loc[df["open_time"] == 0, "close"].iloc[0] == 103.5


class TestStoreGapDetectionAndRepair:
    """Validate detection of missing candles and automatic repair."""

    def test_detect_no_gaps_on_continuous_series(self, mem_store: CandleStore):
        tf = "4h"
        tf_ms = timeframe_to_ms(tf)
        candles = [[i * tf_ms, 100.0 + i, 105.0 + i, 99.0 + i, 102.0 + i, 1000.0] for i in range(5)]
        mem_store.upsert_candles("BTC/USDT", tf, candles, now_ms=100 * tf_ms)
        gaps = mem_store.detect_gaps("BTC/USDT", tf)
        assert gaps == []

    def test_detect_internal_gap(self, mem_store: CandleStore):
        tf = "4h"
        tf_ms = timeframe_to_ms(tf)
        # Candles at t=0, t=1, MISSING t=2 and t=3, present t=4
        candles = [
            [0 * tf_ms, 100.0, 105.0, 99.0, 102.0, 1000.0],
            [1 * tf_ms, 102.0, 107.0, 101.0, 105.0, 1000.0],
            [4 * tf_ms, 108.0, 112.0, 107.0, 110.0, 1000.0],
        ]
        mem_store.upsert_candles("BTC/USDT", tf, candles, now_ms=100 * tf_ms)
        gaps = mem_store.detect_gaps("BTC/USDT", tf)
        # Gap spans from 2*tf_ms to 3*tf_ms
        assert len(gaps) == 1
        assert gaps[0] == (2 * tf_ms, 3 * tf_ms)

    @pytest.mark.asyncio
    async def test_repair_gaps_using_mock_client(self, mem_store: CandleStore):
        tf = "4h"
        tf_ms = timeframe_to_ms(tf)
        candles = [
            [0 * tf_ms, 100.0, 105.0, 99.0, 102.0, 1000.0],
            [1 * tf_ms, 102.0, 107.0, 101.0, 105.0, 1000.0],
            [4 * tf_ms, 108.0, 112.0, 107.0, 110.0, 1000.0],
        ]
        mem_store.upsert_candles("BTC/USDT", tf, candles, now_ms=100 * tf_ms)

        gaps = mem_store.detect_gaps("BTC/USDT", tf)
        assert len(gaps) == 1

        mock_client = AsyncMock(spec=ExchangeClient)
        # When repair fetches since=2*tf_ms, return missing candles 2 and 3
        missing = [
            [2 * tf_ms, 105.0, 109.0, 104.0, 106.0, 1000.0],
            [3 * tf_ms, 106.0, 110.0, 105.0, 108.0, 1000.0],
        ]
        mock_client.fetch_ohlcv.return_value = missing

        repaired = await mem_store.repair_gaps(
            "BTC/USDT", tf, gaps, client=mock_client, now_ms=100 * tf_ms
        )
        assert repaired == 2

        # Verify no gaps remain
        remaining_gaps = mem_store.detect_gaps("BTC/USDT", tf)
        assert remaining_gaps == []
        assert len(mem_store.get_candles("BTC/USDT", tf, now_ms=100 * tf_ms)) == 5


class TestStoreBackfillAndUpdate:
    """Validate backfill and incremental update loops with mocked exchange."""

    @pytest.mark.asyncio
    async def test_backfill_loop(self, mem_store: CandleStore):
        tf = "4h"
        tf_ms = timeframe_to_ms(tf)
        now_ms = 100 * tf_ms

        mock_client = AsyncMock(spec=ExchangeClient)
        # Mock returning 10 closed candles
        batch = [
            [(80 + i) * tf_ms, 100.0 + i, 105.0 + i, 95.0 + i, 102.0 + i, 1000.0] for i in range(10)
        ]
        mock_client.fetch_ohlcv.side_effect = [batch, []]

        upserted = await mem_store.backfill(
            "BTC/USDT", tf, days=3, client=mock_client, now_ms=now_ms
        )
        assert upserted == 10
        assert len(mem_store.get_candles("BTC/USDT", tf, now_ms=now_ms)) == 10

    @pytest.mark.asyncio
    async def test_update_to_latest(self, mem_store: CandleStore):
        tf = "4h"
        tf_ms = timeframe_to_ms(tf)
        now_ms = 10 * tf_ms

        # Initial candle stored at t=0
        mem_store.upsert_candles(
            "BTC/USDT", tf, [[0, 100.0, 105.0, 95.0, 102.0, 1000.0]], now_ms=now_ms
        )

        mock_client = AsyncMock(spec=ExchangeClient)
        # Mock next closed candles at t=1, t=2
        mock_client.fetch_ohlcv.side_effect = [
            [
                [1 * tf_ms, 102.0, 106.0, 101.0, 104.0, 1000.0],
                [2 * tf_ms, 104.0, 108.0, 103.0, 107.0, 1000.0],
            ],
            [],
        ]

        new_count = await mem_store.update_to_latest(
            "BTC/USDT", tf, client=mock_client, now_ms=now_ms
        )
        assert new_count == 2
        assert len(mem_store.get_candles("BTC/USDT", tf, now_ms=now_ms)) == 3
