"""Tests for ExchangeClient retry logic, backoff, and exception mapping – Phase 2.

All tests use mocked underlying CCXT exchange; no real network calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import ccxt.async_support as ccxt_async
import pytest

from app.common.errors import ExchangeError, NetworkError, RateLimitError
from app.data.exchange_client import ExchangeClient


@pytest.fixture
def client() -> ExchangeClient:
    # Use very small backoff for fast tests
    return ExchangeClient(exchange_id="binance", max_retries=3, base_backoff_s=0.01)


class TestExchangeClientRetryAndBackoff:
    """Validate exponential backoff and retry behavior on mocked failures."""

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_exceeded_then_succeed(self, client: ExchangeClient):
        # 2 failures followed by 1 success
        mock_ohlcv = AsyncMock(
            side_effect=[
                ccxt_async.RateLimitExceeded("Rate limit exceeded"),
                ccxt_async.RateLimitExceeded("Rate limit exceeded"),
                [[1000, 10.0, 11.0, 9.0, 10.5, 100.0]],
            ]
        )
        client._exchange.fetch_ohlcv = mock_ohlcv

        res = await client.fetch_ohlcv("BTC/USDT", timeframe="4h")
        assert len(res) == 1
        assert mock_ohlcv.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_network_error_then_succeed(self, client: ExchangeClient):
        mock_ticker = AsyncMock(
            side_effect=[
                ccxt_async.NetworkError("Connection reset"),
                {"symbol": "BTC/USDT", "last": 65_000.0, "quoteVolume": 10_000_000.0},
            ]
        )
        client._exchange.fetch_ticker = mock_ticker

        res = await client.fetch_ticker("BTC/USDT")
        assert res["last"] == 65_000.0
        assert mock_ticker.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises_ratelimit_error(self, client: ExchangeClient):
        mock_ohlcv = AsyncMock(side_effect=ccxt_async.RateLimitExceeded("429 Too Many Requests"))
        client._exchange.fetch_ohlcv = mock_ohlcv

        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            await client.fetch_ohlcv("BTC/USDT")
        # max_retries=3 means initial attempt + 3 retries = 4 total attempts
        assert mock_ohlcv.call_count == 4

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises_network_error(self, client: ExchangeClient):
        mock_ohlcv = AsyncMock(side_effect=ccxt_async.NetworkError("Host unreachable"))
        client._exchange.fetch_ohlcv = mock_ohlcv

        with pytest.raises(NetworkError, match="Network error"):
            await client.fetch_ohlcv("BTC/USDT")
        assert mock_ohlcv.call_count == 4

    @pytest.mark.asyncio
    async def test_non_retryable_exchange_error_fails_immediately(self, client: ExchangeClient):
        mock_ohlcv = AsyncMock(side_effect=ccxt_async.BadSymbol("Symbol not found: INVALID/USDT"))
        client._exchange.fetch_ohlcv = mock_ohlcv

        with pytest.raises(ExchangeError, match="Symbol not found"):
            await client.fetch_ohlcv("INVALID/USDT")
        # Fatal error must fail on first attempt without wasteful retries
        assert mock_ohlcv.call_count == 1


class TestExchangeClientMetadataMethods:
    """Validate parsing of order book top, ticker, and market limits."""

    @pytest.mark.asyncio
    async def test_fetch_order_book_top(self, client: ExchangeClient):
        client._exchange.fetch_order_book = AsyncMock(
            return_value={
                "bids": [[60_000.0, 1.5], [59_990.0, 2.0]],
                "asks": [[60_010.0, 1.0], [60_020.0, 3.0]],
            }
        )
        bid, ask = await client.fetch_order_book_top("BTC/USDT")
        assert bid == 60_000.0
        assert ask == 60_010.0

    @pytest.mark.asyncio
    async def test_fetch_market_parsing(self, client: ExchangeClient):
        client._exchange.load_markets = AsyncMock(
            return_value={
                "BTC/USDT": {
                    "limits": {
                        "amount": {"min": 0.0002},
                        "cost": {"min": 15.0},
                    },
                    "precision": {"price": 2, "amount": 5},
                }
            }
        )
        client._exchange.markets = {
            "BTC/USDT": {
                "limits": {
                    "amount": {"min": 0.0002},
                    "cost": {"min": 15.0},
                },
                "precision": {"price": 2, "amount": 5},
            }
        }
        client._markets_loaded = True

        info = await client.fetch_market("BTC/USDT")
        assert info["symbol"] == "BTC/USDT"
        assert info["lot_size"] == 0.0002
        assert info["min_notional"] == 15.0
        assert info["price_precision"] == 2
        assert info["amount_precision"] == 5
