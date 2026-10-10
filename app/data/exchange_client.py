"""Async CCXT wrapper for market data fetching – Rule / Phase 2.

Features:
- Rate limiting enabled via ccxt's built-in token bucket
- Automatic retry with exponential backoff on transient network/rate-limit errors
- Typed exceptions mapped to app.common.errors hierarchy
- Methods:
    - fetch_ohlcv(symbol, timeframe, since, limit)
    - fetch_ticker(symbol)
    - fetch_order_book_top(symbol) -> (best_bid, best_ask)
    - fetch_markets() and fetch_market(symbol)
    - close()
"""

from __future__ import annotations

import asyncio
from typing import Any

import ccxt.async_support as ccxt_async

from app.common.errors import (
    DataError,
    ExchangeError,
    NetworkError,
    RateLimitError,
)
from app.common.logging import get_logger

logger = get_logger(__name__)


class ExchangeClient:
    """Resilient async client wrapping CCXT exchange operations."""

    def __init__(
        self,
        exchange_id: str = "binance",
        max_retries: int = 3,
        base_backoff_s: float = 0.5,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.exchange_id = exchange_id.lower()
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s

        exchange_class = getattr(ccxt_async, self.exchange_id, None)
        if exchange_class is None:
            raise ExchangeError(f"Unsupported exchange id: {exchange_id}")

        config: dict[str, Any] = {
            "enableRateLimit": True,
            "timeout": 15000,
        }
        if options:
            config.update(options)

        self._exchange = exchange_class(config)
        self._markets_loaded = False

    async def close(self) -> None:
        """Close the underlying exchange session."""
        if self._exchange:
            await self._exchange.close()

    async def __aenter__(self) -> ExchangeClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def _execute_with_retry(self, coro_func: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute a coroutine function with exponential backoff on transient errors."""
        attempt = 0
        while True:
            try:
                return await coro_func(*args, **kwargs)
            except ccxt_async.RateLimitExceeded as exc:
                attempt += 1
                if attempt > self.max_retries:
                    logger.error("Rate limit exceeded; max retries reached: %s", exc)
                    raise RateLimitError(
                        f"Rate limit exceeded after {attempt} attempts: {exc}"
                    ) from exc
                delay = self.base_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "Rate limit hit; backing off %.2fs (attempt %d/%d)",
                    delay,
                    attempt,
                    self.max_retries,
                )
                await asyncio.sleep(delay)

            except (TimeoutError, ccxt_async.NetworkError) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    logger.error("Network error; max retries reached: %s", exc)
                    raise NetworkError(f"Network error after {attempt} attempts: {exc}") from exc
                delay = self.base_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "Network failure; backing off %.2fs (attempt %d/%d): %s",
                    delay,
                    attempt,
                    self.max_retries,
                    exc,
                )
                await asyncio.sleep(delay)

            except (ccxt_async.ExchangeNotAvailable, ccxt_async.RequestTimeout) as exc:
                attempt += 1
                if attempt > self.max_retries:
                    logger.error("Exchange unavailable; max retries reached: %s", exc)
                    raise ExchangeError(
                        f"Exchange unavailable after {attempt} attempts: {exc}"
                    ) from exc
                delay = self.base_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "Exchange transient failure; backing off %.2fs (attempt %d/%d): %s",
                    delay,
                    attempt,
                    self.max_retries,
                    exc,
                )
                await asyncio.sleep(delay)

            except (ccxt_async.BadSymbol, ccxt_async.ExchangeError) as exc:
                # Fatal or non-retryable exchange error
                logger.error("Non-retryable exchange error: %s", exc)
                raise ExchangeError(f"Exchange error: {exc}") from exc

            except Exception as exc:
                logger.error("Unexpected error in exchange client: %s", exc)
                raise DataError(f"Unexpected exchange client failure: {exc}") from exc

    async def load_markets(self, reload: bool = False) -> dict[str, Any]:
        """Load and cache exchange markets definition."""
        if not self._markets_loaded or reload:
            await self._execute_with_retry(self._exchange.load_markets, reload)
            self._markets_loaded = True
        return self._exchange.markets

    async def fetch_markets(self) -> dict[str, Any]:
        """Fetch all markets metadata."""
        return await self.load_markets()

    async def fetch_market(self, symbol: str) -> dict[str, Any]:
        """Fetch market metadata for a single symbol (lot size, precision, min notional)."""
        markets = await self.load_markets()
        market = markets.get(symbol)
        if not market:
            raise ExchangeError(f"Symbol {symbol} not found on exchange {self.exchange_id}")

        lot_size = 0.001
        min_notional = 10.0
        price_precision = 2
        amount_precision = 4

        precision = market.get("precision", {})
        if "price" in precision and precision["price"] is not None:
            price_precision = precision["price"]
        if "amount" in precision and precision["amount"] is not None:
            amount_precision = precision["amount"]

        limits = market.get("limits", {})
        amount_limits = limits.get("amount", {})
        if amount_limits.get("min"):
            lot_size = float(amount_limits["min"])

        cost_limits = limits.get("cost", {})
        if cost_limits.get("min"):
            min_notional = float(cost_limits["min"])

        return {
            "symbol": symbol,
            "lot_size": lot_size,
            "min_notional": min_notional,
            "price_precision": price_precision,
            "amount_precision": amount_precision,
            "raw": market,
        }

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[float]]:
        """Fetch OHLCV candles from the exchange.

        Returns list of candles in format:
        [[timestamp_ms, open, high, low, close, volume], ...]
        """
        return await self._execute_with_retry(
            self._exchange.fetch_ohlcv,
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit,
        )

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch current ticker for *symbol*."""
        ticker = await self._execute_with_retry(self._exchange.fetch_ticker, symbol)
        return {
            "symbol": symbol,
            "last": ticker.get("last") or ticker.get("close"),
            "bid": ticker.get("bid"),
            "ask": ticker.get("ask"),
            "quote_volume": ticker.get("quoteVolume") or 0.0,
            "base_volume": ticker.get("baseVolume") or 0.0,
            "timestamp": ticker.get("timestamp"),
            "raw": ticker,
        }

    async def fetch_tickers(self, symbols: list[str] | None = None) -> dict[str, Any]:
        """Fetch tickers for *symbols* (or all exchange spot tickers if None)."""
        return await self._execute_with_retry(self._exchange.fetch_tickers, symbols)

    async def fetch_order_book_top(self, symbol: str) -> tuple[float, float]:
        """Fetch best bid and best ask from the order book.

        Returns `(best_bid, best_ask)`.
        """
        order_book = await self._execute_with_retry(
            self._exchange.fetch_order_book,
            symbol,
            limit=5,
        )
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        if not bids or not asks:
            raise DataError(f"Incomplete order book returned for {symbol}")

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        return best_bid, best_ask
