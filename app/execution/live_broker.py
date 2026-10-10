"""Live trading CCXT broker implementation – Phase 9.

Implements the Broker contract routing orders directly to real/mocked exchange endpoints:
1. Idempotent client order IDs: duplicate IDs return original OrderFill.
2. Immediate protective exchange-side stop order placement:
   - Placed immediately after each entry fill (stop-market or stop-limit).
   - If placing the stop order fails, immediately emergency-closes the position and alerts.
3. Partial fill handling: sizes protective stop to actual filled quantity.
4. Stop synchronization: updates exchange stop order whenever invalidation price tightens.
5. Max capital cap: respects config.live.max_capital regardless of account equity.
6. Order cancellation and reconciliation helpers.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from datetime import UTC, datetime
from typing import Any

from app.common.alerts import AlertManager
from app.common.enums import ConvictionTier, Side
from app.common.errors import (
    ExchangeError,
    InsufficientFundsError,
    NetworkError,
    OrderRejectedError,
)
from app.common.logging import get_logger
from app.common.models import OrderFill, Position
from app.config.settings import TradingConfig
from app.data.exchange_client import ExchangeClient
from app.execution.base import Broker

logger = get_logger(__name__)


class LiveBroker(Broker):
    """Production live execution broker communicating via CCXT."""

    def __init__(
        self,
        exchange_client: ExchangeClient | None = None,
        exchange: Any | None = None,
        config: TradingConfig | None = None,
        alert_manager: AlertManager | None = None,
        max_capital: float | None = None,
    ) -> None:
        self.exchange_client = exchange_client
        self.config = config
        self.alert_manager = alert_manager

        # Resolve CCXT exchange instance
        if exchange is not None:
            self._exchange = exchange
        elif exchange_client is not None:
            self._exchange = exchange_client._exchange
        else:
            self._exchange = None

        # Max capital cap
        if max_capital is not None:
            self.max_capital = float(max_capital)
        elif config and hasattr(config, "live") and config.live.max_capital:
            self.max_capital = float(config.live.max_capital)
        else:
            self.max_capital = 10_000.0

        # State tracking
        self._fills_by_client_id: dict[str, OrderFill] = {}
        self._stop_orders: dict[str, dict[str, Any]] = {}  # {symbol: {order_id, stop_price, qty, side}}
        self._positions: dict[str, Position] = {}
        self._is_started: bool = False

    @property
    def exchange(self) -> Any:
        if self._exchange is None:
            raise ExchangeError("LiveBroker CCXT exchange instance is not initialized.")
        return self._exchange

    async def start(self) -> None:
        """Start broker and verify connectivity."""
        self._is_started = True
        logger.info("LiveBroker initialized (max_capital cap: $%.2f)", self.max_capital)

    async def stop(self) -> None:
        """Gracefully close exchange session."""
        self._is_started = False
        if self.exchange_client:
            await self.exchange_client.close()
        elif self._exchange and hasattr(self._exchange, "close"):
            try:
                res = self._exchange.close()
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass
        logger.info("LiveBroker shut down cleanly.")

    # ── Market Orders & Protective Stops ─────────────────────────────────────

    async def place_market_order(
        self,
        symbol: str,
        side: Side,
        qty: float,
        client_order_id: str,
        **kwargs: Any,
    ) -> OrderFill:
        """Place an aggressive market order filled immediately at market price.

        Idempotent: repeating the same client_order_id returns the existing fill.
        Places protective exchange-side stop immediately after successful entry fill.
        """
        # 1. Idempotency check
        if client_order_id in self._fills_by_client_id:
            logger.info("Idempotent market order retrieved: %s", client_order_id)
            return self._fills_by_client_id[client_order_id]

        ccxt_side = "buy" if side == Side.LONG else "sell"
        params: dict[str, Any] = {"clientOrderId": client_order_id}

        # 2. Place order on exchange
        raw_order: dict[str, Any] | None = None
        try:
            raw_order = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=ccxt_side,
                amount=qty,
                params=params,
            )
        except (TimeoutError, Exception) as exc:
            # Handle transient/timeout or explicit errors
            exc_name = type(exc).__name__
            if "Timeout" in exc_name or "Network" in exc_name:
                logger.warning(
                    "Network timeout placing order %s (%s). Attempting verification...",
                    client_order_id,
                    symbol,
                )
                raw_order = await self._resolve_order_after_timeout(symbol, client_order_id)
                if raw_order is None:
                    raise NetworkError(
                        f"Order {client_order_id} timed out and could not be verified on exchange: {exc}"
                    ) from exc
            elif "InsufficientFunds" in exc_name:
                raise InsufficientFundsError(f"Insufficient funds for order {client_order_id}: {exc}") from exc
            else:
                logger.error("Order submission rejected by exchange: %s", exc)
                raise OrderRejectedError(f"Exchange rejected order {client_order_id}: {exc}") from exc

        if not raw_order:
            raise OrderRejectedError(f"Failed to obtain order response for {client_order_id}")

        # 3. Parse fill and handle partial fills
        filled_qty = float(raw_order.get("filled") or 0.0)
        status = raw_order.get("status", "")

        # If exchange reports 0 filled and closed/canceled
        if filled_qty <= 0 and status in ("canceled", "rejected", "expired"):
            raise OrderRejectedError(
                f"Order {client_order_id} ended with status '{status}' and zero filled quantity."
            )

        # Fallback if simulator/exchange didn't populate filled yet assumed filled
        actual_qty = filled_qty if filled_qty > 0 else float(raw_order.get("amount") or qty)

        avg_price = float(raw_order.get("average") or raw_order.get("price") or 0.0)
        if avg_price <= 0 and raw_order.get("cost") and actual_qty > 0:
            avg_price = float(raw_order["cost"]) / actual_qty
        if avg_price <= 0:
            avg_price = float(kwargs.get("expected_price") or 1.0)

        fee = 0.0
        if raw_order.get("fee") and isinstance(raw_order["fee"], dict):
            fee = float(raw_order["fee"].get("cost") or 0.0)

        order_fill = OrderFill(
            fill_id=str(raw_order.get("id") or f"fill_{uuid.uuid4().hex[:12]}"),
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            price=avg_price,
            quantity=actual_qty,
            fee=fee,
            slippage_bps=0.0,
            timestamp=datetime.now(UTC),
        )
        self._fills_by_client_id[client_order_id] = order_fill

        # 4. Immediate protective exchange-side stop order placement for entry orders
        invalidation_price = kwargs.get("invalidation_price")
        if invalidation_price is not None and float(invalidation_price) > 0:
            stop_side = Side.SHORT if side == Side.LONG else Side.LONG
            try:
                stop_res = await self._place_exchange_stop_order(
                    symbol=symbol,
                    side=stop_side,
                    qty=actual_qty,
                    stop_price=float(invalidation_price),
                )
                self._stop_orders[symbol] = {
                    "order_id": str(stop_res.get("id")),
                    "symbol": symbol,
                    "side": stop_side,
                    "qty": actual_qty,
                    "stop_price": float(invalidation_price),
                    "raw": stop_res,
                }
                logger.info(
                    "Placed protective exchange stop for %s at %.4f (Order: %s)",
                    symbol,
                    invalidation_price,
                    stop_res.get("id"),
                )
            except Exception as stop_exc:
                logger.critical(
                    "CRITICAL: Failed to place exchange-side protective stop for %s: %s. "
                    "Initiating immediate emergency position close!",
                    symbol,
                    stop_exc,
                )
                if self.alert_manager:
                    await self.alert_manager.critical(
                        title=f"CRITICAL: Protective Stop Placement Failed ({symbol})",
                        message=(
                            f"Protective stop order for {symbol} at {invalidation_price} failed: {stop_exc}. "
                            f"Immediately emergency-closing position of {actual_qty} {symbol}."
                        ),
                        data={"symbol": symbol, "qty": actual_qty, "error": str(stop_exc)},
                    )

                # Emergency market close
                await self._emergency_close_position(symbol=symbol, qty=actual_qty, position_side=side)
                raise OrderRejectedError(
                    f"Protective stop placement failed for {symbol}: {stop_exc}. Position was emergency-closed."
                ) from stop_exc

        return order_fill

    async def _place_exchange_stop_order(
        self,
        symbol: str,
        side: Side | str,
        qty: float,
        stop_price: float,
    ) -> dict[str, Any]:
        """Place protective stop order on exchange, supporting stop_market or stop_limit."""
        ccxt_side = "sell" if side in (Side.SHORT, "short", "sell") else "buy"
        stop_cid = f"stop_{symbol.replace('/', '_')}_{uuid.uuid4().hex[:6]}"

        # Attempt 1: Stop Market
        try:
            return await self.exchange.create_order(
                symbol=symbol,
                type="stop_market",
                side=ccxt_side,
                amount=qty,
                params={"stopPrice": stop_price, "clientOrderId": stop_cid},
            )
        except Exception as exc1:
            # Attempt 2: Stop Loss Limit (with price set beyond stop price to guarantee fill)
            try:
                limit_price = stop_price * 0.99 if ccxt_side == "sell" else stop_price * 1.01
                return await self.exchange.create_order(
                    symbol=symbol,
                    type="stop_loss_limit",
                    side=ccxt_side,
                    amount=qty,
                    price=limit_price,
                    params={"stopPrice": stop_price, "clientOrderId": stop_cid},
                )
            except Exception:
                # Attempt 3: Generic 'stop'
                try:
                    return await self.exchange.create_order(
                        symbol=symbol,
                        type="stop",
                        side=ccxt_side,
                        amount=qty,
                        params={"stopPrice": stop_price, "clientOrderId": stop_cid},
                    )
                except Exception:
                    raise exc1 from None

    async def _emergency_close_position(
        self,
        symbol: str,
        qty: float,
        position_side: Side,
    ) -> OrderFill | None:
        """Immediately market-close position upon protective stop failure."""
        close_side = "sell" if position_side == Side.LONG else "buy"
        emerg_cid = f"emerg_{uuid.uuid4().hex[:8]}"
        try:
            raw = await self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=close_side,
                amount=qty,
                params={"clientOrderId": emerg_cid},
            )
            price = float(raw.get("average") or raw.get("price") or 0.0)
            fill = OrderFill(
                fill_id=str(raw.get("id") or f"fill_{uuid.uuid4().hex[:8]}"),
                client_order_id=emerg_cid,
                symbol=symbol,
                side=Side.SHORT if position_side == Side.LONG else Side.LONG,
                price=price,
                quantity=qty,
                fee=0.0,
                slippage_bps=0.0,
                timestamp=datetime.now(UTC),
            )
            logger.info("Emergency close completed for %s (%s)", symbol, fill.fill_id)
            return fill
        except Exception as exc:
            logger.critical("FATAL: Emergency position close failed for %s: %s", symbol, exc)
            return None

    async def _resolve_order_after_timeout(
        self, symbol: str, client_order_id: str
    ) -> dict[str, Any] | None:
        """Query exchange after a network timeout to check if order was registered."""
        # 1. Direct fetch_order by clientOrderId
        if hasattr(self.exchange, "fetch_order"):
            try:
                ord_info = await self.exchange.fetch_order(
                    id=client_order_id,
                    symbol=symbol,
                    params={"clientOrderId": client_order_id},
                )
                if ord_info:
                    return ord_info
            except Exception:
                pass

        # 2. Check open orders
        try:
            open_orders = await self.exchange.fetch_open_orders(symbol)
            for ord_info in open_orders:
                if ord_info.get("clientOrderId") == client_order_id:
                    return ord_info
        except Exception:
            pass

        # 3. Check closed orders
        if hasattr(self.exchange, "fetch_closed_orders"):
            try:
                closed_orders = await self.exchange.fetch_closed_orders(symbol, limit=20)
                for ord_info in closed_orders:
                    if ord_info.get("clientOrderId") == client_order_id:
                        return ord_info
            except Exception:
                pass

        return None

    async def set_protective_stop(self, symbol: str, price: float) -> None:
        """Sync exchange-side stop order when invalidation price is tightened."""
        if price <= 0:
            raise ValueError(f"Stop price must be positive, got {price}")

        existing = self._stop_orders.get(symbol)
        if existing:
            old_order_id = existing["order_id"]
            qty = existing["qty"]
            stop_side = existing["side"]

            # Cancel old stop order on exchange
            try:
                await self.exchange.cancel_order(old_order_id, symbol)
            except Exception as exc:
                logger.warning("Failed to cancel old stop order %s for %s: %s", old_order_id, symbol, exc)

            # Place updated stop order
            new_stop = await self._place_exchange_stop_order(
                symbol=symbol,
                side=stop_side,
                qty=qty,
                stop_price=price,
            )
            self._stop_orders[symbol] = {
                "order_id": str(new_stop.get("id")),
                "symbol": symbol,
                "side": stop_side,
                "qty": qty,
                "stop_price": price,
                "raw": new_stop,
            }
            logger.info("Synchronized exchange-side stop for %s to %.4f", symbol, price)

    async def close_position(
        self,
        symbol: str,
        reason: str = "",
        client_order_id: str | None = None,
    ) -> OrderFill | None:
        """Close an existing position in full on the exchange."""
        cid = client_order_id or f"close_{symbol.replace('/', '_')}_{uuid.uuid4().hex[:6]}"
        if cid in self._fills_by_client_id:
            return self._fills_by_client_id[cid]

        # 1. Cancel active exchange stop order first to prevent duplicate fills
        if symbol in self._stop_orders:
            stop_info = self._stop_orders.pop(symbol)
            try:
                await self.exchange.cancel_order(stop_info["order_id"], symbol)
                logger.info("Cancelled protective stop %s before position close", stop_info["order_id"])
            except Exception as exc:
                logger.warning("Failed to cancel protective stop %s: %s", stop_info["order_id"], exc)

        # 2. Determine quantity to close from positions or exchange
        pos = self._positions.get(symbol)
        qty = pos.quantity if pos else 0.0
        if qty <= 0:
            # Query open positions on exchange
            open_pos = await self.get_open_positions()
            if symbol in open_pos:
                qty = open_pos[symbol].quantity

        if qty <= 0:
            logger.info("No position found to close for %s", symbol)
            return None

        side = Side.SHORT if (pos and pos.side == Side.LONG) else Side.LONG
        fill = await self.place_market_order(
            symbol=symbol,
            side=side,
            qty=qty,
            client_order_id=cid,
            reason=reason,
        )
        self._positions.pop(symbol, None)
        return fill

    # ── Balances, Positions, and Orders ───────────────────────────────────────

    async def get_balances(self) -> dict[str, float]:
        """Fetch balances and enforce live.max_capital equity ceiling."""
        bal = await self.exchange.fetch_balance()
        quote = "USDT"

        # Resolve cash
        raw_cash = 0.0
        if quote in bal:
            raw_cash = float(bal[quote].get("free") or 0.0)
        elif "free" in bal and quote in bal["free"]:
            raw_cash = float(bal["free"][quote] or 0.0)

        # Resolve equity
        raw_equity = 0.0
        if quote in bal:
            raw_equity = float(bal[quote].get("total") or raw_cash)
        elif "total" in bal and quote in bal["total"]:
            raw_equity = float(bal["total"][quote] or raw_cash)
        else:
            raw_equity = raw_cash

        # Enforce live.max_capital ceiling
        capped_equity = min(raw_equity, self.max_capital)
        capped_cash = min(raw_cash, self.max_capital)

        return {
            "cash": capped_cash,
            "equity": capped_equity,
            "raw_cash": raw_cash,
            "raw_equity": raw_equity,
            "max_capital": self.max_capital,
        }

    async def get_open_positions(self) -> dict[str, Position]:
        """Fetch open positions from exchange."""
        # If exchange supports fetch_positions (futures/margin)
        if hasattr(self.exchange, "fetch_positions"):
            try:
                raw_positions = await self.exchange.fetch_positions()
                result: dict[str, Position] = {}
                for rp in raw_positions:
                    contracts = float(rp.get("contracts") or rp.get("amount") or 0.0)
                    if contracts > 0:
                        sym = rp.get("symbol")
                        side_str = rp.get("side", "long")
                        entry_p = float(rp.get("entryPrice") or 0.0)
                        result[sym] = Position(
                            symbol=sym,
                            side=Side.LONG if side_str == "long" else Side.SHORT,
                            entry_price=entry_p,
                            quantity=contracts,
                            invalidation_price=entry_p * 0.95,
                            conviction=ConvictionTier.A,
                            risk_amount=100.0,
                            opened_at=datetime.now(UTC),
                        )
                if result:
                    self._positions = result
                    return dict(result)
            except Exception:
                pass

        # Spot balance inspection fallback
        bal = await self.exchange.fetch_balance()
        positions: dict[str, Position] = {}
        total_bal = bal.get("total", {})
        universe = self.config.universe if self.config else ["BTC/USDT", "ETH/USDT", "SOL/USDT"]

        for symbol in universe:
            base, quote = symbol.split("/")
            base_amt = float(total_bal.get(base, 0.0))
            if base_amt > 0.0001:  # Non-trivial position
                # Retain known entry price if tracked
                old_pos = self._positions.get(symbol)
                entry_p = old_pos.entry_price if old_pos else 1.0
                inv_p = old_pos.invalidation_price if old_pos else (entry_p * 0.95)
                positions[symbol] = Position(
                    symbol=symbol,
                    side=Side.LONG,
                    entry_price=entry_p,
                    quantity=base_amt,
                    invalidation_price=inv_p,
                    conviction=ConvictionTier.A,
                    risk_amount=100.0,
                    opened_at=old_pos.opened_at if old_pos else datetime.now(UTC),
                )

        self._positions = positions
        return dict(positions)

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Fetch all resting open orders from exchange."""
        if hasattr(self.exchange, "fetch_open_orders"):
            try:
                orders = await self.exchange.fetch_open_orders(symbol)
                return list(orders) if orders else []
            except Exception as exc:
                logger.warning("Failed to fetch open orders from exchange: %s", exc)
        return []

    async def cancel_all(self) -> None:
        """Cancel all resting orders and stops on exchange."""
        if hasattr(self.exchange, "cancel_all_orders"):
            with contextlib.suppress(Exception):
                await self.exchange.cancel_all_orders()
        else:
            open_orders = await self.get_open_orders()
            for ord_info in open_orders:
                with contextlib.suppress(Exception):
                    await self.exchange.cancel_order(ord_info["id"], ord_info.get("symbol"))

        self._stop_orders.clear()
        logger.info("Cancelled all open orders and cleared stop tracking.")
