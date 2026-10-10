"""Realistic paper trading broker simulator – Rule / Phase 3.

Simulates market order fills against real or injected market prices:
- Market orders fill at current price adjusted by slippage_bps against the trader, plus taker fees.
- Rounds quantity to lot size; rejects orders below min notional.
- Tracks cash, positions (long, and short behind shorts_enabled with funding rate model),
  realized and unrealized PnL.
- Never allows negative cash or leverage in v1.
- All order submissions are idempotent by client_order_id.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from app.common.enums import ConvictionTier, Side
from app.common.errors import (
    InsufficientFundsError,
    OrderRejectedError,
)
from app.common.models import OrderFill, Position
from app.data.exchange_client import ExchangeClient
from app.execution.base import Broker


class PaperBroker(Broker):
    """Realistic Paper Trading Broker implementation."""

    def __init__(
        self,
        initial_cash: float = 10_000.0,
        slippage_bps: float = 5.0,
        fees_taker_bps: float = 10.0,
        shorts_enabled: bool = False,
        funding_rate_8h: float = 0.0001,
        exchange_client: ExchangeClient | None = None,
        price_feed: Callable[[str], float] | None = None,
    ) -> None:
        self.cash: float = float(initial_cash)
        self.initial_cash: float = float(initial_cash)
        self.slippage_bps: float = float(slippage_bps)
        self.fees_taker_bps: float = float(fees_taker_bps)
        self.shorts_enabled: bool = bool(shorts_enabled)
        self.funding_rate_8h: float = float(funding_rate_8h)
        self.exchange_client: ExchangeClient | None = exchange_client
        self.price_feed: Callable[[str], float] | None = price_feed

        # State collections
        self._positions: dict[str, Position] = {}
        self._protective_stops: dict[str, float] = {}
        self._fills_by_client_id: dict[str, OrderFill] = {}
        self._market_prices: dict[str, float] = {}
        self._market_rules: dict[
            str, dict[str, float]
        ] = {}  # {symbol: {"lot_size": ..., "min_notional": ...}}
        self._entry_fees: dict[str, float] = {}  # {symbol: entry_fee_paid}
        self.realized_pnl: float = 0.0
        self.total_fees_paid: float = 0.0
        self.total_funding_paid: float = 0.0

    def set_market_price(self, symbol: str, price: float) -> None:
        """Inject or override current market price for testing or simulated execution."""
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        self._market_prices[symbol] = float(price)

    def set_market_rules(
        self,
        symbol: str,
        lot_size: float = 0.001,
        min_notional: float = 10.0,
    ) -> None:
        """Set lot size and min notional for symbol."""
        self._market_rules[symbol] = {
            "lot_size": float(lot_size),
            "min_notional": float(min_notional),
        }

    def get_lot_size(self, symbol: str) -> float:
        """Get lot size for symbol (defaults to 0.001)."""
        rules = self._market_rules.get(symbol)
        if rules and "lot_size" in rules:
            return rules["lot_size"]
        return 0.001

    def get_min_notional(self, symbol: str) -> float:
        """Get min notional for symbol in quote currency (defaults to 10.0)."""
        rules = self._market_rules.get(symbol)
        if rules and "min_notional" in rules:
            return rules["min_notional"]
        return 10.0

    async def get_market_price(self, symbol: str) -> float:
        """Resolve current market price from price cache, price feed callable, or exchange client."""
        if symbol in self._market_prices:
            return self._market_prices[symbol]

        if self.price_feed is not None:
            price = self.price_feed(symbol)
            if price > 0:
                self._market_prices[symbol] = price
                return price

        if self.exchange_client is not None:
            ticker = await self.exchange_client.fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close")
            if price and price > 0:
                self._market_prices[symbol] = float(price)
                return float(price)

        raise OrderRejectedError(f"No market price available for {symbol}")

    def round_to_lot_size(self, symbol: str, qty: float) -> float:
        """Round quantity down to nearest valid lot size."""
        lot_size = self.get_lot_size(symbol)
        if lot_size <= 0:
            return qty
        # Calculate step count using high precision to avoid floating point artifacts
        steps = math.floor(round(qty / lot_size, 9))
        rounded = steps * lot_size
        # Format to lot_size decimal places
        decimals = max(0, -int(math.floor(math.log10(lot_size)))) if lot_size < 1 else 0
        return round(rounded, decimals)

    async def place_market_order(
        self,
        symbol: str,
        side: Side,
        qty: float,
        client_order_id: str,
        invalidation_price: float | None = None,
        conviction: ConvictionTier | None = None,
        risk_amount: float | None = None,
        timestamp: datetime | None = None,
        **kwargs: Any,
    ) -> OrderFill:
        """Place an aggressive market order with slippage and fees.

        Idempotent: Re-submitting the same client_order_id returns the existing fill.
        """
        # Idempotency check
        if client_order_id in self._fills_by_client_id:
            return self._fills_by_client_id[client_order_id]

        if qty <= 0:
            raise OrderRejectedError(f"Order quantity must be positive, got {qty}")

        if side == Side.SHORT and not self.shorts_enabled:
            raise OrderRejectedError("Short selling is disabled in configuration")

        now = timestamp or datetime.now(UTC)
        market_price = await self.get_market_price(symbol)

        # 1. Round quantity to lot size
        rounded_qty = self.round_to_lot_size(symbol, qty)
        if rounded_qty <= 0:
            raise OrderRejectedError(
                f"Quantity {qty} rounds to 0 for lot size {self.get_lot_size(symbol)}"
            )

        # 2. Compute fill price with slippage (slippage always against the trader)
        slip_rate = self.slippage_bps / 10_000.0
        if side == Side.LONG:
            fill_price = market_price * (1.0 + slip_rate)
        else:
            fill_price = market_price * (1.0 - slip_rate)

        # 3. Min notional check
        notional = rounded_qty * fill_price
        min_notional = self.get_min_notional(symbol)
        if notional < min_notional:
            raise OrderRejectedError(
                f"Order notional {notional:.2f} is below min_notional {min_notional:.2f}"
            )

        # 4. Taker fee calculation
        fee = notional * (self.fees_taker_bps / 10_000.0)

        # 5. Cash check (no negative cash, no leverage in v1)
        required_cash = notional + fee
        if required_cash > self.cash:
            raise InsufficientFundsError(
                f"Insufficient cash: required {required_cash:.2f}, available {self.cash:.2f}"
            )

        # Deduct cash
        self.cash -= required_cash
        self.total_fees_paid += fee

        # 6. Book position
        inv_price = invalidation_price or (
            fill_price * 0.95 if side == Side.LONG else fill_price * 1.05
        )
        c_tier = conviction or ConvictionTier.B
        r_amt = risk_amount or (abs(fill_price - inv_price) * rounded_qty)

        if symbol in self._positions:
            # Adding to existing position (must be same side)
            existing = self._positions[symbol]
            if existing.side != side:
                raise OrderRejectedError(
                    f"Cannot add opposite side order ({side}) to existing {existing.side} position"
                )
            new_qty = existing.quantity + rounded_qty
            avg_entry = (
                (existing.entry_price * existing.quantity) + (fill_price * rounded_qty)
            ) / new_qty
            new_risk = existing.risk_amount + r_amt
            self._positions[symbol] = Position(
                symbol=symbol,
                side=side,
                entry_price=avg_entry,
                quantity=new_qty,
                invalidation_price=existing.invalidation_price,
                conviction=existing.conviction,
                risk_amount=new_risk,
                opened_at=existing.opened_at,
            )
            self._entry_fees[symbol] = self._entry_fees.get(symbol, 0.0) + fee
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                side=side,
                entry_price=fill_price,
                quantity=rounded_qty,
                invalidation_price=inv_price,
                conviction=c_tier,
                risk_amount=r_amt,
                opened_at=now,
            )
            self._entry_fees[symbol] = fee

        # 7. Create Fill record
        fill = OrderFill(
            fill_id=f"fill_{uuid.uuid4().hex[:12]}",
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            price=fill_price,
            quantity=rounded_qty,
            fee=fee,
            slippage_bps=self.slippage_bps,
            timestamp=now,
        )
        self._fills_by_client_id[client_order_id] = fill
        return fill

    async def close_position(
        self,
        symbol: str,
        reason: str = "",
        client_order_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> OrderFill | None:
        """Close an existing position in full.

        Returns OrderFill or None if no position exists.
        Idempotent if client_order_id provided.
        """
        cid = client_order_id or f"close_{symbol}_{uuid.uuid4().hex[:8]}"
        if cid in self._fills_by_client_id:
            return self._fills_by_client_id[cid]

        if symbol not in self._positions:
            return None

        pos = self._positions[symbol]
        now = timestamp or datetime.now(UTC)
        market_price = await self.get_market_price(symbol)
        slip_rate = self.slippage_bps / 10_000.0

        # Closing a LONG means selling; closing a SHORT means buying to cover
        if pos.side == Side.LONG:
            close_side = Side.SHORT
            fill_price = market_price * (1.0 - slip_rate)
            gross_proceeds = fill_price * pos.quantity
            fee = gross_proceeds * (self.fees_taker_bps / 10_000.0)
            net_proceeds = gross_proceeds - fee
            entry_cost = pos.entry_price * pos.quantity
            entry_fee = self._entry_fees.get(symbol, 0.0)
            trade_pnl = (gross_proceeds - entry_cost) - (entry_fee + fee)

            # Return net cash proceeds
            self.cash += net_proceeds
        else:
            close_side = Side.LONG
            fill_price = market_price * (1.0 + slip_rate)
            cover_cost = fill_price * pos.quantity
            fee = cover_cost * (self.fees_taker_bps / 10_000.0)
            entry_collateral = pos.entry_price * pos.quantity
            entry_fee = self._entry_fees.get(symbol, 0.0)
            trade_pnl = ((pos.entry_price - fill_price) * pos.quantity) - (entry_fee + fee)

            # Return collateral + pnl
            self.cash += entry_collateral + (pos.entry_price - fill_price) * pos.quantity - fee

        self.total_fees_paid += fee
        self.realized_pnl += trade_pnl

        # Cleanup internal stores
        del self._positions[symbol]
        self._protective_stops.pop(symbol, None)
        self._entry_fees.pop(symbol, None)

        fill = OrderFill(
            fill_id=f"fill_{uuid.uuid4().hex[:12]}",
            client_order_id=cid,
            symbol=symbol,
            side=close_side,
            price=fill_price,
            quantity=pos.quantity,
            fee=fee,
            slippage_bps=self.slippage_bps,
            timestamp=now,
        )
        self._fills_by_client_id[cid] = fill
        return fill

    def apply_funding(self, elapsed_hours: float) -> float:
        """Apply funding rate to open short positions."""
        if elapsed_hours <= 0:
            return 0.0
        total_funding = 0.0
        for pos in self._positions.values():
            if pos.side == Side.SHORT:
                funding_fee = pos.notional * self.funding_rate_8h * (elapsed_hours / 8.0)
                self.cash = max(0.0, self.cash - funding_fee)
                total_funding += funding_fee
        self.total_funding_paid += total_funding
        self.total_fees_paid += total_funding
        return total_funding

    async def get_balances(self) -> dict[str, float]:
        """Fetch current balances including cash and equity."""
        positions = self._positions
        unrealized = 0.0
        for sym, pos in positions.items():
            try:
                price = await self.get_market_price(sym)
                unrealized += pos.unrealized_pnl(price)
            except Exception:
                unrealized += pos.unrealized_pnl(pos.entry_price)

        # In spot / 100% collateralized, equity = cash + sum(pos.notional + pnl)
        # For open longs: cash was deducted (entry_price * qty + fee). So equity = cash + current_value.
        long_value = sum(
            pos.quantity * self._market_prices.get(sym, pos.entry_price)
            for sym, pos in positions.items()
            if pos.side == Side.LONG
        )
        # For open shorts: collateral was deducted. Value = collateral + unrealized_pnl
        short_value = sum(
            pos.quantity * pos.entry_price
            + pos.unrealized_pnl(self._market_prices.get(sym, pos.entry_price))
            for sym, pos in positions.items()
            if pos.side == Side.SHORT
        )
        equity = self.cash + long_value + short_value

        return {
            "cash": self.cash,
            "equity": equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized,
            "total_fees_paid": self.total_fees_paid,
        }

    def get_positions(self) -> dict[str, Position]:
        """Synchronous helper to get current open positions."""
        return dict(self._positions)

    async def get_open_positions(self) -> dict[str, Position]:
        """Fetch all open positions."""
        return dict(self._positions)

    async def set_protective_stop(self, symbol: str, price: float) -> None:
        """Set or update internal protective stop price for symbol."""
        if price <= 0:
            raise ValueError(f"Stop price must be positive, got {price}")
        self._protective_stops[symbol] = float(price)

    def get_protective_stop(self, symbol: str) -> float | None:
        """Retrieve protective stop level if set."""
        return self._protective_stops.get(symbol)

    async def cancel_all(self) -> None:
        """Cancel all resting orders / stops."""
        self._protective_stops.clear()
