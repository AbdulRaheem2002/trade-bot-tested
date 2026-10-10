"""Abstract broker interface – Rule / Phase 3."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.common.enums import Side
from app.common.models import OrderFill, Position


class Broker(ABC):
    """Abstract broker interface.

    Both PaperBroker and future LiveBroker implement this contract.
    All methods are asynchronous, and orders must be idempotent with respect
    to client_order_id.
    """

    @abstractmethod
    async def place_market_order(
        self,
        symbol: str,
        side: Side,
        qty: float,
        client_order_id: str,
        **kwargs: Any,
    ) -> OrderFill:
        """Place an aggressive market order filled immediately at market price.

        Must be idempotent: submitting the same client_order_id twice returns
        the original OrderFill without creating a second fill.
        """
        ...

    @abstractmethod
    async def close_position(
        self,
        symbol: str,
        reason: str = "",
        client_order_id: str | None = None,
    ) -> OrderFill | None:
        """Close an existing open position in full for the given symbol.

        Returns the resulting OrderFill, or None if no position exists.
        """
        ...

    @abstractmethod
    async def get_balances(self) -> dict[str, float]:
        """Fetch current balances including cash and equity."""
        ...

    @abstractmethod
    async def get_open_positions(self) -> dict[str, Position]:
        """Fetch all open positions keyed by symbol."""
        ...

    @abstractmethod
    async def set_protective_stop(self, symbol: str, price: float) -> None:
        """Set or update a protective stop price for the symbol.

        In PaperBroker this is recorded as internal bookkeeping.
        """
        ...

    @abstractmethod
    async def cancel_all(self) -> None:
        """Cancel all resting orders or protective stops."""
        ...

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Fetch all resting or open orders."""
        return []

    async def start(self) -> None:
        """Lifecycle hook to start connections or background processes."""
        return None

    async def stop(self) -> None:
        """Lifecycle hook to gracefully stop connections or background processes."""
        return None
