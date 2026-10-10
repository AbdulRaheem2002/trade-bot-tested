"""Tests for PaperBroker simulator – Rule / Phase 3."""

from __future__ import annotations

import math

import pytest

from app.common.enums import ConvictionTier, Side
from app.common.errors import InsufficientFundsError, OrderRejectedError
from app.execution.paper_broker import PaperBroker


class TestPaperBrokerFillsAndMath:
    """Hand-checked numbers for slippage and taker fees."""

    @pytest.mark.asyncio
    async def test_hand_checked_long_fill_and_fees(self):
        """Exact hand-checked calculation for LONG buy and subsequent close.

        Setup:
            market_price = 50,000.0
            slippage_bps = 5.0  (0.0005)
            fees_taker_bps = 10.0 (0.0010)
            qty = 0.1
            initial_cash = 10,000.0

        Expected Buy:
            fill_price = 50000 * 1.0005 = 50,025.0
            notional = 50025.0 * 0.1 = 5,002.50
            fee = 5002.50 * 0.001 = 5.0025
            total_cash_spent = 5002.50 + 5.0025 = 5,007.5025
            remaining_cash = 10000 - 5007.5025 = 4,992.4975
        """
        broker = PaperBroker(
            initial_cash=10_000.0,
            slippage_bps=5.0,
            fees_taker_bps=10.0,
            shorts_enabled=False,
        )
        broker.set_market_price("BTC/USDT", 50_000.0)

        # Place BUY market order
        fill = await broker.place_market_order(
            symbol="BTC/USDT",
            side=Side.LONG,
            qty=0.1,
            client_order_id="order_buy_1",
            conviction=ConvictionTier.A,
        )

        assert math.isclose(fill.price, 50_025.0, abs_tol=1e-6)
        assert math.isclose(fill.quantity, 0.1, abs_tol=1e-6)
        assert math.isclose(fill.fee, 5.0025, abs_tol=1e-6)
        assert math.isclose(broker.cash, 4_992.4975, abs_tol=1e-6)
        assert math.isclose(broker.total_fees_paid, 5.0025, abs_tol=1e-6)

        # Verify open position
        positions = await broker.get_open_positions()
        assert "BTC/USDT" in positions
        pos = positions["BTC/USDT"]
        assert pos.side == Side.LONG
        assert math.isclose(pos.entry_price, 50_025.0, abs_tol=1e-6)
        assert math.isclose(pos.quantity, 0.1, abs_tol=1e-6)

        # Close position at market_price = 60,000.0
        # Expected Sell:
        #   fill_price = 60000 * (1 - 0.0005) = 59,970.0
        #   gross_proceeds = 59970.0 * 0.1 = 5,997.0
        #   exit_fee = 5997.0 * 0.001 = 5.9970
        #   net_proceeds = 5997.0 - 5.997 = 5,991.003
        #   realized_pnl = (5997.0 - 5002.50) - (5.0025 + 5.9970) = 994.50 - 11.0000 = 983.5005
        #   final_cash = 4992.4975 + 5991.003 = 10,983.5005
        broker.set_market_price("BTC/USDT", 60_000.0)
        close_fill = await broker.close_position("BTC/USDT", reason="take_profit")

        assert close_fill is not None
        assert math.isclose(close_fill.price, 59_970.0, abs_tol=1e-6)
        assert math.isclose(close_fill.fee, 5.9970, abs_tol=1e-6)
        assert math.isclose(broker.cash, 10_983.5005, abs_tol=1e-6)
        assert math.isclose(broker.realized_pnl, 983.5005, abs_tol=1e-6)
        assert math.isclose(broker.total_fees_paid, 10.9995, abs_tol=1e-6)

        # Positions should now be empty
        assert len(await broker.get_open_positions()) == 0


class TestPaperBrokerConstraints:
    """Lot size, min notional, and cash bounds."""

    @pytest.mark.asyncio
    async def test_cannot_spend_more_than_available_cash(self):
        """Order requiring more cash than available is rejected."""
        broker = PaperBroker(initial_cash=500.0)
        broker.set_market_price("BTC/USDT", 50_000.0)

        # 0.1 BTC requires ~5,007 USDT, but cash is only 500
        with pytest.raises(InsufficientFundsError):
            await broker.place_market_order(
                symbol="BTC/USDT",
                side=Side.LONG,
                qty=0.1,
                client_order_id="too_big",
            )
        assert broker.cash == 500.0

    @pytest.mark.asyncio
    async def test_lot_size_rounding(self):
        """Quantity is floored to lot size step."""
        broker = PaperBroker(initial_cash=10_000.0)
        broker.set_market_price("BTC/USDT", 50_000.0)
        broker.set_market_rules("BTC/USDT", lot_size=0.01, min_notional=10.0)

        # Qty 0.057 should round to 0.05
        fill = await broker.place_market_order(
            symbol="BTC/USDT",
            side=Side.LONG,
            qty=0.057,
            client_order_id="lot_test_1",
        )
        assert math.isclose(fill.quantity, 0.05, abs_tol=1e-6)

    @pytest.mark.asyncio
    async def test_lot_size_rounds_to_zero_rejected(self):
        """Quantity smaller than lot size step is rejected."""
        broker = PaperBroker(initial_cash=10_000.0)
        broker.set_market_price("BTC/USDT", 50_000.0)
        broker.set_market_rules("BTC/USDT", lot_size=0.1, min_notional=10.0)

        with pytest.raises(OrderRejectedError):
            await broker.place_market_order(
                symbol="BTC/USDT",
                side=Side.LONG,
                qty=0.05,
                client_order_id="too_small_lot",
            )

    @pytest.mark.asyncio
    async def test_min_notional_rejection(self):
        """Order whose notional is below min_notional is rejected."""
        broker = PaperBroker(initial_cash=10_000.0)
        broker.set_market_price("SOL/USDT", 20.0)
        broker.set_market_rules("SOL/USDT", lot_size=0.1, min_notional=50.0)

        # 1.0 SOL * 20.01 fill price = ~20.01 USDT < 50.0 min_notional
        with pytest.raises(OrderRejectedError):
            await broker.place_market_order(
                symbol="SOL/USDT",
                side=Side.LONG,
                qty=1.0,
                client_order_id="sub_notional",
            )


class TestPaperBrokerIdempotencyAndStops:
    """Idempotency, protective stops, and short selling."""

    @pytest.mark.asyncio
    async def test_submitting_same_client_order_id_is_idempotent(self):
        """Submitting same client_order_id returns existing fill without duplicate spend."""
        broker = PaperBroker(initial_cash=10_000.0)
        broker.set_market_price("ETH/USDT", 3_000.0)

        fill1 = await broker.place_market_order(
            symbol="ETH/USDT",
            side=Side.LONG,
            qty=1.0,
            client_order_id="idemp_1",
        )
        cash_after_1 = broker.cash

        fill2 = await broker.place_market_order(
            symbol="ETH/USDT",
            side=Side.LONG,
            qty=1.0,
            client_order_id="idemp_1",
        )

        assert fill1.fill_id == fill2.fill_id
        assert broker.cash == cash_after_1
        positions = await broker.get_open_positions()
        assert math.isclose(positions["ETH/USDT"].quantity, 1.0, abs_tol=1e-6)

    @pytest.mark.asyncio
    async def test_protective_stops(self):
        """Setting and cancelling protective stops."""
        broker = PaperBroker()
        await broker.set_protective_stop("BTC/USDT", 48_000.0)
        assert broker.get_protective_stop("BTC/USDT") == 48_000.0

        await broker.cancel_all()
        assert broker.get_protective_stop("BTC/USDT") is None

    @pytest.mark.asyncio
    async def test_shorts_disabled_by_default(self):
        """Opening a short when shorts_enabled=False raises OrderRejectedError."""
        broker = PaperBroker(shorts_enabled=False)
        broker.set_market_price("BTC/USDT", 50_000.0)

        with pytest.raises(OrderRejectedError, match="Short selling is disabled"):
            await broker.place_market_order(
                symbol="BTC/USDT",
                side=Side.SHORT,
                qty=0.1,
                client_order_id="short_1",
            )

    @pytest.mark.asyncio
    async def test_short_with_funding_rate(self):
        """Short position accrues funding fee."""
        broker = PaperBroker(
            initial_cash=10_000.0,
            shorts_enabled=True,
            funding_rate_8h=0.001,  # 0.1% per 8h
        )
        broker.set_market_price("BTC/USDT", 50_000.0)

        await broker.place_market_order(
            symbol="BTC/USDT",
            side=Side.SHORT,
            qty=0.1,
            client_order_id="short_order_1",
        )
        cash_before_funding = broker.cash

        # Apply 8 hours of funding on ~5000 notional: 5000 * 0.001 = 5.0
        funding_fee = broker.apply_funding(elapsed_hours=8.0)
        assert funding_fee > 0
        assert math.isclose(broker.cash, cash_before_funding - funding_fee, abs_tol=1e-4)
