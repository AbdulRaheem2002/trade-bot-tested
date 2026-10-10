"""Tests for Phase 9 live trading adapter (mocked exchange only).

Covers:
1. Protective stop placed immediately after entry
2. Stop failure immediately closes position and alerts
3. Partial fills handle actual filled qty and size stop accordingly
4. Reconciliation mismatch pauses trading and alerts
5. Live mode refuses to start without every gate
6. Idempotent retries do not double-order
7. Stop tightening syncs exchange stop
8. live.max_capital caps equity regardless of exchange balance
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.common.alerts import AlertManager
from app.common.enums import AgentState, ConvictionTier, Side, TradingMode
from app.common.errors import ExchangeError, OrderRejectedError
from app.common.models import Position
from app.config.settings import TradingConfig
from app.execution.live_broker import LiveBroker
from app.execution.live_gate import (
    DEFAULT_CONFIRMATION_PHRASE,
    LiveGatingError,
    verify_live_trading_gates,
)
from app.orchestrator import Orchestrator
from app.portfolio.db import init_portfolio_db
from app.portfolio.service import PortfolioService


class MockAsyncCCXTExchange:
    """Mock CCXT async exchange implementing order and balance APIs."""

    def __init__(self) -> None:
        self.orders: dict[str, dict[str, Any]] = {}
        self.order_counter = 100
        self.create_order_calls: list[dict[str, Any]] = []
        self.cancel_order_calls: list[tuple[str, str | None]] = []
        self.balances: dict[str, Any] = {
            "USDT": {"free": 50_000.0, "total": 50_000.0},
            "BTC": {"free": 0.0, "total": 0.0},
        }
        self.fail_stop_orders: bool = False
        self.partial_fill_qty: float | None = None

    async def create_order(
        self,
        symbol: str,
        type: str,  # noqa: A002
        side: str,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        params = params or {}
        self.order_counter += 1
        order_id = f"mock_ord_{self.order_counter}"
        call_info = {
            "order_id": order_id,
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "price": price,
            "params": params,
        }
        self.create_order_calls.append(call_info)

        # Stop order failure injection
        if "stop" in type.lower() and self.fail_stop_orders:
            raise ExchangeError("Exchange rejected protective stop order: trigger price invalid")

        # Partial fill injection
        if type == "market" and self.partial_fill_qty is not None:
            filled = self.partial_fill_qty
            status = "canceled" if filled < amount else "closed"
        else:
            filled = amount
            status = "closed" if type == "market" else "open"

        avg_price = price if price is not None else 65_000.0
        order_data = {
            "id": order_id,
            "clientOrderId": params.get("clientOrderId", ""),
            "symbol": symbol,
            "type": type,
            "side": side,
            "amount": amount,
            "filled": filled,
            "remaining": amount - filled,
            "price": avg_price,
            "average": avg_price,
            "status": status,
            "fee": {"cost": 5.0, "currency": "USDT"},
            "timestamp": int(datetime.now(UTC).timestamp() * 1000),
        }
        self.orders[order_id] = order_data
        return order_data

    async def cancel_order(self, id: str, symbol: str | None = None) -> dict[str, Any]:  # noqa: A002
        self.cancel_order_calls.append((id, symbol))
        if id in self.orders:
            self.orders[id]["status"] = "canceled"
            return self.orders[id]
        return {"id": id, "status": "canceled"}

    async def fetch_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return [o for o in self.orders.values() if o.get("status") == "open"]

    async def fetch_balance(self) -> dict[str, Any]:
        return self.balances

    async def close(self) -> None:
        pass


@pytest.fixture
def mock_exchange() -> MockAsyncCCXTExchange:
    return MockAsyncCCXTExchange()


@pytest.fixture
def live_cfg() -> TradingConfig:
    cfg = TradingConfig(mode=TradingMode.LIVE)
    cfg.live.max_capital = 15_000.0
    return cfg


# ════════════════════════════════════════════════════════════════════════
# 1. Stop Placed After Entry
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_stop_placed_after_entry(mock_exchange: MockAsyncCCXTExchange, live_cfg: TradingConfig) -> None:
    broker = LiveBroker(exchange=mock_exchange, config=live_cfg)
    await broker.start()

    fill = await broker.place_market_order(
        symbol="BTC/USDT",
        side=Side.LONG,
        qty=0.2,
        client_order_id="entry_btc_01",
        invalidation_price=62_000.0,
    )

    # 1. Entry filled correctly
    assert fill.quantity == 0.2
    assert fill.price == 65_000.0
    assert fill.symbol == "BTC/USDT"

    # 2. Exactly 2 calls made to exchange: market buy then protective stop
    calls = mock_exchange.create_order_calls
    assert len(calls) == 2
    assert calls[0]["type"] == "market"
    assert calls[0]["side"] == "buy"
    assert calls[0]["amount"] == 0.2

    # Stop order placed
    stop_call = calls[1]
    assert "stop" in stop_call["type"].lower()
    assert stop_call["side"] == "sell"
    assert stop_call["amount"] == 0.2
    assert stop_call["params"]["stopPrice"] == 62_000.0

    # 3. Stop order tracked in broker
    assert "BTC/USDT" in broker._stop_orders
    assert broker._stop_orders["BTC/USDT"]["stop_price"] == 62_000.0


# ════════════════════════════════════════════════════════════════════════
# 2. Stop Failure Closes Position and Alerts
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_stop_failure_closes_position(mock_exchange: MockAsyncCCXTExchange, live_cfg: TradingConfig) -> None:
    alert_mgr = MagicMock(spec=AlertManager)
    alert_mgr.critical = AsyncMock()

    mock_exchange.fail_stop_orders = True  # Stop order will raise ExchangeError
    broker = LiveBroker(exchange=mock_exchange, config=live_cfg, alert_manager=alert_mgr)
    await broker.start()

    # Attempt order placement
    with pytest.raises(OrderRejectedError) as exc_info:
        await broker.place_market_order(
            symbol="BTC/USDT",
            side=Side.LONG,
            qty=0.15,
            client_order_id="entry_btc_fail",
            invalidation_price=61_500.0,
        )

    assert "Protective stop placement failed" in str(exc_info.value)

    # Verify emergency close was dispatched
    calls = mock_exchange.create_order_calls
    # Call 1: market buy entry
    assert calls[0]["type"] == "market" and calls[0]["side"] == "buy"
    # Intermediary calls: protective stop attempt(s) (failed)
    assert any("stop" in c["type"].lower() for c in calls[1:-1])
    # Final call: emergency market sell
    assert calls[-1]["type"] == "market" and calls[-1]["side"] == "sell"
    assert calls[-1]["amount"] == 0.15

    # Verify critical alert was sent
    alert_mgr.critical.assert_awaited_once()


# ════════════════════════════════════════════════════════════════════════
# 3. Partial Fills
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_partial_fills_sizes_stop_to_actual_fill(
    mock_exchange: MockAsyncCCXTExchange, live_cfg: TradingConfig
) -> None:
    # Requested 1.0 BTC, but exchange only filled 0.4 BTC
    mock_exchange.partial_fill_qty = 0.4
    broker = LiveBroker(exchange=mock_exchange, config=live_cfg)
    await broker.start()

    fill = await broker.place_market_order(
        symbol="BTC/USDT",
        side=Side.LONG,
        qty=1.0,
        client_order_id="partial_entry_1",
        invalidation_price=63_000.0,
    )

    # 1. Fill records actual filled quantity
    assert fill.quantity == 0.4

    # 2. Stop order is placed for 0.4 BTC, NOT 1.0 BTC
    calls = mock_exchange.create_order_calls
    assert len(calls) == 2
    stop_call = calls[1]
    assert stop_call["amount"] == 0.4
    assert broker._stop_orders["BTC/USDT"]["qty"] == 0.4


# ════════════════════════════════════════════════════════════════════════
# 4. Reconciliation Mismatch Pauses Trading
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_reconciliation_mismatch_pauses_trading(
    tmp_path: Path, mock_exchange: MockAsyncCCXTExchange, live_cfg: TradingConfig
) -> None:
    db_url = f"sqlite:///{tmp_path / 'reconcile_test.db'}"
    engine = init_portfolio_db(db_url)
    alert_mgr = MagicMock(spec=AlertManager)
    alert_mgr.critical = AsyncMock()
    alert_mgr.warning = AsyncMock()

    broker = LiveBroker(exchange=mock_exchange, config=live_cfg, alert_manager=alert_mgr)
    service = PortfolioService(engine=engine, broker=broker, config=live_cfg)

    # Internal state tracks BTC position of 0.5
    service._positions["BTC/USDT"] = Position(
        symbol="BTC/USDT",
        side=Side.LONG,
        entry_price=64_000.0,
        quantity=0.5,
        invalidation_price=62_000.0,
        conviction=ConvictionTier.A,
        risk_amount=200.0,
        opened_at=datetime.now(UTC),
    )

    orch = Orchestrator(
        config=live_cfg,
        broker=broker,
        portfolio_service=service,
        engine=engine,
        alert_manager=alert_mgr,
    )

    # Exchange reports an unrecognized open rogue order
    mock_exchange.orders["rogue_order_999"] = {
        "id": "rogue_order_999",
        "symbol": "ETH/USDT",
        "type": "limit",
        "side": "buy",
        "amount": 10.0,
        "status": "open",
    }

    # Run reconciliation
    report = await orch.reconcile_state()

    # Invariants:
    assert report.is_matched is False
    assert orch.breaker_paused is True
    assert orch.agent_state == AgentState.PAUSED_RECONCILIATION
    assert service.agent_state == AgentState.PAUSED_RECONCILIATION.value
    alert_mgr.critical.assert_awaited()


# ════════════════════════════════════════════════════════════════════════
# 5. Live Mode Refuses to Start Without Every Gate
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_live_gating_missing_env_var(live_cfg: TradingConfig) -> None:
    with pytest.raises(LiveGatingError) as exc_info:
        await verify_live_trading_gates(
            mode=TradingMode.LIVE,
            config=live_cfg,
            api_key="real_key",
            secret="real_secret",
            env_live_enabled="false",
            confirmation_input=DEFAULT_CONFIRMATION_PHRASE,
        )
    assert "LIVE_TRADING_ENABLED" in str(exc_info.value)


@pytest.mark.asyncio
async def test_live_gating_wrong_confirmation_phrase(live_cfg: TradingConfig) -> None:
    with pytest.raises(LiveGatingError) as exc_info:
        await verify_live_trading_gates(
            mode=TradingMode.LIVE,
            config=live_cfg,
            api_key="real_key",
            secret="real_secret",
            env_live_enabled="true",
            confirmation_input="yes",
        )
    assert "Confirmation phrase mismatch" in str(exc_info.value)


@pytest.mark.asyncio
async def test_live_gating_missing_api_keys(live_cfg: TradingConfig) -> None:
    with pytest.raises(LiveGatingError) as exc_info:
        await verify_live_trading_gates(
            mode=TradingMode.LIVE,
            config=live_cfg,
            api_key="",
            secret="",
            env_live_enabled="true",
            confirmation_input=DEFAULT_CONFIRMATION_PHRASE,
        )
    assert "Missing exchange API credentials" in str(exc_info.value)


@pytest.mark.asyncio
async def test_live_gating_withdrawals_enabled_rejected(live_cfg: TradingConfig) -> None:
    mock_ex = MagicMock()
    mock_ex.fetch_api_key_permissions = AsyncMock(return_value={"enableWithdrawals": True})

    with pytest.raises(LiveGatingError) as exc_info:
        await verify_live_trading_gates(
            mode=TradingMode.LIVE,
            config=live_cfg,
            api_key="key",
            secret="sec",
            env_live_enabled="true",
            confirmation_input=DEFAULT_CONFIRMATION_PHRASE,
            exchange=mock_ex,
        )
    assert "FATAL SECURITY RISK" in str(exc_info.value)


@pytest.mark.asyncio
async def test_live_gating_all_gates_pass(live_cfg: TradingConfig) -> None:
    mock_ex = MagicMock()
    mock_ex.fetch_api_key_permissions = AsyncMock(return_value={"enableWithdrawals": False})

    # Should complete without raising any exception
    await verify_live_trading_gates(
        mode=TradingMode.LIVE,
        config=live_cfg,
        api_key="valid_api_key",
        secret="valid_secret",
        env_live_enabled="true",
        confirmation_input=DEFAULT_CONFIRMATION_PHRASE,
        exchange=mock_ex,
    )


# ════════════════════════════════════════════════════════════════════════
# 6. Idempotent Retries Do Not Double-Order
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_idempotent_retries_do_not_double_order(
    mock_exchange: MockAsyncCCXTExchange, live_cfg: TradingConfig
) -> None:
    broker = LiveBroker(exchange=mock_exchange, config=live_cfg)
    await broker.start()

    # First call places market order
    fill1 = await broker.place_market_order(
        symbol="BTC/USDT",
        side=Side.LONG,
        qty=0.1,
        client_order_id="idempotent_order_777",
        invalidation_price=63_000.0,
    )

    order_calls_before = len(mock_exchange.create_order_calls)

    # Second call with identical client_order_id
    fill2 = await broker.place_market_order(
        symbol="BTC/USDT",
        side=Side.LONG,
        qty=0.1,
        client_order_id="idempotent_order_777",
        invalidation_price=63_000.0,
    )

    # Must return identical fill object
    assert fill1 == fill2
    # Exchange MUST NOT receive duplicate order
    assert len(mock_exchange.create_order_calls) == order_calls_before


# ════════════════════════════════════════════════════════════════════════
# 7. Stop Synchronization on Tightening
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_stop_synchronization_on_tightening(
    mock_exchange: MockAsyncCCXTExchange, live_cfg: TradingConfig
) -> None:
    broker = LiveBroker(exchange=mock_exchange, config=live_cfg)
    await broker.start()

    # Entry placed with stop at 60_000.0
    await broker.place_market_order(
        symbol="BTC/USDT",
        side=Side.LONG,
        qty=0.25,
        client_order_id="entry_tighten",
        invalidation_price=60_000.0,
    )

    initial_stop_id = broker._stop_orders["BTC/USDT"]["order_id"]

    # Tighten stop to 62_500.0
    await broker.set_protective_stop("BTC/USDT", 62_500.0)

    # Verify old stop cancelled on exchange
    assert any(c[0] == initial_stop_id for c in mock_exchange.cancel_order_calls)

    # Verify new stop created with updated price
    new_stop_id = broker._stop_orders["BTC/USDT"]["order_id"]
    assert new_stop_id != initial_stop_id
    assert broker._stop_orders["BTC/USDT"]["stop_price"] == 62_500.0


# ════════════════════════════════════════════════════════════════════════
# 8. live.max_capital Caps Equity
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_live_max_capital_caps_balances(
    mock_exchange: MockAsyncCCXTExchange, live_cfg: TradingConfig
) -> None:
    # Exchange balance is $50,000 USDT
    mock_exchange.balances["USDT"]["total"] = 50_000.0
    mock_exchange.balances["USDT"]["free"] = 50_000.0

    # Config caps capital at $15,000
    live_cfg.live.max_capital = 15_000.0
    broker = LiveBroker(exchange=mock_exchange, config=live_cfg)

    balances = await broker.get_balances()

    # Reported equity & cash capped at 15,000
    assert balances["equity"] == 15_000.0
    assert balances["cash"] == 15_000.0
    assert balances["raw_equity"] == 50_000.0
    assert balances["max_capital"] == 15_000.0
