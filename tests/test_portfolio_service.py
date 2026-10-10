"""Tests for PortfolioService single-writer, persistence, and reconciliation – Rule / Phase 3."""

from __future__ import annotations

import asyncio
import math
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine

from app.common.enums import ConvictionTier, Side
from app.common.errors import ReconciliationError
from app.common.models import OrderFill
from app.config.settings import TradingConfig
from app.execution.paper_broker import PaperBroker
from app.portfolio.service import PortfolioService


class TestPortfolioServiceSingleWriter:
    """Verify single-writer serialization with concurrent tasks."""

    @pytest.mark.asyncio
    async def test_concurrent_mutations_are_serialized(self):
        """Many concurrent tasks mutate state simultaneously; final state is strictly consistent."""
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        broker = PaperBroker(initial_cash=10_000.0)
        config = TradingConfig(starting_equity=10_000.0)

        async with PortfolioService(engine, broker, config) as service:
            # Concurrently update cash and record snapshots
            async def worker(task_id: int):
                fill = OrderFill(
                    fill_id=f"f_{task_id}",
                    client_order_id=f"c_{task_id}",
                    symbol="BTC/USDT",
                    side=Side.LONG,
                    price=50_000.0,
                    quantity=0.001,
                    fee=0.05,
                    slippage_bps=5.0,
                    timestamp=datetime.now(UTC),
                )
                await service.apply_open_fill(
                    fill=fill,
                    invalidation_price=48_000.0,
                    conviction=ConvictionTier.A,
                    risk_amount=2.0,
                )
                await service.record_equity_snapshot(
                    prices={"BTC/USDT": 50_000.0},
                    timestamp=datetime.now(UTC),
                )

            # Spawn 30 concurrent tasks
            tasks = [asyncio.create_task(worker(i)) for i in range(30)]
            await asyncio.gather(*tasks)

            state = service.get_state(prices={"BTC/USDT": 50_000.0})
            # Verify position quantity accumulated cleanly: 30 * 0.001 = 0.03
            assert "BTC/USDT" in state.positions
            assert math.isclose(state.positions["BTC/USDT"].quantity, 0.03, abs_tol=1e-6)
            # Cash deducted: 30 * ((50000 * 0.001) + 0.05) = 30 * 50.05 = 1501.50
            # 10000 - 1501.50 = 8498.50
            assert math.isclose(state.cash, 8498.50, abs_tol=1e-4)
            # Exactly 30 equity history points
            assert len(state.equity_history) == 30


class TestPortfolioServicePersistenceAndRecovery:
    """Restart recovery: persist state to database, reload, and verify identical state."""

    @pytest.mark.asyncio
    async def test_restart_recovery_identical_state(self, tmp_path):
        """State persisted in DB is restored with full fidelity upon restart."""
        db_path = tmp_path / "test_persistence.db"
        db_url = f"sqlite:///{db_path}"

        broker = PaperBroker(initial_cash=10_000.0)
        config = TradingConfig(starting_equity=10_000.0)

        # 1. Start original service, perform mutations
        async with PortfolioService(db_url, broker, config) as service:
            fill_btc = OrderFill(
                fill_id="fill_rec_1",
                client_order_id="order_rec_1",
                symbol="BTC/USDT",
                side=Side.LONG,
                price=50_000.0,
                quantity=0.1,
                fee=5.0,
                slippage_bps=5.0,
                timestamp=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
            )
            await service.apply_open_fill(
                fill=fill_btc,
                invalidation_price=47_000.0,
                conviction=ConvictionTier.A,
                risk_amount=300.0,
            )

            fill_eth = OrderFill(
                fill_id="fill_rec_2",
                client_order_id="order_rec_2",
                symbol="ETH/USDT",
                side=Side.LONG,
                price=3_000.0,
                quantity=1.0,
                fee=3.0,
                slippage_bps=5.0,
                timestamp=datetime(2026, 6, 1, 13, 0, tzinfo=UTC),
            )
            await service.apply_open_fill(
                fill=fill_eth,
                invalidation_price=2_800.0,
                conviction=ConvictionTier.B,
                risk_amount=200.0,
            )

            await service.record_equity_snapshot(
                prices={"BTC/USDT": 52_000.0, "ETH/USDT": 3_100.0},
                timestamp=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
            )

            state_before = service.get_state({"BTC/USDT": 52_000.0, "ETH/USDT": 3_100.0})

        # 2. Restart: Rebuild new PortfolioService from DB without in-memory state
        recovered = PortfolioService.load_from_db(db_url, broker, config)
        async with recovered:
            state_after = recovered.get_state({"BTC/USDT": 52_000.0, "ETH/USDT": 3_100.0})

            # Check cash equality
            assert math.isclose(state_before.cash, state_after.cash, abs_tol=1e-6)
            # Check equity equality
            assert math.isclose(state_before.equity, state_after.equity, abs_tol=1e-6)
            # Check peak equity
            assert math.isclose(state_before.peak_equity, state_after.peak_equity, abs_tol=1e-6)

            # Check positions
            assert set(state_before.positions.keys()) == set(state_after.positions.keys())
            for sym in state_before.positions:
                pos_b = state_before.positions[sym]
                pos_a = state_after.positions[sym]
                assert pos_b.symbol == pos_a.symbol
                assert pos_b.side == pos_a.side
                assert math.isclose(pos_b.entry_price, pos_a.entry_price, abs_tol=1e-6)
                assert math.isclose(pos_b.quantity, pos_a.quantity, abs_tol=1e-6)
                assert math.isclose(
                    pos_b.invalidation_price, pos_a.invalidation_price, abs_tol=1e-6
                )
                assert pos_b.conviction == pos_a.conviction
                assert math.isclose(pos_b.risk_amount, pos_a.risk_amount, abs_tol=1e-6)

            # Check equity history length
            assert len(state_before.equity_history) == len(state_after.equity_history)


class TestPortfolioServiceReconcile:
    """Test state reconciliation against broker."""

    @pytest.mark.asyncio
    async def test_reconcile_success_with_paper_broker(self):
        """Reconciliation succeeds when internal state matches broker."""
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        broker = PaperBroker(initial_cash=10_000.0)
        broker.set_market_price("BTC/USDT", 50_000.0)

        # Place trade on broker
        fill = await broker.place_market_order(
            symbol="BTC/USDT",
            side=Side.LONG,
            qty=0.1,
            client_order_id="trade_recon_1",
            invalidation_price=48_000.0,
            conviction=ConvictionTier.A,
        )

        async with PortfolioService(engine, broker, initial_cash=10_000.0) as service:
            # Sync fill into service
            await service.apply_open_fill(
                fill=fill,
                invalidation_price=48_000.0,
                conviction=ConvictionTier.A,
                risk_amount=200.0,
            )

            report = await service.reconcile(strict=True)
            assert report.is_matched is True
            assert len(report.mismatches) == 0

    @pytest.mark.asyncio
    async def test_reconcile_detects_discrepancy(self):
        """Reconciliation flags missing or mismatched positions and raises ReconciliationError."""
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        broker = PaperBroker(initial_cash=10_000.0)
        broker.set_market_price("BTC/USDT", 50_000.0)

        async with PortfolioService(engine, broker, initial_cash=10_000.0) as service:
            # Create internal position that broker does not have
            dummy_fill = OrderFill(
                fill_id="ghost_fill",
                client_order_id="ghost_order",
                symbol="BTC/USDT",
                side=Side.LONG,
                price=50_000.0,
                quantity=0.1,
                fee=5.0,
                slippage_bps=5.0,
                timestamp=datetime.now(UTC),
            )
            await service.apply_open_fill(
                fill=dummy_fill,
                invalidation_price=48_000.0,
                conviction=ConvictionTier.A,
                risk_amount=200.0,
            )

            # Broker does not have this position -> reconciliation must fail
            with pytest.raises(ReconciliationError, match="(?i)reconciliation failed"):
                await service.reconcile(strict=True)

            report = await service.reconcile(strict=False)
            assert report.is_matched is False
            assert any("BTC/USDT" in m for m in report.mismatches)


class TestPortfolioServiceComputations:
    """Equity, exposure, and unrealized R calculations."""

    @pytest.mark.asyncio
    async def test_computations_equity_exposure_unrealized_r(self):
        """Verify calculations of equity, exposure, and per-position unrealized R."""
        engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        broker = PaperBroker(initial_cash=10_000.0)

        async with PortfolioService(engine, broker, initial_cash=10_000.0) as service:
            fill = OrderFill(
                fill_id="f_calc",
                client_order_id="c_calc",
                symbol="BTC/USDT",
                side=Side.LONG,
                price=50_000.0,
                quantity=0.1,
                fee=5.0,
                slippage_bps=5.0,
                timestamp=datetime.now(UTC),
            )
            await service.apply_open_fill(
                fill=fill,
                invalidation_price=48_000.0,
                conviction=ConvictionTier.A,
                risk_amount=200.0,  # 1 R = 200 USDT
            )

            # At entry price 50,000:
            # cash = 10000 - 5005 = 4995.0
            # equity = 4995.0 + (0.1 * 50000) = 9995.0
            # exposure = 5000.0
            assert math.isclose(
                service.compute_equity({"BTC/USDT": 50_000.0}), 9995.0, abs_tol=1e-4
            )
            assert math.isclose(
                service.compute_exposure({"BTC/USDT": 50_000.0}), 5000.0, abs_tol=1e-4
            )
            assert math.isclose(
                service.compute_unrealized_r("BTC/USDT", 50_000.0), 0.0, abs_tol=1e-4
            )

            # At price 52,000 (+2000 per BTC -> +200 USDT unrealized -> +1.0 R):
            # equity = 4995 + 5200 = 10195.0
            assert math.isclose(
                service.compute_equity({"BTC/USDT": 52_000.0}), 10_195.0, abs_tol=1e-4
            )
            assert math.isclose(
                service.compute_unrealized_r("BTC/USDT", 52_000.0), 1.0, abs_tol=1e-4
            )

            # Close position
            close_fill = OrderFill(
                fill_id="f_close",
                client_order_id="c_close",
                symbol="BTC/USDT",
                side=Side.SHORT,
                price=52_000.0,
                quantity=0.1,
                fee=5.20,
                slippage_bps=5.0,
                timestamp=datetime.now(UTC),
            )
            trade = await service.apply_close_fill(close_fill, reason="target_hit")
            assert trade is not None
            # Realized PnL = (5200 - 5000) - (5.0 + 5.20) = 200 - 10.20 = 189.80
            assert math.isclose(trade.realized_pnl, 189.80, abs_tol=1e-4)
            # R multiple = 189.80 / 200.0 = 0.949
            assert math.isclose(trade.r_multiple, 0.949, abs_tol=1e-4)
            assert len(service.get_state().positions) == 0
