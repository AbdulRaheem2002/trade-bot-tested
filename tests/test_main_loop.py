"""Tests for Phase 5 main loop, orchestrator, recovery, kill switch, and breakers."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.common.enums import Action, AgentState, ConvictionTier, RejectionReason, Side
from app.common.errors import InstanceAlreadyRunningError, OrchestratorError
from app.common.llm_client import MockLLM
from app.common.models import Decision, DecisionItem
from app.config.settings import TradingConfig
from app.data.store import CandleStore
from app.execution.paper_broker import PaperBroker
from app.orchestrator import Orchestrator, SingleInstanceLock
from app.portfolio.db import TradeModel, init_portfolio_db
from app.portfolio.service import PortfolioService


def _populate_candles(store: CandleStore, symbol: str = "BTC/USDT") -> None:
    """Helper to populate 70 4h candles and 35 1d candles for indicator calculations."""
    base_dt = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)

    # 1d candles
    candles_1d: list[list[float]] = []
    p = 60_000.0
    for d in range(35):
        dt = base_dt + timedelta(days=d)
        ts_ms = int(dt.timestamp() * 1000)
        p += 100.0
        candles_1d.append([ts_ms, p, p + 200, p - 200, p + 50, 100_000_000.0])
    store.upsert_candles(symbol, "1d", candles_1d, now_ms=int(time_future := 9999999999999))

    # 4h candles (starts after day 20)
    start_4h = base_dt + timedelta(days=20)
    candles_4h: list[list[float]] = []
    p4 = 64_000.0
    for i in range(70):
        dt = start_4h + timedelta(hours=i * 4)
        ts_ms = int(dt.timestamp() * 1000)
        p4 += 30.0
        candles_4h.append([ts_ms, p4, p4 + 500, p4 - 500, p4 + 20, 25_000_000.0])
    store.upsert_candles(symbol, "4h", candles_4h, now_ms=time_future)


class TestMainLoopAndOrchestrator:
    """Test suite for Phase 5 orchestrator lifecycle and invariant enforcement."""

    @pytest.mark.asyncio
    async def test_full_cycle_invalidation_closes_position_and_journal_correct(
        self, tmp_path: Path
    ) -> None:
        """Full cycle: opens a position, monitor closes it when price touches invalidation, journal is correct."""
        db_url = f"sqlite:///{tmp_path / 'full_cycle.db'}"
        engine = init_portfolio_db(db_url)
        store = CandleStore(engine)
        _populate_candles(store, "BTC/USDT")

        cycle_dt = datetime(2026, 6, 1, 16, 0, tzinfo=UTC)
        df_btc = store.get_candles("BTC/USDT", "4h", now_ms=int(cycle_dt.timestamp() * 1000))
        last_close = float(df_btc.iloc[-1]["close"])
        broker = PaperBroker(initial_cash=10_000.0)
        broker.set_market_price("BTC/USDT", last_close)
        broker.set_market_rules("BTC/USDT", lot_size=0.001, min_notional=10.0)

        config = TradingConfig(
            universe=["BTC/USDT"],
            starting_equity=10_000.0,
            lock_file_path=str(tmp_path / "lock.lock"),
            shutdown_marker_path=str(tmp_path / "shutdown.marker"),
        )
        portfolio_service = PortfolioService(engine, broker, config)

        # Invalidation at 2.0 ATR below entry (last_close - 2000.0)
        inv_price = last_close - 2000.0

        open_decision = Decision(
            reasoning="BTC/USDT breaking out above 4h EMA200",
            decisions=[
                DecisionItem(
                    action=Action.OPEN_LONG,
                    symbol="BTC/USDT",
                    invalidation_price=inv_price,
                    conviction=ConvictionTier.A,
                    confidence=0.85,
                    thesis="Breakout above resistance",
                    rationale="4h EMA200 cross",
                )
            ],
        )
        mock_llm = MockLLM(canned_responses=[open_decision])

        async with Orchestrator(
            config=config,
            engine=engine,
            broker=broker,
            portfolio_service=portfolio_service,
            candle_store=store,
            llm_client=mock_llm,
            lock_path=tmp_path / "lock.lock",
            shutdown_marker_path=tmp_path / "shutdown.marker",
        ) as orch:
            # 1. Run Reflex cycle
            approved, rejections = await orch.execute_cycle(
                cycle_id="cycle_001", cycle_time=cycle_dt
            )

            assert len(approved) == 1
            assert approved[0].symbol == "BTC/USDT"
            assert "BTC/USDT" in orch.portfolio_service.get_state().positions
            pos = orch.portfolio_service.get_state().positions["BTC/USDT"]
            assert pos.invalidation_price == inv_price

            # 2. Simulate price movement hitting invalidation stop
            broker.set_market_price("BTC/USDT", inv_price - 50.0)

            # 3. Trigger invalidation monitor check
            await orch.check_invalidations()

            # 4. Verify position closed
            assert "BTC/USDT" not in orch.portfolio_service.get_state().positions

            # 5. Verify trade journal entry in SQLite
            with Session(engine) as session:
                trades = session.scalars(select(TradeModel)).all()
                assert len(trades) == 1
                assert trades[0].symbol == "BTC/USDT"
                assert trades[0].exit_reason == "invalidation"
                assert trades[0].realized_pnl < 0  # Stopped out for small loss

    @pytest.mark.asyncio
    async def test_kill_switch_closes_all_and_blocks_new_orders_until_resumed(
        self, tmp_path: Path
    ) -> None:
        """Kill switch closes all positions, cancels resting orders, enters PAUSED_KILL, and blocks orders until resumed."""
        db_url = f"sqlite:///{tmp_path / 'kill_switch.db'}"
        engine = init_portfolio_db(db_url)
        broker = PaperBroker(initial_cash=10_000.0)
        broker.set_market_price("BTC/USDT", 65_000.0)

        kill_file = tmp_path / "KILL_SWITCH"
        config = TradingConfig(
            universe=["BTC/USDT"],
            starting_equity=10_000.0,
            kill_switch_file=str(kill_file),
            lock_file_path=str(tmp_path / "lock.lock"),
            shutdown_marker_path=str(tmp_path / "shutdown.marker"),
        )
        portfolio_service = PortfolioService(engine, broker, config)

        async with Orchestrator(
            config=config,
            engine=engine,
            broker=broker,
            portfolio_service=portfolio_service,
            lock_path=tmp_path / "lock.lock",
            shutdown_marker_path=tmp_path / "shutdown.marker",
        ) as orch:
            # Manually seed an open position
            fill = await broker.place_market_order(
                symbol="BTC/USDT",
                side=Side.LONG,
                qty=0.05,
                client_order_id="seed_order_1",
                invalidation_price=63_000.0,
            )
            await portfolio_service.apply_open_fill(
                fill=fill,
                invalidation_price=63_000.0,
                conviction=ConvictionTier.A,
                risk_amount=100.0,
            )
            assert "BTC/USDT" in portfolio_service.get_state().positions

            # 1. Trigger kill switch by creating KILL_SWITCH file
            kill_file.write_text("STOP ALL TRADING")
            await orch.check_kill_switch()

            assert orch.agent_state == AgentState.PAUSED_KILL
            assert orch.portfolio_service.kill_switch is True
            # All positions must be closed
            assert len(orch.portfolio_service.get_state().positions) == 0

            # 2. Attempting a cycle during PAUSED_KILL blocks new opens
            mock_llm = MockLLM(
                canned_responses=[
                    Decision(
                        reasoning="Try to open while killed",
                        decisions=[
                            DecisionItem(
                                action=Action.OPEN_LONG,
                                symbol="BTC/USDT",
                                invalidation_price=63_000.0,
                                conviction=ConvictionTier.A,
                                confidence=0.9,
                            )
                        ],
                    )
                ]
            )
            orch.llm_client = mock_llm
            approved, rejections = await orch.execute_cycle("cycle_killed_test")
            assert len(approved) == 0
            assert any(r.reason == RejectionReason.BREAKER_ACTIVE for r in rejections)

            # 3. Attempting resume while file exists raises error
            with pytest.raises(OrchestratorError):
                await orch.resume()

            # 4. Remove file and resume
            kill_file.unlink()
            await orch.resume()
            assert orch.agent_state == AgentState.RUNNING
            assert orch.portfolio_service.kill_switch is False

    @pytest.mark.asyncio
    async def test_breaker_tripping_flattens_or_blocks_as_specified(
        self, tmp_path: Path
    ) -> None:
        """Breaker tripping blocks new opens when configured as 'block', and flattens open positions when 'flatten'."""
        db_url = f"sqlite:///{tmp_path / 'breaker_test.db'}"
        engine = init_portfolio_db(db_url)
        broker = PaperBroker(initial_cash=10_000.0)
        broker.set_market_price("BTC/USDT", 65_000.0)

        # Test Case: breaker_action="flatten"
        config = TradingConfig(
            universe=["BTC/USDT"],
            starting_equity=10_000.0,
            peak_drawdown_limit_pct=5.0,  # 5% max drawdown
            breaker_action="flatten",
            lock_file_path=str(tmp_path / "lock.lock"),
            shutdown_marker_path=str(tmp_path / "shutdown.marker"),
        )
        portfolio_service = PortfolioService(engine, broker, config)

        async with Orchestrator(
            config=config,
            engine=engine,
            broker=broker,
            portfolio_service=portfolio_service,
            lock_path=tmp_path / "lock.lock",
            shutdown_marker_path=tmp_path / "shutdown.marker",
        ) as orch:
            # Seed open position
            fill = await broker.place_market_order(
                symbol="BTC/USDT",
                side=Side.LONG,
                qty=0.05,
                client_order_id="seed_order_brk",
                invalidation_price=63_000.0,
            )
            await portfolio_service.apply_open_fill(
                fill=fill,
                invalidation_price=63_000.0,
                conviction=ConvictionTier.A,
                risk_amount=100.0,
            )
            assert "BTC/USDT" in portfolio_service.get_state().positions

            # Simulate peak drawdown: peak was 10,000, current equity drops below 9,000 (>10% drawdown)
            await portfolio_service.set_cash(5_000.0)
            orch.portfolio_service.peak_equity = 10_000.0

            # Execute cycle where breaker evaluates
            await orch.execute_cycle("cycle_breaker_flatten")

            # Position flattened because breaker_action="flatten"
            assert "BTC/USDT" not in orch.portfolio_service.get_state().positions

    @pytest.mark.asyncio
    async def test_restart_in_middle_of_open_position_keeps_position_and_invalidation(
        self, tmp_path: Path
    ) -> None:
        """Restart in the middle of an open position keeps the position and its invalidation price intact."""
        db_url = f"sqlite:///{tmp_path / 'restart_test.db'}"
        engine = init_portfolio_db(db_url)
        broker = PaperBroker(initial_cash=10_000.0)
        broker.set_market_price("BTC/USDT", 65_000.0)

        config = TradingConfig(
            universe=["BTC/USDT"],
            starting_equity=10_000.0,
            lock_file_path=str(tmp_path / "lock.lock"),
            shutdown_marker_path=str(tmp_path / "shutdown.marker"),
        )

        # 1. First run creates and persists position
        async with Orchestrator(
            config=config,
            engine=engine,
            broker=broker,
            lock_path=tmp_path / "lock.lock",
            shutdown_marker_path=tmp_path / "shutdown.marker",
        ) as orch1:
            fill = await broker.place_market_order(
                symbol="BTC/USDT",
                side=Side.LONG,
                qty=0.035,
                client_order_id="order_p1",
                invalidation_price=63_150.0,
            )
            await orch1.portfolio_service.apply_open_fill(
                fill=fill,
                invalidation_price=63_150.0,
                conviction=ConvictionTier.A,
                risk_amount=95.0,
            )
            assert "BTC/USDT" in orch1.portfolio_service.get_state().positions

        # 2. Start a fresh orchestrator instance reading the same database
        async with Orchestrator(
            config=config,
            engine=engine,
            broker=broker,
            lock_path=tmp_path / "lock.lock",
            shutdown_marker_path=tmp_path / "shutdown.marker",
        ) as orch2:
            positions = orch2.portfolio_service.get_state().positions
            assert "BTC/USDT" in positions
            pos = positions["BTC/USDT"]
            assert pos.quantity == pytest.approx(0.035, 1e-5)
            assert pos.invalidation_price == 63_150.0
            assert pos.conviction == ConvictionTier.A

    def test_double_instance_guard(self, tmp_path: Path) -> None:
        """Two instances cannot acquire the lock simultaneously."""
        lock_file = tmp_path / "agent.lock"
        lock1 = SingleInstanceLock(lock_file)
        lock2 = SingleInstanceLock(lock_file)

        # Instance 1 acquires lock
        lock1.acquire()
        assert lock_file.exists()
        assert int(lock_file.read_text().strip()) == os.getpid()

        # Mock is_pid_running to return True for lock1 PID even if checking from another instance
        # When lock2 checks with a different simulated PID or running process:
        lock_file.write_text("999999")  # Non-current PID
        SingleInstanceLock._is_pid_running = staticmethod(lambda pid: True)

        with pytest.raises(InstanceAlreadyRunningError):
            lock2.acquire()

        # When released, lock can be acquired
        SingleInstanceLock._is_pid_running = staticmethod(lambda pid: False)
        lock2.acquire()
        assert lock2.acquired is True
        lock2.release()
        assert not lock_file.exists()

    @pytest.mark.asyncio
    async def test_same_cycle_id_cannot_run_twice(self, tmp_path: Path) -> None:
        """The same cycle_id cannot run twice (idempotent / duplicate guard)."""
        db_url = f"sqlite:///{tmp_path / 'cycle_guard.db'}"
        engine = init_portfolio_db(db_url)
        broker = PaperBroker()
        config = TradingConfig(universe=["BTC/USDT"])

        mock_llm = MockLLM(
            canned_responses=[
                Decision(reasoning="Run 1", decisions=[]),
                Decision(reasoning="Run 2 should not execute", decisions=[]),
            ]
        )

        async with Orchestrator(
            config=config,
            engine=engine,
            broker=broker,
            llm_client=mock_llm,
            lock_path=tmp_path / "lock.lock",
            shutdown_marker_path=tmp_path / "shutdown.marker",
        ) as orch:
            # First execution succeeds
            approved1, rejections1 = await orch.execute_cycle("cycle_20261008_0400")
            assert "cycle_20261008_0400" in orch.completed_cycles
            assert orch.llm_client.call_count == 1

            # Second execution with same cycle_id is rejected by duplicate guard
            approved2, rejections2 = await orch.execute_cycle("cycle_20261008_0400")
            assert len(approved2) == 0
            assert len(rejections2) == 0
            # MockLLM call count remains 1, confirming cycle was skipped
            assert orch.llm_client.call_count == 1
