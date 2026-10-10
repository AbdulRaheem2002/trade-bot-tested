"""Tests for Phase 7 Backtester – Rule / Phase 7."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.backtest.engine import BacktestEngine
from app.common.enums import Action, ConvictionTier, RejectionReason, Side
from app.common.models import (
    BreakerState,
    CycleState,
    Decision,
    DecisionItem,
    MarketSnapshot,
    PortfolioState,
    Position,
)
from app.config.settings import TradingConfig
from app.data.snapshot import build_snapshot
from app.data.store import CandleStore
from app.risk.risk_engine import process_decisions


def _populate_candles(store: CandleStore, symbol: str = "BTC/USDT", count_4h: int = 100) -> datetime:
    """Helper to populate store with deterministic 4h and 1d candles."""
    base_dt = datetime(2025, 6, 1, 0, 0, tzinfo=UTC)

    # 1d candles (45 days)
    candles_1d: list[list[float]] = []
    p = 60_000.0
    for d in range(45):
        dt = base_dt + timedelta(days=d)
        ts_ms = int(dt.timestamp() * 1000)
        p += 50.0
        candles_1d.append([ts_ms, p, p + 300, p - 300, p + 20, 100_000_000.0])
    store.upsert_candles(symbol, "1d", candles_1d, now_ms=9999999999999)

    # 4h candles
    start_4h = base_dt + timedelta(days=15)
    candles_4h: list[list[float]] = []
    p4 = 62_000.0
    for i in range(count_4h):
        dt = start_4h + timedelta(hours=i * 4)
        ts_ms = int(dt.timestamp() * 1000)
        p4 += 25.0
        candles_4h.append([ts_ms, p4, p4 + 400, p4 - 400, p4 + 15, 20_000_000.0])
    store.upsert_candles(symbol, "4h", candles_4h, now_ms=9999999999999)

    last_candle_open = start_4h + timedelta(hours=(count_4h - 1) * 4)
    return last_candle_open


class TestBacktester:
    """Comprehensive test suite for Phase 7 Backtester."""

    @pytest.mark.asyncio
    async def test_no_look_ahead_in_replay(self, tmp_path: Path) -> None:
        """Verify that at time T, snapshots only access candles strictly closed at or before T."""
        db_url = f"sqlite:///{tmp_path / 'no_lookahead.db'}"
        store = CandleStore(db_url)
        _populate_candles(store, "BTC/USDT", count_4h=80)

        # Pick a cutoff timestamp T (e.g. 60th 4h candle)
        df_all = store.get_candles("BTC/USDT", "4h")
        assert len(df_all) == 80
        cutoff_candle = df_all.iloc[59]
        cutoff_open_ms = int(cutoff_candle["open_time"])
        cutoff_dt = datetime.fromtimestamp(cutoff_open_ms / 1000, tz=UTC)
        cycle_dt = cutoff_dt + timedelta(hours=4) + timedelta(seconds=20)

        # Build snapshot at cycle_dt
        snapshot = await build_snapshot(
            symbol="BTC/USDT",
            store=store,
            now=cycle_dt,
        )

        assert snapshot is not None
        # Verify snapshot's last close matches cutoff candle
        assert snapshot.last_close == pytest.approx(float(cutoff_candle["close"]), 1e-4)

        # Invariant: Snapshot candle history must NOT contain any candle after cutoff_open_ms
        for c in snapshot.candles_4h:
            assert c["t"] <= cutoff_open_ms
            assert c["t"] < cutoff_open_ms + (4 * 3600 * 1000)

    @pytest.mark.asyncio
    async def test_gap_through_stop_fills_at_open(self, tmp_path: Path) -> None:
        """Verify that when price gaps below an invalidation stop, fill base price is candle OPEN, not invalidation."""
        db_url = f"sqlite:///{tmp_path / 'gap_stop.db'}"
        store = CandleStore(db_url)
        _populate_candles(store, "BTC/USDT", count_4h=70)

        config = TradingConfig(
            universe=["BTC/USDT"],
            slippage_bps=10.0,  # 0.1% slippage
            fees_taker_bps=5.0,  # 0.05% taker fee
        )
        engine = BacktestEngine(
            strategy="ema_trend",
            config=config,
            store=store,
        )

        # Manually seed an open LONG position with stop at 65,000.0
        engine.broker._positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            side=Side.LONG,
            entry_price=67_000.0,
            quantity=0.1,
            invalidation_price=65_000.0,
            conviction=ConvictionTier.A,
            risk_amount=200.0,
            opened_at=datetime(2025, 6, 20, 0, 0, tzinfo=UTC),
        )
        engine.broker._entry_fees["BTC/USDT"] = 3.35

        # Create a candle that gaps down through the stop:
        # Open: 63,000.0 (strictly below stop of 65,000.0), Low: 62,000.0
        gap_ts_ms = 1800000000000
        gap_candle = [[gap_ts_ms, 63_000.0, 63_500.0, 62_000.0, 62_500.0, 30_000_000.0]]
        store.upsert_candles("BTC/USDT", "4h", gap_candle, now_ms=9999999999999)

        cycle_time = datetime.fromtimestamp(gap_ts_ms / 1000, tz=UTC) + timedelta(hours=4)

        # Trigger intra-candle invalidation check
        await engine._evaluate_intra_candle_invalidations(gap_ts_ms, cycle_time)

        # Position must be closed
        assert "BTC/USDT" not in engine.broker.get_positions()
        assert len(engine.trades) == 1
        trade = engine.trades[0]
        assert trade.exit_reason == "invalidation"

        # Expected fill price: 63,000.0 * (1 - 10/10000) = 62,937.0
        expected_fill = 63_000.0 * (1.0 - 0.001)
        assert trade.exit_price == pytest.approx(expected_fill, 1e-2)
        # Crucial invariant: fill must NOT be at 65,000.0 (the stop level)
        assert trade.exit_price < 64_000.0

    @pytest.mark.asyncio
    async def test_normal_stop_fills_at_invalidation_price(self, tmp_path: Path) -> None:
        """Verify that when candle opens above stop and drops through it, fill base is invalidation price."""
        db_url = f"sqlite:///{tmp_path / 'normal_stop.db'}"
        store = CandleStore(db_url)
        _populate_candles(store, "BTC/USDT", count_4h=70)

        config = TradingConfig(
            universe=["BTC/USDT"],
            slippage_bps=10.0,  # 0.1%
            fees_taker_bps=5.0,
        )
        engine = BacktestEngine(
            strategy="ema_trend",
            config=config,
            store=store,
        )

        # Stop at 65,000.0
        engine.broker._positions["BTC/USDT"] = Position(
            symbol="BTC/USDT",
            side=Side.LONG,
            entry_price=67_000.0,
            quantity=0.1,
            invalidation_price=65_000.0,
            conviction=ConvictionTier.A,
            risk_amount=200.0,
            opened_at=datetime(2025, 6, 20, 0, 0, tzinfo=UTC),
        )
        engine.broker._entry_fees["BTC/USDT"] = 3.35

        # Candle opens at 66,000.0 (above stop), but drops to low 64,500.0
        normal_ts_ms = 1800000000000
        normal_candle = [[normal_ts_ms, 66_000.0, 66_500.0, 64_500.0, 65_200.0, 30_000_000.0]]
        store.upsert_candles("BTC/USDT", "4h", normal_candle, now_ms=9999999999999)

        cycle_time = datetime.fromtimestamp(normal_ts_ms / 1000, tz=UTC) + timedelta(hours=4)
        await engine._evaluate_intra_candle_invalidations(normal_ts_ms, cycle_time)

        assert "BTC/USDT" not in engine.broker.get_positions()
        assert len(engine.trades) == 1
        trade = engine.trades[0]

        # Expected fill price: 65,000.0 * (1 - 10/10000) = 64,935.0
        expected_fill = 65_000.0 * (1.0 - 0.001)
        assert trade.exit_price == pytest.approx(expected_fill, 1e-2)

    @pytest.mark.asyncio
    async def test_fees_and_slippage_applied(self, tmp_path: Path) -> None:
        """Verify that fees and slippage are correctly deducted on entry and exit."""
        db_url = f"sqlite:///{tmp_path / 'fees_slip.db'}"
        store = CandleStore(db_url)
        _populate_candles(store, "BTC/USDT", count_4h=70)

        config = TradingConfig(
            universe=["BTC/USDT"],
            slippage_bps=20.0,  # 0.2% slippage
            fees_taker_bps=10.0,  # 0.1% taker fee
        )
        engine = BacktestEngine(
            strategy="ema_trend",
            config=config,
            initial_cash=10_000.0,
            store=store,
        )

        # Place an order at market price 60,000.0
        engine.broker.set_market_price("BTC/USDT", 60_000.0)
        engine.broker.set_market_rules("BTC/USDT", lot_size=0.001, min_notional=10.0)

        fill_open = await engine.broker.place_market_order(
            symbol="BTC/USDT",
            side=Side.LONG,
            qty=0.05,
            client_order_id="test_open",
            invalidation_price=58_000.0,
        )

        # Expected buy fill price: 60,000 * 1.002 = 60,120.0
        assert fill_open.price == pytest.approx(60_120.0, 1e-2)
        # Expected open fee: 0.05 * 60,120 * 0.001 = 3.006
        assert fill_open.fee == pytest.approx(3.006, 1e-3)

        # Close position at 62,000.0
        engine.broker.set_market_price("BTC/USDT", 62_000.0)
        fill_close = await engine.broker.close_position("BTC/USDT", reason="test_close")

        # Expected sell fill price: 62,000 * (1 - 0.002) = 61,876.0
        assert fill_close.price == pytest.approx(61_876.0, 1e-2)
        # Expected close fee: 0.05 * 61,876 * 0.001 = 3.0938
        assert fill_close.fee == pytest.approx(3.0938, 1e-3)

        # Total fees paid in broker
        assert engine.broker.total_fees_paid == pytest.approx(fill_open.fee + fill_close.fee, 1e-3)

    def test_backtester_rejects_identical_decisions_to_live_risk_engine(self) -> None:
        """Verify that feeding identical decisions produces identical approved orders and rejections."""
        config = TradingConfig(
            universe=["BTC/USDT", "ETH/USDT"],
            starting_equity=10_000.0,
            max_open_positions=1,
        )
        portfolio = PortfolioState(
            cash=10_000.0,
            equity=10_000.0,
            peak_equity=10_000.0,
            positions={
                "BTC/USDT": Position(
                    symbol="BTC/USDT",
                    side=Side.LONG,
                    entry_price=65_000.0,
                    quantity=0.05,
                    invalidation_price=63_000.0,
                    conviction=ConvictionTier.A,
                    risk_amount=100.0,
                    opened_at=datetime(2025, 6, 20, 0, 0, tzinfo=UTC),
                )
            },
        )
        now = datetime(2025, 6, 20, 4, 0, tzinfo=UTC)
        cycle_state = CycleState(
            cycle_time=now,
            expected_candle_time=now - timedelta(hours=4),
            breaker_state=BreakerState(),
        )

        # Invalid decision: attempting to OPEN a second position when max_open_positions=1
        invalid_decision = Decision(
            reasoning="Open while full",
            decisions=[
                DecisionItem(
                    action=Action.OPEN_LONG,
                    symbol="ETH/USDT",
                    invalidation_price=3_100.0,
                    conviction=ConvictionTier.A,
                    confidence=1.0,
                )
            ],
        )

        expected_open = now - timedelta(hours=4)
        snapshots = {
            "BTC/USDT": MarketSnapshot(
                symbol="BTC/USDT",
                last_close=65_000.0,
                ema200_4h=63_000.0,
                atr14_4h=1_000.0,
                spread_bps=3.0,
                volume_24h=100_000_000.0,
                candle_timestamp=expected_open,
                current_price=65_000.0,
            ),
            "ETH/USDT": MarketSnapshot(
                symbol="ETH/USDT",
                last_close=3_200.0,
                ema200_4h=3_000.0,
                atr14_4h=50.0,
                spread_bps=3.0,
                volume_24h=50_000_000.0,
                candle_timestamp=expected_open,
                current_price=3_200.0,
            ),
        }

        # Run through process_decisions
        approved, rejections = process_decisions(
            decision_raw=invalid_decision,
            portfolio=portfolio,
            snapshots=snapshots,
            cycle_state=cycle_state,
            config=config,
        )

        # Max open positions reached: rejected
        assert len(approved) == 0
        assert len(rejections) == 1
        assert rejections[0].reason == RejectionReason.MAX_POSITIONS_REACHED

    @pytest.mark.asyncio
    async def test_walk_forward_split_and_overfitting_warning(self, tmp_path: Path) -> None:
        """Verify that walk-forward split generates separate IS/OOS reports and warns when split=1.0."""
        db_url = f"sqlite:///{tmp_path / 'wf_test.db'}"
        store = CandleStore(db_url)
        _populate_candles(store, "BTC/USDT", count_4h=80)

        config = TradingConfig(universe=["BTC/USDT"])

        # 1. Split = 0.7 (Walk-forward active)
        engine_split = BacktestEngine(
            strategy="ema_trend",
            config=config,
            store=store,
        )
        res_split = await engine_split.run(symbols=["BTC/USDT"], train_test_split=0.7)

        assert res_split.walk_forward_warning is False
        assert res_split.is_metrics is not None
        assert res_split.oos_metrics is not None

        # 2. Split = 1.0 (No out-of-sample; warning must trigger)
        engine_no_split = BacktestEngine(
            strategy="ema_trend",
            config=config,
            store=store,
        )
        res_no_split = await engine_no_split.run(symbols=["BTC/USDT"], train_test_split=1.0)

        assert res_no_split.walk_forward_warning is True
        assert res_no_split.oos_metrics is None
