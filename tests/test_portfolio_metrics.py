"""Tests for Portfolio Metrics and CSV Trade Journal – Rule / Phase 3."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from app.common.enums import Side
from app.common.models import ClosedTrade
from app.portfolio.db import EquitySnapshotModel, init_portfolio_db
from app.portfolio.metrics import (
    compute_metrics,
    compute_metrics_from_db,
    export_trade_journal_csv,
)


class TestPortfolioMetricsCalculations:
    """Hand-computed calculations for performance metrics."""

    def test_hand_computed_trade_metrics(self):
        """Verify trade metrics against deterministic hand-checked values.

        Dataset: 4 trades
            Trade 1: PnL = +300.0, R = +2.0, fees = 5.0
            Trade 2: PnL = -150.0, R = -1.0, fees = 4.0
            Trade 3: PnL = +450.0, R = +3.0, fees = 6.0
            Trade 4: PnL = -100.0, R = -0.5, fees = 3.0

        Hand-checked math:
            Total trades: 4
            Winning trades: 2
            Losing trades: 2
            Win rate: (2 / 4) * 100 = 50.0%
            Gross profit: 300 + 450 = 750.0
            Gross loss: 150 + 100 = 250.0
            Profit factor: 750.0 / 250.0 = 3.0
            Average R: (2.0 - 1.0 + 3.0 - 0.5) / 4 = 3.5 / 4 = 0.875
            Expectancy R: (0.5 * 2.5) - (0.5 * 0.75) = 1.25 - 0.375 = 0.875
            Expectancy PnL: (0.5 * 375.0) - (0.5 * 125.0) = 187.5 - 62.5 = 125.0
            Total fees paid: 5 + 4 + 6 + 3 = 18.0
        """
        t1 = ClosedTrade(
            trade_id="t1",
            symbol="BTC/USDT",
            side=Side.LONG,
            entry_price=50_000.0,
            exit_price=53_000.0,
            quantity=0.1,
            entry_time=datetime(2026, 6, 1, 10, 0, tzinfo=UTC),
            exit_time=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
            realized_pnl=300.0,
            r_multiple=2.0,
            fees_total=5.0,
            exit_reason="take_profit",
        )
        t2 = ClosedTrade(
            trade_id="t2",
            symbol="ETH/USDT",
            side=Side.LONG,
            entry_price=3_000.0,
            exit_price=2_850.0,
            quantity=1.0,
            entry_time=datetime(2026, 6, 2, 10, 0, tzinfo=UTC),
            exit_time=datetime(2026, 6, 2, 14, 0, tzinfo=UTC),
            realized_pnl=-150.0,
            r_multiple=-1.0,
            fees_total=4.0,
            exit_reason="stop_loss",
        )
        t3 = ClosedTrade(
            trade_id="t3",
            symbol="SOL/USDT",
            side=Side.LONG,
            entry_price=100.0,
            exit_price=145.0,
            quantity=10.0,
            entry_time=datetime(2026, 6, 3, 10, 0, tzinfo=UTC),
            exit_time=datetime(2026, 6, 3, 14, 0, tzinfo=UTC),
            realized_pnl=450.0,
            r_multiple=3.0,
            fees_total=6.0,
            exit_reason="take_profit",
        )
        t4 = ClosedTrade(
            trade_id="t4",
            symbol="BTC/USDT",
            side=Side.LONG,
            entry_price=54_000.0,
            exit_price=53_000.0,
            quantity=0.1,
            entry_time=datetime(2026, 6, 4, 10, 0, tzinfo=UTC),
            exit_time=datetime(2026, 6, 4, 14, 0, tzinfo=UTC),
            realized_pnl=-100.0,
            r_multiple=-0.5,
            fees_total=3.0,
            exit_reason="stop_loss",
        )

        trades = [t1, t2, t3, t4]

        # Snapshots:
        # Initial equity: 10,000.0
        # Final equity: 10,500.0 -> total return = 5.0%
        # Equity curve: [10000.0, 10300.0, 10150.0, 10600.0, 10500.0]
        # Peak: 10,600.0, trough: 10,500.0 -> drawdown = (10600 - 10500) / 10600 = ~0.9434%
        snapshots = [
            EquitySnapshotModel(
                id=1,
                timestamp=datetime(2026, 6, 1, tzinfo=UTC),
                equity=10_000.0,
                cash=10000.0,
                exposure=0.0,
            ),
            EquitySnapshotModel(
                id=2,
                timestamp=datetime(2026, 6, 2, tzinfo=UTC),
                equity=10_300.0,
                cash=10300.0,
                exposure=1000.0,
            ),
            EquitySnapshotModel(
                id=3,
                timestamp=datetime(2026, 6, 3, tzinfo=UTC),
                equity=10_150.0,
                cash=10150.0,
                exposure=1500.0,
            ),
            EquitySnapshotModel(
                id=4,
                timestamp=datetime(2026, 6, 4, tzinfo=UTC),
                equity=10_600.0,
                cash=10600.0,
                exposure=1000.0,
            ),
            EquitySnapshotModel(
                id=5,
                timestamp=datetime(2026, 6, 5, tzinfo=UTC),
                equity=10_500.0,
                cash=10500.0,
                exposure=0.0,
            ),
        ]

        metrics = compute_metrics(trades, snapshots)

        assert metrics.number_of_trades == 4
        assert metrics.winning_trades == 2
        assert metrics.losing_trades == 2
        assert math.isclose(metrics.win_rate_pct, 50.0, abs_tol=1e-4)
        assert math.isclose(metrics.gross_profit, 750.0, abs_tol=1e-4)
        assert math.isclose(metrics.gross_loss, 250.0, abs_tol=1e-4)
        assert math.isclose(metrics.profit_factor, 3.0, abs_tol=1e-4)
        assert math.isclose(metrics.average_r, 0.875, abs_tol=1e-4)
        assert math.isclose(metrics.expectancy_r, 0.875, abs_tol=1e-4)
        assert math.isclose(metrics.expectancy_pnl, 125.0, abs_tol=1e-4)
        assert math.isclose(metrics.total_fees_paid, 18.0, abs_tol=1e-4)

        # Snapshot metrics
        assert math.isclose(metrics.total_return_pct, 5.0, abs_tol=1e-4)
        expected_dd = (150.0 / 10300.0) * 100.0
        assert math.isclose(metrics.max_drawdown_pct, expected_dd, abs_tol=1e-2)
        assert metrics.sharpe_ratio > 0
        assert metrics.sortino_ratio > 0
        # Exposure: 3 out of 5 snapshots had exposure > 0 -> 60.0%
        assert math.isclose(metrics.exposure_time_pct, 60.0, abs_tol=1e-4)

    def test_empty_trades_metrics(self):
        """Zero trades returns clean zeroed metrics without crashing or ZeroDivisionError."""
        metrics = compute_metrics([], [])
        assert metrics.number_of_trades == 0
        assert metrics.win_rate_pct == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.average_r == 0.0
        assert metrics.max_drawdown_pct == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0

    def test_trade_journal_csv_export(self, tmp_path):
        """CSV export contains expected headers and formatted rows."""
        trade = ClosedTrade(
            trade_id="trade_123",
            symbol="BTC/USDT",
            side=Side.LONG,
            entry_price=50_000.0,
            exit_price=52_000.0,
            quantity=0.1,
            entry_time=datetime(2026, 6, 1, 10, 0, tzinfo=UTC),
            exit_time=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
            realized_pnl=190.0,
            r_multiple=0.95,
            fees_total=10.0,
            exit_reason="take_profit",
        )

        csv_file = tmp_path / "journal.csv"
        csv_text = export_trade_journal_csv([trade], output_path=csv_file)

        assert "trade_id,symbol,side,entry_price" in csv_text
        assert "trade_123,BTC/USDT,long,50000.0000,52000.0000" in csv_text
        assert csv_file.exists()
        assert csv_file.read_bytes().decode("utf-8") == csv_text

    def test_compute_metrics_from_db(self, tmp_path):
        """Verify metric calculation directly against SQLite database."""
        from sqlalchemy.orm import Session

        from app.portfolio.db import TradeModel

        db_path = tmp_path / "metrics_test.db"
        engine = init_portfolio_db(f"sqlite:///{db_path}")

        with Session(engine) as session:
            trade = TradeModel(
                trade_id="db_trade_1",
                symbol="BTC/USDT",
                side="long",
                entry_price=50_000.0,
                exit_price=55_000.0,
                quantity=0.1,
                entry_time=datetime(2026, 6, 1, 10, 0, tzinfo=UTC),
                exit_time=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
                realized_pnl=500.0,
                r_multiple=2.5,
                fees_total=10.0,
                exit_reason="take_profit",
            )
            snap1 = EquitySnapshotModel(
                timestamp=datetime(2026, 6, 1, 10, 0, tzinfo=UTC),
                equity=10_000.0,
                cash=10_000.0,
                exposure=0.0,
            )
            snap2 = EquitySnapshotModel(
                timestamp=datetime(2026, 6, 1, 14, 0, tzinfo=UTC),
                equity=10_500.0,
                cash=10_500.0,
                exposure=0.0,
            )
            session.add_all([trade, snap1, snap2])
            session.commit()

        metrics = compute_metrics_from_db(engine)
        assert metrics.number_of_trades == 1
        assert metrics.winning_trades == 1
        assert math.isclose(metrics.win_rate_pct, 100.0, abs_tol=1e-4)
        assert math.isclose(metrics.gross_profit, 500.0, abs_tol=1e-4)
        assert math.isclose(metrics.total_return_pct, 5.0, abs_tol=1e-4)
