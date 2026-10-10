"""Tests for circuit breakers – Rule 12.

Pure functions testing:
- compute_daily_loss_pct
- compute_weekly_loss_pct
- compute_peak_drawdown_pct
- compute_breaker_state with exact boundary conditions and latching behavior
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.common.models import EquityPoint
from app.config.settings import TradingConfig
from app.risk.breakers import (
    compute_breaker_state,
    compute_daily_loss_pct,
    compute_peak_drawdown_pct,
    compute_weekly_loss_pct,
)

# Reference timestamp: Wednesday, June 18, 2025, 14:30:00 UTC
# Start of day (SOD): 2025-06-18 00:00:00 UTC
# Start of week (SOW): Monday, 2025-06-16 00:00:00 UTC
NOW = datetime(2025, 6, 18, 14, 30, 0, tzinfo=UTC)
SOD = datetime(2025, 6, 18, 0, 0, 0, tzinfo=UTC)
SOW = datetime(2025, 6, 16, 0, 0, 0, tzinfo=UTC)


class TestDailyLossBreaker:
    """Daily loss calculation: (SOD equity - current) / SOD equity * 100."""

    def test_daily_loss_empty_history(self):
        """Returns 0.0% loss when no history is available."""
        assert compute_daily_loss_pct(9_000.0, [], NOW) == 0.0

    def test_daily_loss_no_point_before_sod(self):
        """History points only exist after start of day."""
        hist = [
            EquityPoint(timestamp=datetime(2025, 6, 18, 1, 0, 0, tzinfo=UTC), equity=10_000.0),
        ]
        assert compute_daily_loss_pct(9_000.0, hist, NOW) == 0.0

    def test_daily_loss_profitable_day(self):
        """If equity increased since SOD, loss is 0.0%."""
        hist = [
            EquityPoint(timestamp=SOD, equity=10_000.0),
        ]
        assert compute_daily_loss_pct(10_500.0, hist, NOW) == 0.0

    def test_daily_loss_calculation(self):
        """Exact percentage drop from SOD."""
        hist = [
            EquityPoint(timestamp=SOD, equity=10_000.0),
        ]
        # Drop 300 from 10,000 = 3.0%
        assert compute_daily_loss_pct(9_700.0, hist, NOW) == pytest.approx(3.0)
        # Drop 250 from 10,000 = 2.5%
        assert compute_daily_loss_pct(9_750.0, hist, NOW) == pytest.approx(2.5)


class TestWeeklyLossBreaker:
    """Weekly loss calculation: (SOW equity - current) / SOW equity * 100."""

    def test_weekly_loss_empty_history(self):
        assert compute_weekly_loss_pct(9_000.0, [], NOW) == 0.0

    def test_weekly_loss_profitable_week(self):
        hist = [
            EquityPoint(timestamp=SOW, equity=10_000.0),
        ]
        assert compute_weekly_loss_pct(11_000.0, hist, NOW) == 0.0

    def test_weekly_loss_calculation(self):
        hist = [
            EquityPoint(timestamp=SOW, equity=10_000.0),
            EquityPoint(timestamp=SOD, equity=9_800.0),
        ]
        # Drop 600 from SOW 10,000 = 6.0%
        assert compute_weekly_loss_pct(9_400.0, hist, NOW) == pytest.approx(6.0)


class TestPeakDrawdownBreaker:
    """Peak drawdown calculation: (peak - current) / peak * 100."""

    def test_peak_drawdown_at_new_high(self):
        assert compute_peak_drawdown_pct(10_000.0, 10_000.0) == 0.0

    def test_peak_drawdown_above_peak(self):
        assert compute_peak_drawdown_pct(12_000.0, 10_000.0) == 0.0

    def test_peak_drawdown_zero_or_negative_peak(self):
        assert compute_peak_drawdown_pct(5_000.0, 0.0) == 0.0
        assert compute_peak_drawdown_pct(5_000.0, -100.0) == 0.0

    def test_peak_drawdown_calculation(self):
        # 10,000 peak -> 9,000 current = 10.0%
        assert compute_peak_drawdown_pct(9_000.0, 10_000.0) == pytest.approx(10.0)
        # 10,000 peak -> 9_500 current = 5.0%
        assert compute_peak_drawdown_pct(9_500.0, 10_000.0) == pytest.approx(5.0)


class TestBreakerStateBoundariesAndLatching:
    """Boundary conditions for all 3 breakers and is_paused latching."""

    def test_no_breakers_tripped(self, cfg: TradingConfig):
        """All drawdowns below limits."""
        hist = [
            EquityPoint(timestamp=SOW, equity=10_000.0),
            EquityPoint(timestamp=SOD, equity=10_000.0),
        ]
        # Limits: daily=3%, weekly=6%, peak_dd=10%
        # Current = 9,800 -> daily=2%, weekly=2%, peak=2%
        state = compute_breaker_state(
            current_equity=9_800.0,
            peak_equity=10_000.0,
            equity_history=hist,
            config=cfg,
            now=NOW,
        )
        assert not state.daily_loss_blocked
        assert not state.weekly_loss_blocked
        assert not state.peak_drawdown_blocked
        assert not state.is_paused
        assert not state.any_active

    def test_daily_loss_boundary(self, cfg: TradingConfig):
        """Daily loss trips at exactly daily_loss_limit_pct (3.0%), not below."""
        hist = [
            EquityPoint(timestamp=SOW, equity=10_000.0),
            EquityPoint(timestamp=SOD, equity=10_000.0),
        ]
        # Just below: 2.99% loss (current = 9,701)
        state_below = compute_breaker_state(
            current_equity=9_701.0,
            peak_equity=10_000.0,
            equity_history=hist,
            config=cfg,
            now=NOW,
        )
        assert not state_below.daily_loss_blocked

        # Exactly at limit: 3.00% loss (current = 9,700)
        state_at = compute_breaker_state(
            current_equity=9_700.0,
            peak_equity=10_000.0,
            equity_history=hist,
            config=cfg,
            now=NOW,
        )
        assert state_at.daily_loss_blocked
        assert not state_at.weekly_loss_blocked
        assert not state_at.peak_drawdown_blocked
        assert state_at.any_active
        # Daily loss alone does NOT latch is_paused
        assert not state_at.is_paused

    def test_weekly_loss_boundary(self, cfg: TradingConfig):
        """Weekly loss trips at exactly weekly_loss_limit_pct (6.0%), not below."""
        hist = [
            EquityPoint(timestamp=SOW, equity=10_000.0),
            # SOD was 9,500 so daily loss from SOD is only (9500-9400)/9500 = 1.05%
            EquityPoint(timestamp=SOD, equity=9_500.0),
        ]
        # Just below weekly limit (5.99% from SOW 10,000 -> 9,401)
        state_below = compute_breaker_state(
            current_equity=9_401.0,
            peak_equity=10_000.0,
            equity_history=hist,
            config=cfg,
            now=NOW,
        )
        assert not state_below.weekly_loss_blocked

        # Exactly at weekly limit (6.00% from SOW 10,000 -> 9,400)
        state_at = compute_breaker_state(
            current_equity=9_400.0,
            peak_equity=10_000.0,
            equity_history=hist,
            config=cfg,
            now=NOW,
        )
        assert state_at.weekly_loss_blocked
        assert not state_at.daily_loss_blocked
        assert not state_at.peak_drawdown_blocked
        assert state_at.any_active
        # Weekly loss alone does not latch is_paused
        assert not state_at.is_paused

    def test_peak_drawdown_boundary_and_latching(self, cfg: TradingConfig):
        """Peak drawdown trips at 8.0% (cfg default), latches is_paused=True."""
        hist = [
            EquityPoint(timestamp=SOW, equity=9_250.0),
            EquityPoint(timestamp=SOD, equity=9_210.0),
        ]
        peak = 10_000.0

        # Just below peak drawdown (7.99% -> 9,201)
        state_below = compute_breaker_state(
            current_equity=9_201.0,
            peak_equity=peak,
            equity_history=hist,
            config=cfg,
            now=NOW,
            previously_paused=False,
        )
        assert not state_below.peak_drawdown_blocked
        assert not state_below.is_paused

        # Exactly at peak drawdown (8.00% -> 9,200)
        state_at = compute_breaker_state(
            current_equity=9_200.0,
            peak_equity=peak,
            equity_history=hist,
            config=cfg,
            now=NOW,
            previously_paused=False,
        )
        assert state_at.peak_drawdown_blocked
        assert state_at.is_paused

    def test_previously_paused_stays_latched(self, cfg: TradingConfig):
        """If previously_paused was True, is_paused stays True even after full recovery."""
        hist = [
            EquityPoint(timestamp=SOW, equity=10_000.0),
            EquityPoint(timestamp=SOD, equity=10_000.0),
        ]
        # Current equity is back to peak, drawdowns are 0%
        state = compute_breaker_state(
            current_equity=10_000.0,
            peak_equity=10_000.0,
            equity_history=hist,
            config=cfg,
            now=NOW,
            previously_paused=True,
        )
        assert not state.daily_loss_blocked
        assert not state.weekly_loss_blocked
        assert not state.peak_drawdown_blocked
        # But is_paused remains latched
        assert state.is_paused
        assert state.any_active
