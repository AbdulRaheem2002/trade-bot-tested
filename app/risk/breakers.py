"""Circuit breakers – Rule 12.

Pure functions.  No imports from execution/ or data/.

Breakers are computed from the equity history:
- daily_loss:  (start-of-day equity – current) / start-of-day equity
- weekly_loss: (start-of-week equity – current) / start-of-week equity
- peak_drawdown: (peak equity – current) / peak equity
"""

from __future__ import annotations

from datetime import datetime

from app.common.models import BreakerState, EquityPoint
from app.common.time_utils import start_of_utc_day, start_of_utc_week
from app.config.settings import TradingConfig


def _find_equity_at_or_before(
    history: list[EquityPoint],
    target: datetime,
) -> float | None:
    """Return the equity value at or just before *target*.

    *history* is assumed sorted ascending by timestamp.  Returns ``None``
    if no point exists at or before *target*.
    """
    result: float | None = None
    for point in history:
        if point.timestamp <= target:
            result = point.equity
        else:
            break
    return result


def compute_daily_loss_pct(
    current_equity: float,
    equity_history: list[EquityPoint],
    now: datetime,
) -> float:
    """Return the drawdown percentage from start-of-day equity.

    Returns 0.0 if no history is available (conservative – no block).
    """
    sod = start_of_utc_day(now)
    ref = _find_equity_at_or_before(equity_history, sod)
    if ref is None or ref <= 0:
        return 0.0
    loss = ref - current_equity
    if loss <= 0:
        return 0.0
    return (loss / ref) * 100.0


def compute_weekly_loss_pct(
    current_equity: float,
    equity_history: list[EquityPoint],
    now: datetime,
) -> float:
    """Return the drawdown percentage from start-of-week equity."""
    sow = start_of_utc_week(now)
    ref = _find_equity_at_or_before(equity_history, sow)
    if ref is None or ref <= 0:
        return 0.0
    loss = ref - current_equity
    if loss <= 0:
        return 0.0
    return (loss / ref) * 100.0


def compute_peak_drawdown_pct(
    current_equity: float,
    peak_equity: float,
) -> float:
    """Return the drawdown percentage from the equity peak."""
    if peak_equity <= 0:
        return 0.0
    dd = peak_equity - current_equity
    if dd <= 0:
        return 0.0
    return (dd / peak_equity) * 100.0


def compute_breaker_state(
    current_equity: float,
    peak_equity: float,
    equity_history: list[EquityPoint],
    config: TradingConfig,
    now: datetime,
    previously_paused: bool = False,
) -> BreakerState:
    """Evaluate all three circuit breakers and return the composite state.

    If the peak-drawdown breaker fires, ``is_paused`` is latched ``True``
    and remains so until a manual ``/resume``.  The *previously_paused*
    flag preserves this latch across calls.
    """
    daily = compute_daily_loss_pct(current_equity, equity_history, now)
    weekly = compute_weekly_loss_pct(current_equity, equity_history, now)
    peak_dd = compute_peak_drawdown_pct(current_equity, peak_equity)

    peak_dd_blocked = peak_dd >= config.peak_drawdown_limit_pct

    return BreakerState(
        daily_loss_blocked=daily >= config.daily_loss_limit_pct,
        weekly_loss_blocked=weekly >= config.weekly_loss_limit_pct,
        peak_drawdown_blocked=peak_dd_blocked,
        is_paused=previously_paused or peak_dd_blocked,
    )
