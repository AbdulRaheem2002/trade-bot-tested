"""Unit tests for the 4-Strategy Quantitative Tournament & Meta-Selector."""

from __future__ import annotations

from datetime import UTC, datetime

from app.common.time_utils import (
    expected_closed_candle_time,
    get_next_candle_boundary,
)
from app.strategies.quant_strategies import (
    AstraQuantStructureStrategy,
    DaviddTechMeanReversionStrategy,
    MacroTrendFollowingStrategy,
    ProbablyProfitMomentumStrategy,
    evaluate_all_strategies,
)


def test_astra_quant_structure_long():
    """AstraQuant triggers LONG on healthy pullback to EMA20 above EMA200."""
    result = AstraQuantStructureStrategy.evaluate(
        price=100.0,
        ema20_4h=99.5,   # within 0.5 ATR
        ema50_4h=98.0,
        ema200_4h=90.0,  # price well above EMA200
        atr14_4h=2.0,
        rsi14_4h=48.0,   # healthy pullback RSI
    )
    assert result.action == "LONG"
    assert result.eligible is True
    assert result.confidence_pct >= 80
    assert result.stop_loss < 100.0
    assert result.take_profit_1 > 100.0
    assert result.take_profit_2 > result.take_profit_1


def test_astra_quant_structure_short():
    """AstraQuant triggers SHORT on bear pullback to EMA20 below EMA200."""
    result = AstraQuantStructureStrategy.evaluate(
        price=80.0,
        ema20_4h=80.5,
        ema50_4h=82.0,
        ema200_4h=95.0,  # price well below EMA200
        atr14_4h=1.5,
        rsi14_4h=45.0,
    )
    assert result.action == "SHORT"
    assert result.eligible is True
    assert result.stop_loss > 80.0
    assert result.take_profit_1 < 80.0


def test_astra_quant_rejects_chasing():
    """AstraQuant forbids chasing mid-impulse when price is stretched from structure."""
    result = AstraQuantStructureStrategy.evaluate(
        price=110.0,
        ema20_4h=95.0,   # 15.0 / 2.0 = 7.5x ATR away!
        ema50_4h=90.0,
        ema200_4h=85.0,
        atr14_4h=2.0,
        rsi14_4h=75.0,
    )
    assert result.action == "WAIT"
    assert result.eligible is False


def test_probably_profit_momentum_with_volume():
    """ProbablyProfit triggers LONG when impulse has institutional volume >= 1.5x."""
    result = ProbablyProfitMomentumStrategy.evaluate(
        price=105.0,
        ema20_4h=100.0,
        atr14_4h=2.0,
        rsi14_4h=62.0,
        volume_ratio=1.9,  # Strong institutional volume surge
    )
    assert result.action == "LONG"
    assert result.eligible is True
    assert result.confidence_pct >= 85


def test_probably_profit_rejects_low_volume():
    """ProbablyProfit outputs WAIT if volume confirmation is missing (< 1.45x)."""
    result = ProbablyProfitMomentumStrategy.evaluate(
        price=105.0,
        ema20_4h=100.0,
        atr14_4h=2.0,
        rsi14_4h=62.0,
        volume_ratio=0.8,  # Lacks volume confirmation
    )
    assert result.action == "WAIT"
    assert result.eligible is False


def test_davidd_tech_mean_reversion_overbought():
    """DaviddTech triggers SHORT snap-back on extreme stretch >= 2.0 ATR + RSI >= 68."""
    result = DaviddTechMeanReversionStrategy.evaluate(
        price=115.0,
        ema20_4h=100.0,  # 15.0 / 5.0 = 3.0x ATR stretch
        atr14_4h=5.0,
        rsi14_4h=74.0,   # Overbought exhaustion
    )
    assert result.action == "SHORT"
    assert result.eligible is True
    assert result.take_profit_2 == 100.0  # target is return to EMA20 mean


def test_davidd_tech_rejects_mild_stretch():
    """DaviddTech outputs WAIT when price is within normal fair-value range (< 2.0 ATR)."""
    result = DaviddTechMeanReversionStrategy.evaluate(
        price=102.0,
        ema20_4h=100.0,  # only 0.4x ATR away
        atr14_4h=5.0,
        rsi14_4h=52.0,
    )
    assert result.action == "WAIT"
    assert result.eligible is False


def test_macro_trend_following_stacked():
    """MacroTrend triggers LONG when all EMAs are cleanly stacked."""
    result = MacroTrendFollowingStrategy.evaluate(
        price=120.0,
        ema20_4h=115.0,
        ema50_4h=110.0,
        ema200_4h=95.0,
        ema50_1d=100.0,
        atr14_4h=2.5,
        rsi14_4h=58.0,
    )
    assert result.action == "LONG"
    assert result.eligible is True


def test_evaluate_all_strategies_tournament():
    """Tournament evaluates all 4 strategies and selects the best candidate."""
    eval_res = evaluate_all_strategies(
        price=100.0,
        ema20_4h=99.8,
        ema50_4h=98.0,
        ema200_4h=90.0,
        ema50_1d=92.0,
        atr14_4h=2.0,
        rsi14_4h=49.0,
        volume_ratio=1.1,
    )
    assert "winning_strategy" in eval_res
    assert len(eval_res["all_strategies"]) == 4
    winner = eval_res["winning_strategy"]
    assert winner.strategy_id == "astra_quant_structure"
    assert winner.action == "LONG"


def test_15m_timeframe_expected_closed_candle():
    """Verify 15m timeframe correctly floors to closed candle."""
    # At 14:03 UTC, latest closed 15m candle opened at 13:45 UTC (closed at 14:00)
    dt = datetime(2026, 10, 10, 14, 3, tzinfo=UTC)
    closed = expected_closed_candle_time(dt, timeframe="15m")
    assert closed == datetime(2026, 10, 10, 13, 45, tzinfo=UTC)

    # At 14:15 UTC, latest closed 15m candle opened at 14:00 UTC (closed at 14:15)
    dt_boundary = datetime(2026, 10, 10, 14, 15, tzinfo=UTC)
    closed_boundary = expected_closed_candle_time(dt_boundary, timeframe="15m")
    assert closed_boundary == datetime(2026, 10, 10, 14, 0, tzinfo=UTC)


def test_15m_timeframe_get_next_candle_boundary():
    """Verify 15m timeframe calculates the next close boundary correctly."""
    # At 14:03 UTC, next boundary is 14:15:00 UTC
    dt = datetime(2026, 10, 10, 14, 3, tzinfo=UTC)
    next_boundary = get_next_candle_boundary(dt, timeframe="15m")
    assert next_boundary == datetime(2026, 10, 10, 14, 15, 0, tzinfo=UTC)

    # At 23:55 UTC, next boundary crosses midnight to 00:00:00 UTC of next day
    dt_midnight = datetime(2026, 10, 10, 23, 55, tzinfo=UTC)
    next_midnight = get_next_candle_boundary(dt_midnight, timeframe="15m")
    assert next_midnight == datetime(2026, 10, 11, 0, 0, 0, tzinfo=UTC)


def test_5m_timeframe_calculations():
    """Verify 5m timeframe boundary and closed candle math."""
    # At 15:33 UTC, next boundary is 15:35:00 UTC
    dt = datetime(2026, 10, 10, 15, 33, tzinfo=UTC)
    next_boundary = get_next_candle_boundary(dt, timeframe="5m")
    assert next_boundary == datetime(2026, 10, 10, 15, 35, 0, tzinfo=UTC)

    # At 15:35:20 UTC (wake time), closed candle opened at 15:30:00 UTC
    wake_time = datetime(2026, 10, 10, 15, 35, 20, tzinfo=UTC)
    closed = expected_closed_candle_time(wake_time, timeframe="5m")
    assert closed == datetime(2026, 10, 10, 15, 30, 0, tzinfo=UTC)



