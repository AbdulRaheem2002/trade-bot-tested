"""Tests for pure-pandas indicators and no-lookahead invariant – Phase 2."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.data.indicators import (
    add_1d_indicators,
    add_4h_indicators,
    atr,
    ema,
    rsi,
    volume_sma,
)
from tests.fixtures_indicators import (
    ATR14_CANDLES,
    ATR14_EXPECTED_AT_13,
    ATR14_EXPECTED_AT_14,
    EMA3_EXPECTED,
    EMA3_PRICES,
    RSI14_EXPECTED_AT_14,
    RSI14_EXPECTED_AT_15,
    RSI14_PRICES,
    VOLUME_DATA,
    VOLUME_SMA3_EXPECTED,
)


class TestIndicatorsAgainstHandComputedFixtures:
    """Validate indicators strictly against mathematically hand-computed fixtures."""

    def test_ema3_matches_fixture(self):
        s = pd.Series(EMA3_PRICES)
        result = ema(s, span=3)
        assert len(result) == len(EMA3_EXPECTED)
        for actual, expected in zip(result, EMA3_EXPECTED, strict=True):
            assert actual == pytest.approx(expected, rel=1e-6)

    def test_volume_sma3_matches_fixture(self):
        s = pd.Series(VOLUME_DATA)
        result = volume_sma(s, period=3)
        assert len(result) == len(VOLUME_SMA3_EXPECTED)
        assert math.isnan(result.iloc[0])
        assert math.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(VOLUME_SMA3_EXPECTED[2])
        assert result.iloc[3] == pytest.approx(VOLUME_SMA3_EXPECTED[3])
        assert result.iloc[4] == pytest.approx(VOLUME_SMA3_EXPECTED[4])

    def test_rsi14_matches_wilder_fixture(self):
        s = pd.Series(RSI14_PRICES)
        result = rsi(s, period=14)

        # Before index 14, RSI should be NaN
        for i in range(14):
            assert math.isnan(result.iloc[i])

        # Exact match at index 14
        assert result.iloc[14] == pytest.approx(RSI14_EXPECTED_AT_14, rel=1e-5)
        # Exact match at index 15
        assert result.iloc[15] == pytest.approx(RSI14_EXPECTED_AT_15, rel=1e-5)

    def test_atr14_matches_wilder_fixture(self):
        df = pd.DataFrame(ATR14_CANDLES)
        result = atr(df, period=14)

        # Before index 13, ATR should be NaN
        for i in range(13):
            assert math.isnan(result.iloc[i])

        # Exact match at index 13 (first ATR from 14 bars)
        assert result.iloc[13] == pytest.approx(ATR14_EXPECTED_AT_13, rel=1e-5)
        # Exact match at index 14 (subsequent Wilder smoothed)
        assert result.iloc[14] == pytest.approx(ATR14_EXPECTED_AT_14, rel=1e-5)


class TestIndicatorsNoLookaheadInvariant:
    """Test that future candles NEVER modify any past indicator values."""

    @pytest.fixture
    def synthetic_history(self) -> pd.DataFrame:
        rng = np.random.default_rng(12345)
        n = 100
        close = 100.0 + np.cumsum(rng.normal(0, 1, n))
        high = close + rng.uniform(0.5, 2.0, n)
        low = close - rng.uniform(0.5, 2.0, n)
        open_ = (high + low) / 2.0
        volume = rng.uniform(1000, 5000, n)
        return pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )

    def test_ema_no_lookahead(self, synthetic_history: pd.DataFrame):
        subset = synthetic_history.iloc[:50]
        full = synthetic_history.copy()

        ema_sub = ema(subset["close"], span=20)
        ema_full = ema(full["close"], span=20)

        # Adding 50 future candles must NOT change the first 50 values
        pd.testing.assert_series_equal(ema_sub, ema_full.iloc[:50], check_names=False)

    def test_rsi_no_lookahead(self, synthetic_history: pd.DataFrame):
        subset = synthetic_history.iloc[:50]
        full = synthetic_history.copy()

        rsi_sub = rsi(subset["close"], period=14)
        rsi_full = rsi(full["close"], period=14)

        pd.testing.assert_series_equal(rsi_sub, rsi_full.iloc[:50], check_names=False)

    def test_atr_no_lookahead(self, synthetic_history: pd.DataFrame):
        subset = synthetic_history.iloc[:50]
        full = synthetic_history.copy()

        atr_sub = atr(subset, period=14)
        atr_full = atr(full, period=14)

        pd.testing.assert_series_equal(atr_sub, atr_full.iloc[:50], check_names=False)

    def test_multi_timeframe_indicators_enricher(self, synthetic_history: pd.DataFrame):
        df_4h = add_4h_indicators(synthetic_history)
        assert "ema200_4h" in df_4h.columns
        assert "atr14_4h" in df_4h.columns
        assert "rsi14_4h" in df_4h.columns
        assert "volume_sma20_4h" in df_4h.columns

        df_1d = add_1d_indicators(synthetic_history)
        assert "ema50_1d" in df_1d.columns
        assert "ema200_1d" in df_1d.columns
        assert "volume_sma20_1d" in df_1d.columns
