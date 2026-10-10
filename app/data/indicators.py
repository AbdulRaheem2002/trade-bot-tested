"""Pure-pandas technical indicators – Rule / Phase 2.

Implements:
- ema(series, span): Exponential Moving Average (span N)
- volume_sma(series, period): Simple Moving Average of volume (period N)
- rsi(series, period): Relative Strength Index with Wilder's smoothing
- atr(df, period): Average True Range with Wilder's smoothing
- add_4h_indicators(df): adds ema200, atr14, rsi14, volume_sma20
- add_1d_indicators(df): adds ema50, ema200, volume_sma20

Pure pandas and numpy; no TA libraries. Deterministic and vectorized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average using standard span weighting."""
    if span <= 0:
        raise ValueError(f"span must be positive, got {span}")
    return series.ewm(span=span, adjust=False).mean()


def volume_sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average for volume."""
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    return series.rolling(window=period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Wilder's smoothing:
    Initial avg gain/loss is SMA over the first `period` delta bars.
    Subsequent: avg = (prev_avg * (period - 1) + current) / period.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if len(series) < period + 1:
        return pd.Series(np.nan, index=series.index, dtype=float)

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Convert to numpy for exact Wilder smoothing
    gains = gain.to_numpy()
    losses = loss.to_numpy()
    n = len(series)
    avg_gain = np.full(n, np.nan, dtype=float)
    avg_loss = np.full(n, np.nan, dtype=float)

    # First valid index for period differences is index `period`
    # (since delta.iloc[0] is NaN, delta.iloc[1..period] are first `period` changes)
    if n > period:
        # Sum of first `period` gains (from index 1 to period inclusive)
        first_gain = np.sum(gains[1 : period + 1]) / period
        first_loss = np.sum(losses[1 : period + 1]) / period
        avg_gain[period] = first_gain
        avg_loss[period] = first_loss

        factor = (period - 1) / period
        inv_period = 1.0 / period

        for i in range(period + 1, n):
            avg_gain[i] = avg_gain[i - 1] * factor + gains[i] * inv_period
            avg_loss[i] = avg_loss[i - 1] * factor + losses[i] * inv_period

    rsi_values = np.full(n, np.nan, dtype=float)
    for i in range(period, n):
        ag = avg_gain[i]
        al = avg_loss[i]
        if np.isnan(ag) or np.isnan(al):
            continue
        if al == 0.0:
            rsi_values[i] = 100.0 if ag > 0.0 else 50.0
        else:
            rs = ag / al
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))

    return pd.Series(rsi_values, index=series.index, dtype=float)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing.

    TR = max(H - L, |H - C_prev|, |L - C_prev|)
    First ATR: SMA of first `period` TRs.
    Subsequent: (prev_ATR * (period - 1) + TR) / period.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    if len(df) < period:
        return pd.Series(np.nan, index=df.index, dtype=float)

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    n = len(df)

    # Compute True Range
    tr = np.zeros(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

    atr_vals = np.full(n, np.nan, dtype=float)
    if n >= period:
        atr_vals[period - 1] = np.mean(tr[:period])
        factor = (period - 1) / period
        inv_period = 1.0 / period
        for i in range(period, n):
            atr_vals[i] = atr_vals[i - 1] * factor + tr[i] * inv_period

    return pd.Series(atr_vals, index=df.index, dtype=float)


def add_4h_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append 4h technical indicators:

    - ema200_4h
    - atr14_4h
    - rsi14_4h
    - volume_sma20_4h
    """
    res = df.copy()
    res["ema200_4h"] = ema(res["close"], span=200)
    res["atr14_4h"] = atr(res, period=14)
    res["rsi14_4h"] = rsi(res["close"], period=14)
    res["volume_sma20_4h"] = volume_sma(res["volume"], period=20)
    return res


def add_1d_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append 1d technical indicators:

    - ema50_1d
    - ema200_1d
    - volume_sma20_1d
    """
    res = df.copy()
    res["ema50_1d"] = ema(res["close"], span=50)
    res["ema200_1d"] = ema(res["close"], span=200)
    res["volume_sma20_1d"] = volume_sma(res["volume"], period=20)
    return res


def compute_indicators_for_df(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Convenience helper to compute indicators for a given timeframe."""
    tf = timeframe.strip().lower()
    if tf == "4h":
        return add_4h_indicators(df)
    if tf == "1d":
        return add_1d_indicators(df)
    return df
