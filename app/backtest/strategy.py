"""Deterministic strategy interfaces and implementations – Rule / Phase 7.

Important design rule: The LLM is NOT backtested. Only deterministic strategies
are backtested to eliminate historical data contamination / look-ahead bias.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from app.common.enums import Action, ConvictionTier, Side
from app.common.models import Decision, DecisionItem, MarketSnapshot, PortfolioState
from app.data.indicators import ema


class Strategy(ABC):
    """Base interface for deterministic strategies."""

    name: str = "base_strategy"

    @abstractmethod
    def on_candle_close(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> Decision:
        """Evaluate market snapshot on 4h candle close and return Decision.

        Follows identical Decision contract without using LLM confidence.
        """
        pass


class EMATrendStrategy(Strategy):
    """EMA50 / EMA200 trend-following strategy with ATR-based invalidation.

    Rules:
    - Entry: If no position is held and 4h EMA50 > EMA200 and price > EMA200,
      propose OPEN_LONG with invalidation stop at (current_price - 2.0 * ATR14).
    - Exit: If in LONG position and (EMA50 < EMA200 or price < EMA200),
      propose CLOSE.
    """

    name = "ema_trend"

    def __init__(self, atr_multiplier: float = 2.0) -> None:
        self.atr_multiplier = atr_multiplier

    def on_candle_close(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> Decision:
        decisions: list[DecisionItem] = []
        sym = snapshot.symbol

        # Calculate EMA50 and EMA200 from candle history
        if not snapshot.candles_4h:
            return Decision(reasoning="No 4h candles in snapshot", decisions=[])

        closes = pd.Series([float(c["c"]) for c in snapshot.candles_4h])
        if len(closes) < 50:
            return Decision(reasoning=f"Insufficient candle history ({len(closes)}/50)", decisions=[])

        ema50_s = ema(closes, span=50)
        ema50 = float(ema50_s.iloc[-1])
        ema200 = snapshot.ema200_4h if snapshot.ema200_4h > 0 else float(ema(closes, span=200).iloc[-1])
        price = snapshot.current_price
        atr = snapshot.atr14_4h if snapshot.atr14_4h > 0 else (price * 0.02)

        has_pos = sym in portfolio.positions
        pos = portfolio.positions.get(sym)

        if not has_pos:
            # Bullish trend setup
            if ema50 > ema200 and price > ema200:
                inv_price = round(price - (self.atr_multiplier * atr), 4)
                decisions.append(
                    DecisionItem(
                        action=Action.OPEN_LONG,
                        symbol=sym,
                        invalidation_price=inv_price,
                        conviction=ConvictionTier.A,
                        confidence=1.0,
                        thesis=f"EMA50 ({ema50:.2f}) > EMA200 ({ema200:.2f}) trend alignment",
                        rationale=f"Price {price:.2f} above EMA200 with {self.atr_multiplier} ATR stop",
                    )
                )
        else:
            # Check for trend failure / exit
            if pos and pos.side == Side.LONG and (ema50 < ema200 or price < ema200):
                    decisions.append(
                        DecisionItem(
                            action=Action.CLOSE,
                            symbol=sym,
                            thesis="Trend alignment broken",
                            rationale=f"Price {price:.2f} < EMA200 ({ema200:.2f}) or EMA50 < EMA200",
                        )
                    )

        reasoning = f"EMA trend eval for {sym}: EMA50={ema50:.2f}, EMA200={ema200:.2f}, Price={price:.2f}"
        return Decision(reasoning=reasoning, decisions=decisions)


class Breakout20Strategy(Strategy):
    """20-bar Donchian breakout strategy with ATR-based invalidation.

    Rules:
    - Entry: If no position is held and price breaks above the highest high of the
      prior 20 closed 4h candles, propose OPEN_LONG with invalidation stop at
      (current_price - 2.0 * ATR14).
    - Exit: If in LONG position and price drops below the lowest low of the prior
      10 closed 4h candles, propose CLOSE.
    """

    name = "breakout_20"

    def __init__(self, breakout_bars: int = 20, exit_bars: int = 10, atr_multiplier: float = 2.0) -> None:
        self.breakout_bars = breakout_bars
        self.exit_bars = exit_bars
        self.atr_multiplier = atr_multiplier

    def on_candle_close(
        self,
        snapshot: MarketSnapshot,
        portfolio: PortfolioState,
    ) -> Decision:
        decisions: list[DecisionItem] = []
        sym = snapshot.symbol

        if len(snapshot.candles_4h) < (self.breakout_bars + 1):
            return Decision(
                reasoning=f"Insufficient candles for {self.breakout_bars}-bar breakout",
                decisions=[],
            )

        # Exclude the current candle when calculating the prior channel
        prior_candles = snapshot.candles_4h[-(self.breakout_bars + 1) : -1]
        highest_high = max(float(c["h"]) for c in prior_candles)
        exit_candles = snapshot.candles_4h[-(self.exit_bars + 1) : -1]
        lowest_exit = min(float(c["l"]) for c in exit_candles)

        price = snapshot.current_price
        atr = snapshot.atr14_4h if snapshot.atr14_4h > 0 else (price * 0.02)

        has_pos = sym in portfolio.positions
        pos = portfolio.positions.get(sym)

        if not has_pos:
            # Breakout signal
            if price >= highest_high:
                inv_price = round(price - (self.atr_multiplier * atr), 4)
                decisions.append(
                    DecisionItem(
                        action=Action.OPEN_LONG,
                        symbol=sym,
                        invalidation_price=inv_price,
                        conviction=ConvictionTier.A,
                        confidence=1.0,
                        thesis=f"20-bar breakout above {highest_high:.2f}",
                        rationale=f"Current price {price:.2f} >= prior 20-bar high {highest_high:.2f}",
                    )
                )
        else:
            if pos and pos.side == Side.LONG and price <= lowest_exit:
                    decisions.append(
                        DecisionItem(
                            action=Action.CLOSE,
                            symbol=sym,
                            thesis="Breakout exit trailing stop touched",
                            rationale=f"Price {price:.2f} <= prior {self.exit_bars}-bar low {lowest_exit:.2f}",
                        )
                    )

        reasoning = (
            f"Breakout eval for {sym}: High20={highest_high:.2f}, Low10={lowest_exit:.2f}, Price={price:.2f}"
        )
        return Decision(reasoning=reasoning, decisions=decisions)


STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "ema_trend": EMATrendStrategy,
    "breakout_20": Breakout20Strategy,
}


def get_strategy(name: str, **kwargs: Any) -> Strategy:
    """Instantiate a registered strategy by name."""
    key = name.lower().strip()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available strategies: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[key](**kwargs)
