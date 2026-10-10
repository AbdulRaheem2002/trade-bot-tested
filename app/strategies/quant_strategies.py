"""Quantitative Strategy Ensemble & Meta-Selector – 4 Elite Trading Architectures.

Implements the 4 distinct quantitative trading models:
1. AstraQuant Multi-Timeframe Structure Pullback (0xethanq/astra-quant-agent)
   - Focus: High win rate, shortest evidence chain, nearest invalidation, fast capital turnover.
2. ProbablyProfit Momentum Breakout with Volume Confirmation (randomness11/probablyprofit)
   - Focus: Directional impulse, volume ratio >= 1.5-2.0x, acceleration, no zombie positions.
3. DaviddTech Extreme Stretch Mean-Reversion (DaviddTech / AstraQuant box edge)
   - Focus: Overextended snap-back (>= 2.0 ATR stretch from EMA20), RSI exhaustion, mean reversion.
4. Core Systematic Macro Trend-Following Breakout
   - Focus: Multi-timeframe trend alignment (4h & 1d EMAs), structural swing continuation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class StrategyEvaluationResult:
    """Outcome of evaluating one quantitative strategy against market data."""

    strategy_id: str
    strategy_name: str
    source_attribution: str
    action: str  # "LONG", "SHORT", "WAIT"
    confidence_pct: int
    score: float  # 0.0 - 10.0
    eligible: bool
    entry_zone: str
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: str
    evidence: str
    invalidation_condition: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AstraQuantStructureStrategy:
    """Strategy 1: High-Win-Rate Multi-Timeframe Structure Pullback.

    Doctrine:
    - Win rate first — only take trades with the shortest evidence chain and nearest invalidation.
    - Capital turnover second — rapid resolution at first structure target; no zombie positions.
    - Risk-reward third — layered take-profit (TP1 locks break-even, TP2 captures extension).
    """

    STRATEGY_ID = "astra_quant_structure"
    STRATEGY_NAME = "AstraQuant MTF Structure Pullback"
    SOURCE = "0xethanq/astra-quant-agent"

    @classmethod
    def evaluate(
        cls,
        price: float,
        ema20_4h: float,
        ema50_4h: float,
        ema200_4h: float,
        atr14_4h: float,
        rsi14_4h: float,
        spread_bps: float = 0.0,
    ) -> StrategyEvaluationResult:
        if atr14_4h <= 0 or price <= 0:
            return cls._empty_wait("Invalid price or ATR values")

        dist_ema20_atr = abs(price - ema20_4h) / atr14_4h
        is_above_ema200 = price >= ema200_4h

        # Long Setup: Trend is bullish (above EMA200), pulling back to EMA20/50 structure
        if is_above_ema200:
            is_near_structure = dist_ema20_atr <= 1.6
            healthy_rsi = 38.0 <= rsi14_4h <= 63.0

            if is_near_structure and healthy_rsi:
                stop_dist = max(atr14_4h * 1.8, price * 0.008)
                stop_loss = round(price - stop_dist, 4)
                tp1 = round(price + (stop_dist * 1.5), 4)
                tp2 = round(price + (stop_dist * 2.5), 4)

                score = 8.5
                if dist_ema20_atr <= 0.8:
                    score += 0.8
                if 42.0 <= rsi14_4h <= 55.0:
                    score += 0.5
                score = min(score, 9.8)

                evidence = (
                    f"Price (${price:,.4f}) is holding above 4h EMA200 (${ema200_4h:,.4f}) with a clean pullback "
                    f"to EMA20 (${ema20_4h:,.4f}, {dist_ema20_atr:.1f}x ATR away). RSI {rsi14_4h:.1f} shows stable consolidation."
                )
                invalidation = f"Candle close below structural stop at ${stop_loss:,.4f} or break of EMA50 (${ema50_4h:,.4f})."

                return StrategyEvaluationResult(
                    strategy_id=cls.STRATEGY_ID,
                    strategy_name=cls.STRATEGY_NAME,
                    source_attribution=cls.SOURCE,
                    action="LONG",
                    confidence_pct=int(score * 9.5),
                    score=round(score, 1),
                    eligible=True,
                    entry_zone=f"${price * 0.998:,.4f} - ${price * 1.002:,.4f}",
                    stop_loss=stop_loss,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    risk_reward_ratio="1 : 2.0 (Tiered TP1/TP2)",
                    evidence=evidence,
                    invalidation_condition=invalidation,
                )

        # Short Setup: Trend is bearish (below EMA200), pulling back up to EMA20/50 resistance
        else:
            is_near_structure = dist_ema20_atr <= 1.6
            healthy_rsi = 37.0 <= rsi14_4h <= 62.0

            if is_near_structure and healthy_rsi:
                stop_dist = max(atr14_4h * 1.8, price * 0.008)
                stop_loss = round(price + stop_dist, 4)
                tp1 = round(price - (stop_dist * 1.5), 4)
                tp2 = round(price - (stop_dist * 2.5), 4)

                score = 8.3
                if dist_ema20_atr <= 0.8:
                    score += 0.8
                if 45.0 <= rsi14_4h <= 58.0:
                    score += 0.5
                score = min(score, 9.7)

                evidence = (
                    f"Price (${price:,.4f}) rejected below 4h EMA200 (${ema200_4h:,.4f}) with bear pullback "
                    f"into EMA20 resistance (${ema20_4h:,.4f}, {dist_ema20_atr:.1f}x ATR). RSI at {rsi14_4h:.1f}."
                )
                invalidation = f"Candle close above structural stop at ${stop_loss:,.4f} or EMA20 reclaim."

                return StrategyEvaluationResult(
                    strategy_id=cls.STRATEGY_ID,
                    strategy_name=cls.STRATEGY_NAME,
                    source_attribution=cls.SOURCE,
                    action="SHORT",
                    confidence_pct=int(score * 9.5),
                    score=round(score, 1),
                    eligible=True,
                    entry_zone=f"${price * 0.998:,.4f} - ${price * 1.002:,.4f}",
                    stop_loss=stop_loss,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    risk_reward_ratio="1 : 2.0 (Tiered TP1/TP2)",
                    evidence=evidence,
                    invalidation_condition=invalidation,
                )

        return StrategyEvaluationResult(
            strategy_id=cls.STRATEGY_ID,
            strategy_name=cls.STRATEGY_NAME,
            source_attribution=cls.SOURCE,
            action="WAIT",
            confidence_pct=35,
            score=3.5,
            eligible=False,
            entry_zone=f"${price:,.4f}",
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            risk_reward_ratio="N/A",
            evidence=f"Price is stretched {dist_ema20_atr:.1f}x ATR away from structure; chasing mid-range is forbidden by AstraQuant doctrine.",
            invalidation_condition="Wait for pullback directly to EMA20/50 structure.",
        )

    @classmethod
    def _empty_wait(cls, reason: str) -> StrategyEvaluationResult:
        return StrategyEvaluationResult(
            strategy_id=cls.STRATEGY_ID,
            strategy_name=cls.STRATEGY_NAME,
            source_attribution=cls.SOURCE,
            action="WAIT",
            confidence_pct=20,
            score=2.0,
            eligible=False,
            entry_zone="N/A",
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            risk_reward_ratio="N/A",
            evidence=reason,
            invalidation_condition="N/A",
        )


class ProbablyProfitMomentumStrategy:
    """Strategy 2: Momentum Breakout with Volume Confirmation.

    Doctrine:
    - Trade strictly in the direction of established short-term impulse.
    - Require volume confirmation (volume >= 1.5x - 2.0x 20-period average) to avoid bull/bear traps.
    - Exit quickly when momentum stalls or reverses.
    """

    STRATEGY_ID = "momentum_volume_breakout"
    STRATEGY_NAME = "Volume Momentum Breakout"
    SOURCE = "randomness11/probablyprofit"

    @classmethod
    def evaluate(
        cls,
        price: float,
        ema20_4h: float,
        atr14_4h: float,
        rsi14_4h: float,
        volume_ratio: float = 1.0,
    ) -> StrategyEvaluationResult:
        if atr14_4h <= 0 or price <= 0:
            return cls._empty_wait("Invalid price or ATR")

        has_volume_confirmation = volume_ratio >= 1.45
        strong_volume = volume_ratio >= 1.8

        # Bullish Breakout Impulse
        if price > ema20_4h and rsi14_4h >= 54.0:
            if has_volume_confirmation:
                stop_dist = max(atr14_4h * 1.8, price * 0.01)
                stop_loss = round(price - stop_dist, 4)
                tp1 = round(price + (stop_dist * 1.8), 4)
                tp2 = round(price + (stop_dist * 3.0), 4)

                score = 8.2 if not strong_volume else 9.4
                if rsi14_4h >= 60.0:
                    score += 0.3
                score = min(score, 9.8)

                evidence = (
                    f"High-volume directional surge: 4h volume is {volume_ratio:.2f}x the 20-period average. "
                    f"Momentum accelerating above EMA20 (${ema20_4h:,.4f}) with RSI at {rsi14_4h:.1f}."
                )
                invalidation = f"Price reverses below breakout base at ${stop_loss:,.4f} or volume collapses on stall."

                return StrategyEvaluationResult(
                    strategy_id=cls.STRATEGY_ID,
                    strategy_name=cls.STRATEGY_NAME,
                    source_attribution=cls.SOURCE,
                    action="LONG",
                    confidence_pct=int(score * 9.5),
                    score=round(score, 1),
                    eligible=True,
                    entry_zone=f"${price * 0.997:,.4f} - ${price * 1.003:,.4f}",
                    stop_loss=stop_loss,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    risk_reward_ratio="1 : 2.4 (Breakout Runner)",
                    evidence=evidence,
                    invalidation_condition=invalidation,
                )
            else:
                return StrategyEvaluationResult(
                    strategy_id=cls.STRATEGY_ID,
                    strategy_name=cls.STRATEGY_NAME,
                    source_attribution=cls.SOURCE,
                    action="WAIT",
                    confidence_pct=45,
                    score=4.5,
                    eligible=False,
                    entry_zone=f"${price:,.4f}",
                    stop_loss=0.0,
                    take_profit_1=0.0,
                    take_profit_2=0.0,
                    risk_reward_ratio="N/A",
                    evidence=f"Price is trending, but volume ratio ({volume_ratio:.2f}x) is below the mandatory 1.45x confirmation threshold.",
                    invalidation_condition="Wait for volume spike >= 1.5x on breakout candle.",
                )

        # Bearish Breakdown Impulse
        if price < ema20_4h and rsi14_4h <= 46.0:
            if has_volume_confirmation:
                stop_dist = max(atr14_4h * 1.8, price * 0.01)
                stop_loss = round(price + stop_dist, 4)
                tp1 = round(price - (stop_dist * 1.8), 4)
                tp2 = round(price - (stop_dist * 3.0), 4)

                score = 8.0 if not strong_volume else 9.2
                evidence = (
                    f"Heavy-volume bearish breakdown: volume is {volume_ratio:.2f}x 20-period average. "
                    f"RSI {rsi14_4h:.1f} confirms downward acceleration below EMA20 (${ema20_4h:,.4f})."
                )
                invalidation = f"Price reclaims ${stop_loss:,.4f} or upside impulse breaks downward structure."

                return StrategyEvaluationResult(
                    strategy_id=cls.STRATEGY_ID,
                    strategy_name=cls.STRATEGY_NAME,
                    source_attribution=cls.SOURCE,
                    action="SHORT",
                    confidence_pct=int(score * 9.5),
                    score=round(score, 1),
                    eligible=True,
                    entry_zone=f"${price * 0.997:,.4f} - ${price * 1.003:,.4f}",
                    stop_loss=stop_loss,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    risk_reward_ratio="1 : 2.4 (Breakdown Runner)",
                    evidence=evidence,
                    invalidation_condition=invalidation,
                )
            else:
                return StrategyEvaluationResult(
                    strategy_id=cls.STRATEGY_ID,
                    strategy_name=cls.STRATEGY_NAME,
                    source_attribution=cls.SOURCE,
                    action="WAIT",
                    confidence_pct=40,
                    score=4.0,
                    eligible=False,
                    entry_zone=f"${price:,.4f}",
                    stop_loss=0.0,
                    take_profit_1=0.0,
                    take_profit_2=0.0,
                    risk_reward_ratio="N/A",
                    evidence=f"Bearish pressure present but volume ratio ({volume_ratio:.2f}x) lacks institutional expansion.",
                    invalidation_condition="Wait for volume confirmation >= 1.5x.",
                )

        return StrategyEvaluationResult(
            strategy_id=cls.STRATEGY_ID,
            strategy_name=cls.STRATEGY_NAME,
            source_attribution=cls.SOURCE,
            action="WAIT",
            confidence_pct=30,
            score=3.0,
            eligible=False,
            entry_zone=f"${price:,.4f}",
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            risk_reward_ratio="N/A",
            evidence=f"Market is consolidating with neutral RSI {rsi14_4h:.1f}; no directional momentum impulse detected.",
            invalidation_condition="Wait for a decisive 4h candle impulse.",
        )

    @classmethod
    def _empty_wait(cls, reason: str) -> StrategyEvaluationResult:
        return StrategyEvaluationResult(
            strategy_id=cls.STRATEGY_ID,
            strategy_name=cls.STRATEGY_NAME,
            source_attribution=cls.SOURCE,
            action="WAIT",
            confidence_pct=20,
            score=2.0,
            eligible=False,
            entry_zone="N/A",
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            risk_reward_ratio="N/A",
            evidence=reason,
            invalidation_condition="N/A",
        )


class DaviddTechMeanReversionStrategy:
    """Strategy 3: Extreme Stretch Mean-Reversion.

    Doctrine:
    - Price stretched too far from fair-value anchor (EMA20/VWAP) snaps back.
    - Requires extreme price stretch (>= 2.0 - 2.5 ATR deviation from EMA20).
    - Requires oscillator in extreme territory (RSI > 70 or < 30) signaling exhaustion.
    """

    STRATEGY_ID = "extreme_stretch_reversion"
    STRATEGY_NAME = "Extreme Stretch Mean-Reversion"
    SOURCE = "DaviddTech/ai-trading-agent & AstraQuant Box Edges"

    @classmethod
    def evaluate(
        cls,
        price: float,
        ema20_4h: float,
        atr14_4h: float,
        rsi14_4h: float,
    ) -> StrategyEvaluationResult:
        if atr14_4h <= 0 or price <= 0:
            return cls._empty_wait("Invalid price or ATR")

        stretch_atr = abs(price - ema20_4h) / atr14_4h
        is_extended = stretch_atr >= 2.0
        is_severely_extended = stretch_atr >= 2.6

        # Overbought Extreme: Short snap-back to EMA20
        if price > ema20_4h and rsi14_4h >= 68.0 and is_extended:
            stop_dist = max(atr14_4h * 1.4, price * 0.009)
            stop_loss = round(price + stop_dist, 4)
            # Targets: Return to EMA20 anchor
            tp1 = round(price - (abs(price - ema20_4h) * 0.6), 4)
            tp2 = round(ema20_4h, 4)

            score = 8.4 if not is_severely_extended else 9.5
            if rsi14_4h >= 75.0:
                score += 0.4
            score = min(score, 9.8)

            evidence = (
                f"Severe upward extension: price is stretched {stretch_atr:.2f}x ATR above fair-value EMA20 (${ema20_4h:,.4f}). "
                f"RSI is overbought at {rsi14_4h:.1f}, indicating high snap-back probability to the mean."
            )
            invalidation = f"Momentum continuation breaking above local extreme at ${stop_loss:,.4f}."

            return StrategyEvaluationResult(
                strategy_id=cls.STRATEGY_ID,
                strategy_name=cls.STRATEGY_NAME,
                source_attribution=cls.SOURCE,
                action="SHORT",
                confidence_pct=int(score * 9.5),
                score=round(score, 1),
                eligible=True,
                entry_zone=f"${price * 0.998:,.4f} - ${price * 1.002:,.4f}",
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                risk_reward_ratio=f"1 : {(abs(price - tp2) / stop_dist):.1f} (Mean-Reversion Snap)",
                evidence=evidence,
                invalidation_condition=invalidation,
            )

        # Oversold Extreme: Long snap-back to EMA20
        if price < ema20_4h and rsi14_4h <= 32.0 and is_extended:
            stop_dist = max(atr14_4h * 1.4, price * 0.009)
            stop_loss = round(price - stop_dist, 4)
            tp1 = round(price + (abs(price - ema20_4h) * 0.6), 4)
            tp2 = round(ema20_4h, 4)

            score = 8.4 if not is_severely_extended else 9.5
            if rsi14_4h <= 25.0:
                score += 0.4
            score = min(score, 9.8)

            evidence = (
                f"Severe downward extension: price is stretched {stretch_atr:.2f}x ATR below fair-value EMA20 (${ema20_4h:,.4f}). "
                f"RSI is oversold at {rsi14_4h:.1f}, signaling exhausted sellers and rubber-band bounce potential."
            )
            invalidation = f"Breakdown continuing below flush low at ${stop_loss:,.4f}."

            return StrategyEvaluationResult(
                strategy_id=cls.STRATEGY_ID,
                strategy_name=cls.STRATEGY_NAME,
                source_attribution=cls.SOURCE,
                action="LONG",
                confidence_pct=int(score * 9.5),
                score=round(score, 1),
                eligible=True,
                entry_zone=f"${price * 0.998:,.4f} - ${price * 1.002:,.4f}",
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                risk_reward_ratio=f"1 : {(abs(tp2 - price) / stop_dist):.1f} (Mean-Reversion Snap)",
                evidence=evidence,
                invalidation_condition=invalidation,
            )

        return StrategyEvaluationResult(
            strategy_id=cls.STRATEGY_ID,
            strategy_name=cls.STRATEGY_NAME,
            source_attribution=cls.SOURCE,
            action="WAIT",
            confidence_pct=25,
            score=2.5,
            eligible=False,
            entry_zone=f"${price:,.4f}",
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            risk_reward_ratio="N/A",
            evidence=f"Price is only {stretch_atr:.1f}x ATR away from EMA20 (RSI {rsi14_4h:.1f}); extreme 2.0x stretch requirement is not met.",
            invalidation_condition="Wait for extreme extension beyond 2.0 ATRs.",
        )

    @classmethod
    def _empty_wait(cls, reason: str) -> StrategyEvaluationResult:
        return StrategyEvaluationResult(
            strategy_id=cls.STRATEGY_ID,
            strategy_name=cls.STRATEGY_NAME,
            source_attribution=cls.SOURCE,
            action="WAIT",
            confidence_pct=20,
            score=2.0,
            eligible=False,
            entry_zone="N/A",
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            risk_reward_ratio="N/A",
            evidence=reason,
            invalidation_condition="N/A",
        )


class MacroTrendFollowingStrategy:
    """Strategy 4: Core Systematic Macro Trend-Following.

    Doctrine:
    - Multi-timeframe trend alignment (Price > EMA20 > EMA50 > EMA200).
    - Capture prolonged macro trends with wide trailing stops.
    """

    STRATEGY_ID = "macro_trend_following"
    STRATEGY_NAME = "Macro Trend-Following Breakout"
    SOURCE = "Core Systematic Trend Architecture"

    @classmethod
    def evaluate(
        cls,
        price: float,
        ema20_4h: float,
        ema50_4h: float,
        ema200_4h: float,
        ema50_1d: float,
        atr14_4h: float,
        rsi14_4h: float,
    ) -> StrategyEvaluationResult:
        if atr14_4h <= 0 or price <= 0:
            return cls._empty_wait("Invalid price or ATR")

        is_bullish_stacked = price > ema20_4h > ema50_4h > ema200_4h
        is_bearish_stacked = price < ema20_4h < ema50_4h < ema200_4h

        # Bullish Macro Continuation
        if is_bullish_stacked and price >= ema50_1d:
            stop_dist = max(atr14_4h * 2.2, price * 0.012)
            stop_loss = round(price - stop_dist, 4)
            tp1 = round(price + (stop_dist * 2.0), 4)
            tp2 = round(price + (stop_dist * 3.5), 4)

            score = 8.8
            if 50.0 <= rsi14_4h <= 65.0:
                score += 0.5
            score = min(score, 9.6)

            evidence = (
                f"Full bullish alignment across timeframes: Price (${price:,.4f}) > EMA20 (${ema20_4h:,.4f}) > "
                f"EMA50 (${ema50_4h:,.4f}) > EMA200 (${ema200_4h:,.4f}) and holding above 1d EMA50 (${ema50_1d:,.4f})."
            )
            invalidation = f"Macro trend failure: close below 4h EMA50 at ${ema50_4h:,.4f} or stop at ${stop_loss:,.4f}."

            return StrategyEvaluationResult(
                strategy_id=cls.STRATEGY_ID,
                strategy_name=cls.STRATEGY_NAME,
                source_attribution=cls.SOURCE,
                action="LONG",
                confidence_pct=int(score * 9.5),
                score=round(score, 1),
                eligible=True,
                entry_zone=f"${price * 0.998:,.4f} - ${price * 1.002:,.4f}",
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                risk_reward_ratio="1 : 2.5 (Macro Trend Rider)",
                evidence=evidence,
                invalidation_condition=invalidation,
            )

        # Bearish Macro Breakdown
        if is_bearish_stacked and price <= ema50_1d:
            stop_dist = max(atr14_4h * 2.2, price * 0.012)
            stop_loss = round(price + stop_dist, 4)
            tp1 = round(price - (stop_dist * 2.0), 4)
            tp2 = round(price - (stop_dist * 3.5), 4)

            score = 8.6
            if 35.0 <= rsi14_4h <= 50.0:
                score += 0.5
            score = min(score, 9.5)

            evidence = (
                f"Full bearish alignment across timeframes: Price (${price:,.4f}) < EMA20 (${ema20_4h:,.4f}) < "
                f"EMA50 (${ema50_4h:,.4f}) < EMA200 (${ema200_4h:,.4f}) below 1d EMA50 (${ema50_1d:,.4f})."
            )
            invalidation = f"Bearish structure break: close above 4h EMA50 at ${ema50_4h:,.4f}."

            return StrategyEvaluationResult(
                strategy_id=cls.STRATEGY_ID,
                strategy_name=cls.STRATEGY_NAME,
                source_attribution=cls.SOURCE,
                action="SHORT",
                confidence_pct=int(score * 9.5),
                score=round(score, 1),
                eligible=True,
                entry_zone=f"${price * 0.998:,.4f} - ${price * 1.002:,.4f}",
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                risk_reward_ratio="1 : 2.5 (Macro Trend Rider)",
                evidence=evidence,
                invalidation_condition=invalidation,
            )

        return StrategyEvaluationResult(
            strategy_id=cls.STRATEGY_ID,
            strategy_name=cls.STRATEGY_NAME,
            source_attribution=cls.SOURCE,
            action="WAIT",
            confidence_pct=35,
            score=3.5,
            eligible=False,
            entry_zone=f"${price:,.4f}",
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            risk_reward_ratio="N/A",
            evidence="Moving averages are intertwined or conflicting; no unified macro trend alignment.",
            invalidation_condition="Wait for clear EMA alignment (EMA20 > EMA50 > EMA200).",
        )

    @classmethod
    def _empty_wait(cls, reason: str) -> StrategyEvaluationResult:
        return StrategyEvaluationResult(
            strategy_id=cls.STRATEGY_ID,
            strategy_name=cls.STRATEGY_NAME,
            source_attribution=cls.SOURCE,
            action="WAIT",
            confidence_pct=20,
            score=2.0,
            eligible=False,
            entry_zone="N/A",
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            risk_reward_ratio="N/A",
            evidence=reason,
            invalidation_condition="N/A",
        )


def evaluate_all_strategies(
    price: float,
    ema20_4h: float,
    ema50_4h: float,
    ema200_4h: float,
    ema50_1d: float,
    atr14_4h: float,
    rsi14_4h: float,
    volume_ratio: float = 1.0,
    spread_bps: float = 0.0,
) -> dict[str, Any]:
    """Execute all 4 quantitative strategies and run the meta-selector tournament."""
    s1 = AstraQuantStructureStrategy.evaluate(
        price=price,
        ema20_4h=ema20_4h,
        ema50_4h=ema50_4h,
        ema200_4h=ema200_4h,
        atr14_4h=atr14_4h,
        rsi14_4h=rsi14_4h,
        spread_bps=spread_bps,
    )
    s2 = ProbablyProfitMomentumStrategy.evaluate(
        price=price,
        ema20_4h=ema20_4h,
        atr14_4h=atr14_4h,
        rsi14_4h=rsi14_4h,
        volume_ratio=volume_ratio,
    )
    s3 = DaviddTechMeanReversionStrategy.evaluate(
        price=price,
        ema20_4h=ema20_4h,
        atr14_4h=atr14_4h,
        rsi14_4h=rsi14_4h,
    )
    s4 = MacroTrendFollowingStrategy.evaluate(
        price=price,
        ema20_4h=ema20_4h,
        ema50_4h=ema50_4h,
        ema200_4h=ema200_4h,
        ema50_1d=ema50_1d,
        atr14_4h=atr14_4h,
        rsi14_4h=rsi14_4h,
    )

    strategies = [s1, s2, s3, s4]
    eligible_strategies = [s for s in strategies if s.eligible and s.action in ("LONG", "SHORT")]

    # Sort by quantitative score descending
    if eligible_strategies:
        eligible_strategies.sort(key=lambda x: x.score, reverse=True)
        winner = eligible_strategies[0]
    else:
        # If none meet entry rules, default to highest score WAIT
        strategies.sort(key=lambda x: x.score, reverse=True)
        winner = strategies[0]

    return {
        "winning_strategy": winner,
        "all_strategies": {
            s1.strategy_id: s1,
            s2.strategy_id: s2,
            s3.strategy_id: s3,
            s4.strategy_id: s4,
        },
        "has_actionable_setup": bool(eligible_strategies),
        "scores_summary": {s.strategy_name: f"{s.action} ({s.score}/10)" for s in strategies},
    }
