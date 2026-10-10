"""Quantitative strategy suite – Phase 4 & Ensemble Strategy Tournament.

Includes:
1. AstraQuant Multi-Timeframe Structure Pullback (High win-rate & fast turnover)
2. ProbablyProfit Momentum Breakout with Volume Confirmation
3. DaviddTech Extreme Stretch Mean-Reversion (Overextended snap-back)
4. Macro Trend-Following Breakout (Multi-timeframe systematic trend)
"""

from __future__ import annotations

from app.strategies.quant_strategies import (
    AstraQuantStructureStrategy,
    DaviddTechMeanReversionStrategy,
    MacroTrendFollowingStrategy,
    ProbablyProfitMomentumStrategy,
    StrategyEvaluationResult,
    evaluate_all_strategies,
)

__all__ = [
    "AstraQuantStructureStrategy",
    "DaviddTechMeanReversionStrategy",
    "MacroTrendFollowingStrategy",
    "ProbablyProfitMomentumStrategy",
    "StrategyEvaluationResult",
    "evaluate_all_strategies",
]
