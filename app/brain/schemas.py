"""Pydantic schemas for Brain market research and regime reports – Rule / Phase 4."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class MarketRegime(StrEnum):
    """Overall crypto market macro regime."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE = "range"
    HIGH_VOLATILITY = "high_volatility"
    UNCLEAR = "unclear"


class BiasDirection(StrEnum):
    """Directional bias for a symbol."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SymbolAnalysis(BaseModel):
    """Brain thesis and technical levels for an individual symbol."""

    symbol: str
    bias: BiasDirection
    key_support: list[float] = Field(default_factory=list)
    key_resistance: list[float] = Field(default_factory=list)
    thesis: str = ""
    risks: str = ""


class BrainReport(BaseModel):
    """Daily Brain research output (purely advisory)."""

    report_id: str = Field(default_factory=lambda: f"brain_{int(datetime.now(UTC).timestamp())}")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    market_regime: MarketRegime
    summary: str
    symbols: list[SymbolAnalysis] = Field(default_factory=list)
    performance_review: str = ""
    suggested_adjustments: list[str] = Field(default_factory=list)
    is_stale: bool = False
