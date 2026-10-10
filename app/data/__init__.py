"""Data module – market data fetching, storage, and indicators."""

from __future__ import annotations

from app.data.exchange_client import ExchangeClient
from app.data.indicators import add_1d_indicators, add_4h_indicators, atr, ema, rsi, volume_sma
from app.data.news import NewsProvider, NullNewsProvider, sanitize_headline
from app.data.snapshot import build_snapshot
from app.data.store import CandleStore, backfill, detect_gaps, repair_gaps, update_to_latest

__all__ = [
    "CandleStore",
    "ExchangeClient",
    "NewsProvider",
    "NullNewsProvider",
    "add_1d_indicators",
    "add_4h_indicators",
    "atr",
    "backfill",
    "build_snapshot",
    "detect_gaps",
    "ema",
    "repair_gaps",
    "rsi",
    "sanitize_headline",
    "update_to_latest",
    "volume_sma",
]
