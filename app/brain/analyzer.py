"""Autonomous Coin Analyzer & Signal Generator – On-Demand Technical & AI Insights.

Allows querying any coin on Binance spot (e.g., 'DOGE', 'SOL', 'BTC/USDT').
Fetches live OHLCV data, calculates technical indicators, prompts Gemini AI,
and returns an institutional-grade trading signal (LONG / SHORT / NEUTRAL).
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from app.common.llm_client import LLMClient
from app.common.logging import get_logger
from app.data.exchange_client import ExchangeClient
from app.data.indicators import (
    add_1d_indicators,
    add_4h_indicators,
)
from app.strategies.quant_strategies import evaluate_all_strategies

logger = get_logger(__name__)


class TradeSignal(BaseModel):
    """Structured actionable trade setup signal for an analyzed coin."""

    symbol: str
    current_price: float
    signal: str = Field(description="LONG, SHORT, or NEUTRAL / WAIT")
    conviction: str = Field(description="HIGH (Tier A), MEDIUM (Tier B), or LOW")
    confidence_pct: int = Field(ge=0, le=100)
    selected_strategy: str = Field(default="AstraQuant MTF Structure Pullback", description="Winning strategy selected from the 4 models")
    strategy_attribution: str = Field(default="0xethanq/astra-quant-agent", description="Source repository of the winning strategy")
    strategy_scores: dict[str, str] = Field(default_factory=dict, description="Status/Score of all 4 evaluated strategies")
    winning_edge: str = Field(default="", description="Why this strategy was selected over the other 3")
    entry_zone: str
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: str
    macro_trend: str
    rsi_status: str
    key_support: list[float] = Field(default_factory=list)
    key_resistance: list[float] = Field(default_factory=list)
    thesis: str
    invalidation_condition: str
    indicators: dict[str, Any] = Field(default_factory=dict)
    model_used: str = ""
    latency_s: float = 0.0


def normalize_symbol(raw_input: str) -> str:
    """Normalize user input like 'doge', 'sol', 'btc/usdt' to standard 'XXX/USDT'."""
    clean = raw_input.strip().upper()
    if "/" in clean:
        return clean
    if clean.endswith("USDT"):
        return f"{clean[:-4]}/USDT"
    return f"{clean}/USDT"


def resolve_model_name(engine_or_model: str | None) -> str | None:
    """Resolve shorthand engine names ('groq', 'gemini', 'auto') to full LiteLLM model strings."""
    if not engine_or_model or engine_or_model.lower() in ("auto", "default"):
        return None
    low = engine_or_model.lower()
    if low in ("groq", "fast"):
        import os
        return os.environ.get("FALLBACK_LLM_MODEL", "groq/openai/gpt-oss-20b")
    if low in ("gemini", "macro"):
        import os
        return os.environ.get("LLM_MODEL_NAME", "gemini/gemini-3.5-flash-lite")
    return engine_or_model


class CoinAnalyzer:
    """Fetches real-time market data and generates AI trading signals for any symbol."""

    def __init__(
        self,
        exchange_client: ExchangeClient | None = None,
        llm_client: LLMClient | None = None,
        model: str | None = None,
    ) -> None:
        self.exchange_client = exchange_client
        self.model = resolve_model_name(model)
        self.llm_client = llm_client

    async def analyze_symbol(self, raw_symbol: str) -> TradeSignal:
        """Run complete live data retrieval, technical computation, and AI signal evaluation."""
        symbol = normalize_symbol(raw_symbol)
        owns_exchange = False

        if self.exchange_client is None:
            self.exchange_client = ExchangeClient(exchange_id="binance")
            owns_exchange = True

        if self.llm_client is None:
            self.llm_client = LLMClient()

        try:
            # 1. Fetch live 4h and 1d OHLCV candles
            logger.info("Fetching live market data for %s from Binance...", symbol)
            candles_4h = await self.exchange_client.fetch_ohlcv(symbol, timeframe="4h", limit=120)
            candles_1d = await self.exchange_client.fetch_ohlcv(symbol, timeframe="1d", limit=60)

            if not candles_4h or len(candles_4h) < 20:
                raise ValueError(f"Insufficient candle history for {symbol} on Binance.")

            # 2. Fetch live ticker & order book
            ticker = await self.exchange_client.fetch_ticker(symbol)
            current_price = float(ticker.get("last") or candles_4h[-1][4])
            quote_vol_24h = float(ticker.get("quote_volume") or 0.0)

            best_bid, best_ask = await self.exchange_client.fetch_order_book_top(symbol)
            mid = (best_bid + best_ask) / 2.0 if (best_bid + best_ask) > 0 else current_price
            spread_bps = round(((best_ask - best_bid) / mid) * 10_000.0, 2) if mid > 0 else 0.0

            # 3. Compute Technical Indicators
            df_4h = pd.DataFrame(
                candles_4h, columns=["open_time", "open", "high", "low", "close", "volume"]
            )
            df_4h_ind = add_4h_indicators(df_4h)
            last_4h = df_4h_ind.iloc[-1]

            ema20_4h = float(last_4h.get("ema20_4h") or current_price)
            ema50_4h = float(last_4h.get("ema50_4h") or current_price)
            ema200_4h = float(last_4h.get("ema200_4h") or current_price)
            atr14_4h = float(last_4h.get("atr14_4h") or (current_price * 0.02))
            rsi14_4h = float(last_4h.get("rsi14_4h") or 50.0)

            df_1d = pd.DataFrame(
                candles_1d, columns=["open_time", "open", "high", "low", "close", "volume"]
            )
            df_1d_ind = add_1d_indicators(df_1d)
            last_1d = df_1d_ind.iloc[-1] if not df_1d_ind.empty else {}
            ema50_1d = float(last_1d.get("ema50_1d") or current_price) if not df_1d_ind.empty else current_price

            # 4. Evaluate all 4 Quantitative Strategies
            vol_last = float(last_4h.get("volume") or 0.0)
            vol_sma = float(last_4h.get("volume_sma20_4h") or vol_last or 1.0)
            volume_ratio = round(vol_last / vol_sma, 2) if vol_sma > 0 else 1.0

            strat_eval = evaluate_all_strategies(
                price=current_price,
                ema20_4h=ema20_4h,
                ema50_4h=ema50_4h,
                ema200_4h=ema200_4h,
                ema50_1d=ema50_1d,
                atr14_4h=atr14_4h,
                rsi14_4h=rsi14_4h,
                volume_ratio=volume_ratio,
                spread_bps=spread_bps,
            )
            quant_winner = strat_eval["winning_strategy"]
            all_strats = strat_eval["all_strategies"]
            scores_summary = strat_eval["scores_summary"]

            # 5. Construct AI Prompt Payload with 4-Strategy Meta-Selector
            analysis_prompt = f"""
You are the Chief Quantitative Strategist for an elite crypto trading desk.
Analyze the live market data for {symbol} and evaluate the trade setup across ALL 4 STRATEGIES below.
Choose the SINGLE BEST STRATEGY that offers the highest statistical edge for this trade/time, or output WAIT if none meet strict criteria.

LIVE MARKET SNAPSHOT FOR {symbol}:
- Current Price: ${current_price:,.4f}
- 24h Quote Volume: ${quote_vol_24h:,.0f} USDT
- Order Book Spread: {spread_bps} bps
- 4h EMA 20: ${ema20_4h:,.4f}
- 4h EMA 50: ${ema50_4h:,.4f}
- 4h EMA 200: ${ema200_4h:,.4f}
- 4h RSI (14): {rsi14_4h:.1f}
- 4h ATR (14): ${atr14_4h:,.4f}
- 4h Volume Ratio: {volume_ratio}x of 20-period average
- 1d EMA 50: ${ema50_1d:,.4f}
- Macro Structure: {'Price ABOVE 4h EMA200 (Bullish bias)' if current_price >= ema200_4h else 'Price BELOW 4h EMA200 (Bearish bias)'}

QUANTITATIVE 4-STRATEGY EVALUATION TOURNAMENT:
1. [AstraQuant MTF Structure Pullback] (0xethanq/astra-quant-agent):
   - Action: {all_strats['astra_quant_structure'].action} (Score: {all_strats['astra_quant_structure'].score}/10, Conf: {all_strats['astra_quant_structure'].confidence_pct}%)
   - Evidence: {all_strats['astra_quant_structure'].evidence}
   - Invalidation: {all_strats['astra_quant_structure'].invalidation_condition}

2. [Volume Momentum Breakout] (randomness11/probablyprofit):
   - Action: {all_strats['momentum_volume_breakout'].action} (Score: {all_strats['momentum_volume_breakout'].score}/10, Conf: {all_strats['momentum_volume_breakout'].confidence_pct}%)
   - Evidence: {all_strats['momentum_volume_breakout'].evidence}
   - Invalidation: {all_strats['momentum_volume_breakout'].invalidation_condition}

3. [Extreme Stretch Mean-Reversion] (DaviddTech / Box Edge):
   - Action: {all_strats['extreme_stretch_reversion'].action} (Score: {all_strats['extreme_stretch_reversion'].score}/10, Conf: {all_strats['extreme_stretch_reversion'].confidence_pct}%)
   - Evidence: {all_strats['extreme_stretch_reversion'].evidence}
   - Invalidation: {all_strats['extreme_stretch_reversion'].invalidation_condition}

4. [Macro Trend-Following Breakout] (Core Systematic Architecture):
   - Action: {all_strats['macro_trend_following'].action} (Score: {all_strats['macro_trend_following'].score}/10, Conf: {all_strats['macro_trend_following'].confidence_pct}%)
   - Evidence: {all_strats['macro_trend_following'].evidence}
   - Invalidation: {all_strats['macro_trend_following'].invalidation_condition}

TOURNAMENT RULES:
1. Win rate and capital turnover first — pick the strategy with the shortest evidence chain and tightest invalidation.
2. Select the WINNING STRATEGY among the 4 models (e.g., AstraQuant Structure Pullback, Volume Momentum Breakout, Extreme Stretch Mean-Reversion, Macro Trend-Following).
3. If none meet entry criteria, output NEUTRAL / WAIT and explain why all 4 were disqualified.
4. Return ONLY a valid JSON object matching the schema below. No markdown fences, no explanatory text outside JSON.

JSON SCHEMA:
{{
  "selected_strategy": "AstraQuant MTF Structure Pullback | Volume Momentum Breakout | Extreme Stretch Mean-Reversion | Macro Trend-Following Breakout | None (Wait)",
  "strategy_attribution": "0xethanq/astra-quant-agent | randomness11/probablyprofit | DaviddTech/ai-trading-agent | Core Systematic Architecture",
  "winning_edge": "1-2 sentences on why this specific strategy beat the other 3 for this setup",
  "signal": "LONG | SHORT | NEUTRAL / WAIT",
  "conviction": "HIGH | MEDIUM | LOW",
  "confidence_pct": 75,
  "entry_zone": "$XX.XX - $XX.XX",
  "stop_loss": 0.00,
  "take_profit_1": 0.00,
  "take_profit_2": 0.00,
  "risk_reward_ratio": "1 : 2.5",
  "macro_trend": "Bullish Continuation | Bearish Breakdown | Choppy Consolidation",
  "rsi_status": "Neutral (48.2) | Overbought | Oversold",
  "key_support": [0.00, 0.00],
  "key_resistance": [0.00, 0.00],
  "thesis": "2-3 sentences explaining the structural edge and catalyst",
  "invalidation_condition": "Explicit price breach or structural breakdown that cancels this trade"
}}
"""
            # 6. Call LLM
            active_engine = self.model or "primary LLM"
            logger.info("Requesting 4-strategy AI tournament for %s using %s...", symbol, active_engine)
            _, meta = await self.llm_client.complete(
                system_prompt="You are an institutional crypto quant managing a high-win-rate multi-strategy execution desk.",
                user_prompt=analysis_prompt,
                model=self.model,
            )
            raw_response = meta.raw_response

            # 7. Parse JSON output
            clean_json = raw_response.strip()
            if "```" in clean_json:
                clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json, flags=re.MULTILINE)
                clean_json = re.sub(r"\s*```$", "", clean_json, flags=re.MULTILINE)
            json_match = re.search(r"\{.*\}", clean_json, re.DOTALL)
            if json_match:
                clean_json = json_match.group(0)

            data = json.loads(clean_json)

            # Strategy attribution fallbacks
            selected_strat = data.get("selected_strategy") or quant_winner.strategy_name
            strat_attr = data.get("strategy_attribution") or quant_winner.source_attribution
            winning_edge = data.get("winning_edge") or quant_winner.evidence

            signal_obj = TradeSignal(
                symbol=symbol,
                current_price=current_price,
                signal=data.get("signal", quant_winner.action).upper(),
                conviction=data.get("conviction", "MEDIUM").upper(),
                confidence_pct=int(data.get("confidence_pct", quant_winner.confidence_pct)),
                selected_strategy=selected_strat,
                strategy_attribution=strat_attr,
                strategy_scores=scores_summary,
                winning_edge=winning_edge,
                entry_zone=data.get("entry_zone", quant_winner.entry_zone),
                stop_loss=float(data.get("stop_loss") or quant_winner.stop_loss),
                take_profit_1=float(data.get("take_profit_1") or quant_winner.take_profit_1),
                take_profit_2=float(data.get("take_profit_2") or quant_winner.take_profit_2),
                risk_reward_ratio=data.get("risk_reward_ratio", quant_winner.risk_reward_ratio),
                macro_trend=data.get("macro_trend", "Consolidation"),
                rsi_status=data.get("rsi_status", f"RSI: {rsi14_4h:.1f}"),
                key_support=[float(x) for x in data.get("key_support", [])],
                key_resistance=[float(x) for x in data.get("key_resistance", [])],
                thesis=data.get("thesis", "Setup aligns with highest-scoring quantitative strategy."),
                invalidation_condition=data.get("invalidation_condition", quant_winner.invalidation_condition),
                indicators={
                    "current_price": current_price,
                    "ema20_4h": round(ema20_4h, 4),
                    "ema50_4h": round(ema50_4h, 4),
                    "ema200_4h": round(ema200_4h, 4),
                    "atr14_4h": round(atr14_4h, 4),
                    "rsi14_4h": round(rsi14_4h, 1),
                    "volume_ratio": volume_ratio,
                    "volume_24h_usd": quote_vol_24h,
                    "spread_bps": spread_bps,
                },
                model_used=meta.model or "unknown",
                latency_s=round(meta.latency_s, 2),
            )
            return signal_obj

        finally:
            if owns_exchange and self.exchange_client:
                await self.exchange_client.close()
