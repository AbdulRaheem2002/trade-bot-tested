"""Autonomous Universe Screener – Dynamic Token Selection & Filtering.

Features:
- Scans all spot trading pairs against USDT on Binance.
- Filters out stablecoins, wrapped tokens, and leveraged assets.
- Enforces institutional liquidity gates: 24h volume > $30M, spread < 10 bps.
- Scores candidates by momentum, volatility, and liquidity depth.
- Ranks and outputs top N highest-opportunity tokens for faster trading.
- Supports updating config.yaml or runtime universe automatically.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import yaml

from app.common.logging import get_logger
from app.data.exchange_client import ExchangeClient

logger = get_logger(__name__)

# Non-tradeable quote currencies or stablecoins / synthetic tokens to exclude
STABLECOIN_AND_SYNTHETIC_BASES = {
    "USDC",
    "FDUSD",
    "TUSD",
    "USDE",
    "EUR",
    "DAI",
    "BUSD",
    "USDP",
    "AEUR",
    "WBTC",
    "WBETH",
    "BTCST",
    "USDD",
    "CUSD",
    "EURI",
}

LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


class UniverseScreener:
    """Automated market scanner and token screener for Binance spot."""

    def __init__(
        self,
        min_quote_volume: float = 30_000_000.0,
        max_spread_bps: float = 10.0,
        quote_asset: str = "USDT",
    ) -> None:
        self.min_quote_volume = min_quote_volume
        self.max_spread_bps = max_spread_bps
        self.quote_asset = quote_asset.upper()

    async def scan_market(
        self,
        client: ExchangeClient,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Scan active exchange tickers and return top N high-opportunity candidates."""
        logger.info("Starting autonomous market scan on %s...", client.exchange_id)
        raw_tickers = await client.fetch_tickers()
        if not raw_tickers:
            logger.warning("No tickers returned from exchange.")
            return []

        candidates: list[dict[str, Any]] = []

        for symbol, ticker in raw_tickers.items():
            # 1. Must be spot USDT pair without margin colon/special suffixes
            if not symbol.endswith(f"/{self.quote_asset}"):
                continue
            if ":" in symbol:
                continue

            base = symbol.split("/")[0].upper()
            if base in STABLECOIN_AND_SYNTHETIC_BASES:
                continue
            if any(base.endswith(sfx) for sfx in LEVERAGED_SUFFIXES):
                continue

            # 2. Extract price, volume, and spread
            last_price = float(ticker.get("last") or ticker.get("close") or 0.0)
            if last_price <= 0:
                continue

            quote_vol = float(ticker.get("quoteVolume") or 0.0)
            if quote_vol < self.min_quote_volume:
                continue

            bid = float(ticker.get("bid") or 0.0)
            ask = float(ticker.get("ask") or 0.0)
            spread_bps = 0.0
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
                spread_bps = ((ask - bid) / mid) * 10_000.0
                if spread_bps > self.max_spread_bps:
                    continue

            # 3. Calculate 24h percentage change (momentum / action)
            percentage = float(ticker.get("percentage") or 0.0)

            # Opportunity score: combines high trading volume with active price movement
            # Active movement generates faster trade setups without taking low-liquidity risk
            vol_score = math.log10(max(1.0, quote_vol))
            momentum_score = abs(percentage)
            opportunity_score = round((momentum_score * 1.5) + vol_score, 2)

            candidates.append(
                {
                    "symbol": symbol,
                    "price": last_price,
                    "quote_volume_24h": quote_vol,
                    "change_24h_pct": percentage,
                    "spread_bps": round(spread_bps, 2),
                    "score": opportunity_score,
                }
            )

        # Sort descending by opportunity score
        candidates.sort(key=lambda c: c["score"], reverse=True)

        # Include core market leaders (BTC, ETH, SOL) if available, and fill remainder with top momentum
        always_include = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        candidate_map = {c["symbol"]: c for c in candidates}

        selected: list[dict[str, Any]] = []
        for anchor in always_include:
            if anchor in candidate_map:
                selected.append(candidate_map[anchor])

        # Fill remaining slots with top momentum tokens
        for c in candidates:
            if len(selected) >= top_n:
                break
            if c["symbol"] not in {s["symbol"] for s in selected}:
                selected.append(c)

        logger.info(
            "Market scan complete. Screened %d eligible pairs. Selected top %d tokens: %s",
            len(candidates),
            len(selected),
            [c["symbol"] for c in selected],
        )
        return selected


def update_config_universe(
    symbols: list[str],
    config_path: Path | str = "config.yaml",
) -> None:
    """Safely update the universe in config.yaml while preserving the rest of the configuration."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

    content = cfg_file.read_text(encoding="utf-8")

    # Replace universe list in YAML
    formatted_universe = "\n".join(f"  - {sym}" for sym in symbols)
    new_universe_block = f"universe:\n{formatted_universe}"

    # Regex replacement for universe block
    pattern = r"universe:\s*\n(?:\s*-\s*[^\n]+\n)+"
    if re.search(pattern, content):
        updated_content = re.sub(pattern, new_universe_block + "\n", content, count=1)
    else:
        # Fallback to YAML parse and rewrite if regex doesn't match
        data = yaml.safe_load(content) or {}
        data["universe"] = symbols
        updated_content = yaml.dump(data, sort_keys=False)

    cfg_file.write_text(updated_content, encoding="utf-8")
    logger.info("Updated config universe with %d symbols: %s", len(symbols), symbols)
