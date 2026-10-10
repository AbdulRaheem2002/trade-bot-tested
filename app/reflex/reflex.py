"""Reflex 4-hour decision loop runner – Rule / Phase 4.

Builds the decision payload:
- Current equity, cash, and open positions (entry, current price, invalidation, unrealized R, thesis)
- Tradable universe and technical indicator snapshots
- Latest BrainReport with its age
- Symbols closed this cycle
- shorts_enabled flag
Calls LiteLLM, passes output to the deterministic RiskEngine,
records the raw request/response and decisions, and returns approved orders & rejections.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.common.llm_client import LLMClient
from app.common.logging import get_logger
from app.common.models import (
    ApprovedOrder,
    CycleState,
    Decision,
    MarketSnapshot,
    Rejection,
)
from app.config.loader import load_config
from app.config.settings import TradingConfig
from app.data.exchange_client import ExchangeClient
from app.data.news import NewsProvider, NullNewsProvider
from app.data.snapshot import build_snapshot
from app.data.store import CandleStore
from app.portfolio.db import BrainReportModel, init_portfolio_db
from app.portfolio.service import PortfolioService
from app.risk.risk_engine import process_decisions

logger = get_logger(__name__)

_DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "prompts" / "reflex_system.md"
)


async def run_reflex(
    cycle_id: str,
    client: LLMClient | None = None,
    portfolio_service: PortfolioService | None = None,
    store: CandleStore | None = None,
    exchange_client: ExchangeClient | None = None,
    config: TradingConfig | None = None,
    engine: Engine | str | None = None,
    cycle_state: CycleState | None = None,
    news_provider: NewsProvider | None = None,
    now: datetime | None = None,
    prompt_path: Path | str | None = None,
    snapshots: dict[str, MarketSnapshot] | None = None,
) -> tuple[list[ApprovedOrder], list[Rejection]]:
    """Execute a single Reflex 4h decision cycle."""
    config = config or load_config()
    now_dt = now or datetime.now(UTC)
    db_engine = init_portfolio_db(engine or "sqlite:///trading_agent.db")
    store = store or CandleStore(db_engine)
    client = client or LLMClient(
        timeout_s=config.decision_llm_timeout_s,
        max_retries=config.llm_max_retries,
    )
    news = news_provider or NullNewsProvider()

    # 1. Initialize or validate CycleState
    if cycle_state is None:
        cycle_state = CycleState(
            cycle_time=now_dt,
            expected_candle_time=now_dt,
        )

    # 2. Build or take MarketSnapshots for universe
    market_snapshots: dict[str, MarketSnapshot] = {}
    if snapshots is not None:
        market_snapshots = dict(snapshots)
    else:
        for sym in config.universe:
            try:
                snap = await build_snapshot(
                    symbol=sym,
                    client=exchange_client,
                    store=store,
                    now=now_dt,
                )
                market_snapshots[sym] = snap
            except Exception as exc:
                logger.warning("Could not build snapshot for %s: %s", sym, exc)

    # Resolve current prices for portfolio state valuation
    current_prices: dict[str, float] = {
        sym: snap.current_price for sym, snap in market_snapshots.items()
    }

    # 3. Retrieve portfolio state
    if portfolio_service is not None:
        portfolio_state = portfolio_service.get_state(prices=current_prices)
    else:
        # Fallback empty portfolio state
        from app.common.models import PortfolioState

        portfolio_state = PortfolioState(
            equity=config.starting_equity,
            cash=config.starting_equity,
            positions={},
            peak_equity=config.starting_equity,
        )

    # 4. Construct positions summary payload
    positions_payload: dict[str, dict[str, Any]] = {}
    for sym, pos in portfolio_state.positions.items():
        c_price = current_prices.get(sym, pos.entry_price)
        positions_payload[sym] = {
            "symbol": pos.symbol,
            "side": pos.side.value,
            "entry_price": pos.entry_price,
            "current_price": c_price,
            "invalidation_price": pos.invalidation_price,
            "unrealized_r": round(pos.unrealized_r(c_price), 2),
            "conviction": pos.conviction.value,
            "risk_amount": pos.risk_amount,
            "thesis": getattr(pos, "thesis", ""),
        }

    # 5. Retrieve latest BrainReport from database
    brain_payload: dict[str, Any] = {"age_hours": None, "report": "None"}
    with Session(db_engine) as session:
        latest_report = session.scalar(
            select(BrainReportModel).order_by(BrainReportModel.created_at.desc())
        )
        if latest_report:
            created_at = latest_report.created_at
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            age_h = (now_dt - created_at).total_seconds() / 3600.0
            brain_payload = {
                "age_hours": round(age_h, 1),
                "report": json.loads(latest_report.report_json),
            }

    # 6. Compact snapshot summary for LLM prompt
    compact_snapshots: dict[str, dict[str, Any]] = {}
    for sym, snap in market_snapshots.items():
        compact_snapshots[sym] = {
            "last_close": snap.last_close,
            "current_price": snap.current_price,
            "ema200_4h": snap.ema200_4h,
            "atr14_4h": snap.atr14_4h,
            "spread_bps": snap.spread_bps,
            "volume_24h": snap.volume_24h,
            "rsi14_4h": snap.rsi14_4h,
            "volume_sma20_4h": snap.volume_sma20_4h,
            "ema50_1d": snap.ema50_1d,
            "ema200_1d": snap.ema200_1d,
        }

    # 7. Fetch news headlines if enabled
    headlines = await news.get_latest_headlines(symbols=config.universe, limit=5)

    # 8. Build user prompt with XML boundary markers
    user_prompt_lines = [
        f"CycleId: {cycle_id}",
        f"Timestamp: {now_dt.isoformat()}",
        f"Universe: {json.dumps(config.universe)}",
        f"ShortsEnabled: {config.shorts_enabled}",
        f"SymbolsClosedThisCycle: {json.dumps(list(cycle_state.symbols_closed_this_cycle))}",
        "",
        "<market_data>",
        f"Equity: {portfolio_state.equity:.2f}",
        f"Cash: {portfolio_state.cash:.2f}",
        f"Positions: {json.dumps(positions_payload)}",
        f"Snapshots: {json.dumps(compact_snapshots)}",
        f"LatestBrainReport: {json.dumps(brain_payload)}",
        "</market_data>",
    ]

    if headlines:
        user_prompt_lines.append("")
        user_prompt_lines.append("<untrusted_news>")
        user_prompt_lines.append(json.dumps(headlines))
        user_prompt_lines.append("</untrusted_news>")

    user_prompt = "\n".join(user_prompt_lines)

    # 9. Read system prompt
    p_path = Path(prompt_path or _DEFAULT_PROMPT_PATH)
    if not p_path.exists():
        raise FileNotFoundError(f"Reflex system prompt not found at {p_path}")
    system_prompt = p_path.read_text(encoding="utf-8")

    # 10. Call LLM
    decision_obj: Decision | None = None
    raw_response: str = ""
    try:
        decision_obj, meta = await client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=Decision,
        )
        raw_response = meta.raw_response
    except Exception as exc:
        logger.warning(
            "Reflex LLM completion failed for cycle %s: %s; falling back to hold-all",
            cycle_id,
            exc,
        )
        raw_response = getattr(exc, "raw_response", "") or str(exc)
        decision_obj = None

    # 11. Process decisions through deterministic Risk Engine
    approved_orders, rejections = process_decisions(
        decision_raw=raw_response,
        portfolio=portfolio_state,
        snapshots=market_snapshots,
        cycle_state=cycle_state,
        config=config,
    )

    # 12. Record decisions and rejections in database
    if portfolio_service is not None:
        await portfolio_service.record_decisions(
            cycle_id=cycle_id,
            raw_response=raw_response,
            parsed_decision=decision_obj or Decision(reasoning="Fallback hold-all", decisions=[]),
            timestamp=now_dt,
        )
        if rejections:
            await portfolio_service.record_rejections(
                cycle_id=cycle_id,
                rejections=rejections,
                timestamp=now_dt,
            )

    logger.info(
        "Reflex cycle %s complete: %d approved orders, %d rejections",
        cycle_id,
        len(approved_orders),
        len(rejections),
    )
    return approved_orders, rejections
