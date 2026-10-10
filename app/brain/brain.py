"""Brain daily research runner – Rule / Phase 4.

Builds the daily context:
- 30 days of daily candles and indicators for the universe
- Current portfolio state and equity curve
- Closed trades from the last 7 days
- Previous BrainReport
Calls LiteLLM, validates the output against BrainReport schema,
and stores the report in SQLite.
On failure: keeps previous report marked as stale.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from app.brain.schemas import BrainReport, MarketRegime, SymbolAnalysis
from app.common.llm_client import LLMClient
from app.common.logging import get_logger
from app.config.loader import load_config
from app.config.settings import TradingConfig
from app.data.indicators import compute_indicators_for_df
from app.data.news import NewsProvider, NullNewsProvider
from app.data.store import CandleStore
from app.portfolio.db import BrainReportModel, TradeModel, init_portfolio_db
from app.portfolio.service import PortfolioService

logger = get_logger(__name__)

_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "brain_system.md"


async def run_brain(
    client: LLMClient | None = None,
    store: CandleStore | None = None,
    portfolio_service: PortfolioService | None = None,
    engine: Engine | str | None = None,
    config: TradingConfig | None = None,
    news_provider: NewsProvider | None = None,
    now: datetime | None = None,
    prompt_path: Path | str | None = None,
) -> BrainReport:
    """Run the Brain daily analysis cycle."""
    config = config or load_config()
    now_dt = now or datetime.now(UTC)
    now_ms = int(now_dt.timestamp() * 1000)

    # 1. Resolve database engine and session
    db_engine = init_portfolio_db(engine or "sqlite:///trading_agent.db")
    client = client or LLMClient(
        timeout_s=config.decision_llm_timeout_s,
        max_retries=config.llm_max_retries,
    )
    store = store or CandleStore(db_engine)
    news = news_provider or NullNewsProvider()

    # 2. Retrieve previous BrainReport from database
    prev_report_model = None
    with Session(db_engine) as session:
        prev_report_model = session.scalar(
            select(BrainReportModel).order_by(BrainReportModel.created_at.desc())
        )

    previous_report: BrainReport | None = None
    if prev_report_model:
        try:
            previous_report = BrainReport.model_validate_json(prev_report_model.report_json)
        except Exception:
            previous_report = None

    # 3. Read system prompt
    p_path = Path(prompt_path or _DEFAULT_PROMPT_PATH)
    if not p_path.exists():
        raise FileNotFoundError(f"Brain system prompt not found at {p_path}")
    system_prompt = p_path.read_text(encoding="utf-8")

    # 4. Gather 30-day daily candle data and technical indicators for universe
    universe_data: dict[str, list[dict]] = {}
    for sym in config.universe:
        df_1d = store.get_candles(sym, "1d", now_ms=now_ms)
        if not df_1d.empty:
            df_enriched = compute_indicators_for_df(df_1d, "1d")
            # Take up to last 30 daily closed candles
            tail_df = df_enriched.tail(30)
            candles_compact = []
            for _, r in tail_df.iterrows():
                candles_compact.append(
                    {
                        "open_time": int(r["open_time"]),
                        "close": float(r["close"]),
                        "volume": float(r["volume"]),
                        "ema50": float(r["ema50_1d"])
                        if "ema50_1d" in r and not r.isna()["ema50_1d"]
                        else None,
                        "ema200": float(r["ema200_1d"])
                        if "ema200_1d" in r and not r.isna()["ema200_1d"]
                        else None,
                    }
                )
            universe_data[sym] = candles_compact
        else:
            universe_data[sym] = []

    # 5. Gather portfolio state and closed trades from last 7 days
    portfolio_info: dict = {
        "equity": config.starting_equity,
        "cash": config.starting_equity,
        "positions": {},
    }
    if portfolio_service is not None:
        p_state = portfolio_service.get_state()
        portfolio_info = {
            "equity": p_state.equity,
            "cash": p_state.cash,
            "peak_equity": p_state.peak_equity,
            "positions": {
                sym: {
                    "side": pos.side.value,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "invalidation_price": pos.invalidation_price,
                    "conviction": pos.conviction.value,
                    "risk_amount": pos.risk_amount,
                }
                for sym, pos in p_state.positions.items()
            },
            "recent_equity_curve": [
                {"timestamp": pt.timestamp.isoformat(), "equity": pt.equity}
                for pt in p_state.equity_history[-30:]
            ],
        }

    # Closed trades from last 7 days
    seven_days_ago = now_dt - timedelta(days=7)
    recent_trades_data: list[dict] = []
    with Session(db_engine) as session:
        trades = session.scalars(
            select(TradeModel)
            .where(TradeModel.exit_time >= seven_days_ago)
            .order_by(TradeModel.exit_time.desc())
        ).all()
        for t in trades:
            recent_trades_data.append(
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "realized_pnl": t.realized_pnl,
                    "r_multiple": t.r_multiple,
                    "exit_reason": t.exit_reason,
                }
            )

    # News headlines
    headlines = await news.get_latest_headlines(symbols=config.universe, limit=10)

    # 6. Assemble user prompt with XML safety wrappers
    user_prompt_sections = [
        f"Timestamp: {now_dt.isoformat()}",
        f"Universe: {json.dumps(config.universe)}",
        "<market_data>",
        f"DailyCandles30d: {json.dumps(universe_data)}",
        f"PortfolioState: {json.dumps(portfolio_info)}",
        f"ClosedTradesLast7d: {json.dumps(recent_trades_data)}",
        f"PreviousBrainReport: {previous_report.model_dump_json() if previous_report else 'None'}",
        "</market_data>",
    ]

    if headlines:
        user_prompt_sections.append("<untrusted_news>")
        user_prompt_sections.append(json.dumps(headlines))
        user_prompt_sections.append("</untrusted_news>")

    user_prompt = "\n".join(user_prompt_sections)

    # 7. Call LLM
    try:
        report, meta = await client.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=BrainReport,
        )

        if report is None:
            raise ValueError("Brain LLM returned empty report")

        # 8. Persist new report to DB
        with Session(db_engine) as session:
            db_report = BrainReportModel(
                report_id=report.report_id,
                created_at=report.created_at,
                market_bias="mixed",
                theses_json=json.dumps([s.model_dump() for s in report.symbols]),
                report_json=report.model_dump_json(),
            )
            session.add(db_report)
            session.commit()

        logger.info(
            "Brain analysis complete for %s (regime=%s, tokens=%d)",
            now_dt.strftime("%Y-%m-%d"),
            report.market_regime.value,
            meta.total_tokens,
        )
        return report

    except Exception as exc:
        logger.error("Brain LLM cycle failed: %s. Preserving previous report as stale.", exc)
        if previous_report is not None:
            previous_report.is_stale = True
            return previous_report

        # If no previous report exists, construct a safe fallback
        fallback = BrainReport(
            market_regime=MarketRegime.UNCLEAR,
            summary="Brain cycle failed to generate report; safe default applied.",
            symbols=[
                SymbolAnalysis(
                    symbol=sym,
                    bias="neutral",  # type: ignore
                    thesis="No data available due to LLM failure",
                    risks="High uncertainty",
                )
                for sym in config.universe
            ],
            performance_review="N/A",
            suggested_adjustments=["Hold current risk stance"],
            is_stale=True,
        )
        return fallback
