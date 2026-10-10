"""FastAPI read-only monitoring and authenticated control API – Rule / Phase 6.

Bound to 127.0.0.1 by default.
Endpoints:
  GET  /status        - General portfolio status, agent state, breaker status.
  GET  /positions     - Open positions list with stops and unrealized metrics.
  GET  /trades        - Closed trades journal.
  GET  /equity-curve  - Historical equity snapshot points.
  GET  /decisions     - Reflex decision log with rejections.
  GET  /brain-reports - Market review and research theses.
  GET  /metrics       - Performance metrics summary.
  GET  /llm-cost      - LLM token usage and estimated cost summary.

Write endpoints (protected by Bearer token from environment DASHBOARD_API_TOKEN):
  POST /pause         - Block new position opens and adds.
  POST /kill          - Emergency kill switch: close all positions, cancel orders, pause.
  POST /resume        - Resume agent to RUNNING state.
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from app.common.enums import AgentState
from app.common.logging import get_logger
from app.config.loader import load_config
from app.orchestrator import Orchestrator
from app.portfolio.db import (
    BrainReportModel,
    DecisionModel,
    EquitySnapshotModel,
    RejectionModel,
    TradeModel,
    init_portfolio_db,
)
from app.portfolio.service import PortfolioService

logger = get_logger(__name__)

security_bearer = HTTPBearer(auto_error=False)


def get_expected_token(request: Request | None = None) -> str:
    """Resolve expected API auth token from request app state, environment, or config."""
    if request and hasattr(request.app.state, "expected_token"):
        return str(request.app.state.expected_token)
    return os.environ.get("DASHBOARD_API_TOKEN") or load_config().dashboard_api_token


def verify_api_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(security_bearer),
) -> str:
    """Verify that the request provides a valid Bearer token for write endpoints."""
    expected = get_expected_token(request)
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid API bearer token.",
        )
    return credentials.credentials


def create_api_app(orchestrator: Orchestrator | None = None) -> FastAPI:
    """Factory creating and configuring the FastAPI dashboard server."""
    app = FastAPI(
        title="Autonomous Crypto Portfolio Trading Agent API",
        version="0.1.0",
        description="Local monitoring and control API (Phase 6)",
    )

    # Resolve orchestrator or standalone portfolio service
    orch = orchestrator
    config = orch.config if orch else load_config()
    app.state.expected_token = (
        orch.config.dashboard_api_token
        if orch
        else (os.environ.get("DASHBOARD_API_TOKEN") or config.dashboard_api_token)
    )
    engine = orch.engine if orch else init_portfolio_db("sqlite:///trading_agent.db")
    portfolio_service = (
        orch.portfolio_service
        if orch
        else PortfolioService.load_from_db(engine=engine, broker=None, config=config)  # type: ignore[arg-type]
    )

    # ── Read-Only Endpoints ──────────────────────────────────────────────────

    @app.get("/")
    async def get_root() -> dict[str, Any]:
        """Root endpoint returning API overview and links."""
        return {
            "name": "Autonomous Crypto Portfolio Trading Agent API",
            "status": "online",
            "docs": "/docs",
            "status_endpoint": "/status",
            "dashboard": "http://localhost:8501",
        }

    @app.get("/status")
    async def get_status() -> dict[str, Any]:
        """Current agent execution status, balances, exposure, and breaker flags."""
        state = portfolio_service.get_state()
        agent_st = orch.agent_state.value if orch else portfolio_service.agent_state
        last_cid = (
            orch.portfolio_service.last_cycle_id
            if orch
            else portfolio_service.last_cycle_id
        )
        breaker_p = orch.breaker_paused if orch else False
        kill_flag = (
            orch.portfolio_service.kill_switch
            if orch
            else portfolio_service.kill_switch
        )

        exposure_pct = (
            (sum(p.notional for p in state.positions.values()) / state.equity * 100.0)
            if state.equity > 0
            else 0.0
        )

        return {
            "mode": config.mode.value,
            "agent_state": agent_st,
            "equity": round(state.equity, 2),
            "cash": round(state.cash, 2),
            "peak_equity": round(state.peak_equity, 2),
            "exposure_pct": round(exposure_pct, 2),
            "open_positions_count": len(state.positions),
            "breaker_paused": breaker_p,
            "kill_switch_active": kill_flag,
            "last_cycle_id": last_cid,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    @app.get("/positions")
    async def get_positions() -> list[dict[str, Any]]:
        """List currently open positions with stops, notionals, and unrealized R."""
        positions = portfolio_service.get_state().positions
        results: list[dict[str, Any]] = []

        for sym, pos in positions.items():
            curr_price = (
                await orch.get_latest_price(sym)
                if orch
                else pos.entry_price
            )
            u_pnl = pos.unrealized_pnl(curr_price)
            u_r = pos.unrealized_r(curr_price)

            results.append({
                "symbol": sym,
                "side": pos.side.value,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": curr_price,
                "invalidation_price": pos.invalidation_price,
                "conviction": pos.conviction.value,
                "notional": round(pos.notional, 2),
                "unrealized_pnl": round(u_pnl, 2),
                "unrealized_r": round(u_r, 2),
                "opened_at": pos.opened_at.isoformat(),
            })

        return results

    @app.get("/trades")
    async def get_trades(limit: int = 100) -> list[dict[str, Any]]:
        """List closed trades journal."""
        with portfolio_service.session_factory() as session:
            stmt = select(TradeModel).order_by(TradeModel.exit_time.desc()).limit(limit)
            trades = session.scalars(stmt).all()
            return [
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "realized_pnl": round(t.realized_pnl, 2),
                    "r_multiple": round(t.r_multiple, 2),
                    "fees_total": round(t.fees_total, 2),
                    "exit_reason": t.exit_reason,
                }
                for t in trades
            ]

    @app.get("/equity-curve")
    async def get_equity_curve(limit: int = 500) -> list[dict[str, Any]]:
        """Historical equity snapshots for charting."""
        with portfolio_service.session_factory() as session:
            stmt = select(EquitySnapshotModel).order_by(EquitySnapshotModel.timestamp.asc()).limit(limit)
            points = session.scalars(stmt).all()
            return [
                {
                    "timestamp": p.timestamp.isoformat() if p.timestamp else None,
                    "equity": round(p.equity, 2),
                    "cash": round(p.cash, 2),
                    "exposure": round(p.exposure, 2),
                }
                for p in points
            ]

    @app.get("/decisions")
    async def get_decisions(limit: int = 50) -> list[dict[str, Any]]:
        """Reflex decisions and associated rule rejections."""
        with portfolio_service.session_factory() as session:
            decisions = session.scalars(
                select(DecisionModel).order_by(DecisionModel.timestamp.desc()).limit(limit)
            ).all()

            results: list[dict[str, Any]] = []
            for d in decisions:
                rejections = session.scalars(
                    select(RejectionModel).where(RejectionModel.cycle_id == d.cycle_id)
                ).all()

                parsed = {}
                with contextlib.suppress(Exception):
                    parsed = json.loads(d.parsed_json)

                results.append({
                    "cycle_id": d.cycle_id,
                    "timestamp": d.timestamp.isoformat() if d.timestamp else None,
                    "reasoning": d.reasoning,
                    "parsed_decisions": parsed.get("decisions", []),
                    "rejections": [
                        {
                            "symbol": r.symbol,
                            "action": r.action,
                            "reason": r.reason,
                            "detail": r.detail,
                        }
                        for r in rejections
                    ],
                })
            return results

    @app.get("/brain-reports")
    async def get_brain_reports(limit: int = 10) -> list[dict[str, Any]]:
        """Historical daily research and thesis reports."""
        with portfolio_service.session_factory() as session:
            stmt = select(BrainReportModel).order_by(BrainReportModel.created_at.desc()).limit(limit)
            rows = session.scalars(stmt).all()
            results = []
            for r in rows:
                rep_data = {}
                with contextlib.suppress(Exception):
                    rep_data = json.loads(r.report_json)
                results.append({
                    "report_id": r.report_id,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "market_bias": r.market_bias,
                    "report": rep_data,
                })
            return results

    @app.get("/metrics")
    async def get_metrics() -> dict[str, Any]:
        """Overall portfolio performance metrics."""
        state = portfolio_service.get_state()
        initial = config.starting_equity
        net_pnl = state.equity - initial
        return_pct = (net_pnl / initial * 100.0) if initial > 0 else 0.0

        with portfolio_service.session_factory() as session:
            trades = session.scalars(select(TradeModel)).all()

        wins = [t for t in trades if t.realized_pnl > 0]
        losses = [t for t in trades if t.realized_pnl <= 0]
        win_rate = (len(wins) / len(trades) * 100.0) if trades else 0.0

        max_dd = 0.0
        peak = initial
        for pt in state.equity_history:
            peak = max(peak, pt.equity)
            if peak > 0:
                dd = (peak - pt.equity) / peak * 100.0
                max_dd = max(max_dd, dd)

        return {
            "equity": round(state.equity, 2),
            "initial_equity": initial,
            "current_equity": round(state.equity, 2),
            "net_pnl": round(net_pnl, 2),
            "total_return_pct": round(return_pct, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate, 2),
            "total_fees_paid": round(sum(t.fees_total for t in trades), 2),
        }

    @app.get("/llm-cost")
    async def get_llm_cost() -> dict[str, Any]:
        """Estimated token usage and cost for Reflex and Brain calls."""
        # Query total decisions as metric of cycles
        with portfolio_service.session_factory() as session:
            dec_count = len(session.scalars(select(DecisionModel)).all())
            rep_count = len(session.scalars(select(BrainReportModel)).all())

        # Approximate tokens based on standard prompts
        reflex_tokens = dec_count * 800
        brain_tokens = rep_count * 2500
        total_tokens = reflex_tokens + brain_tokens
        # Estimate $0.15 per 1M input / $0.60 per 1M output for gpt-4o-mini
        cost_est = (total_tokens / 1_000_000) * 0.30

        return {
            "reflex_cycles_logged": dec_count,
            "brain_reports_logged": rep_count,
            "estimated_total_tokens": total_tokens,
            "estimated_total_cost_usd": round(cost_est, 4),
        }

    # ── Authenticated Write Endpoints ────────────────────────────────────────

    @app.post("/pause")
    async def post_pause(_: str = Depends(verify_api_token)) -> dict[str, Any]:
        """Pause agent from taking new opens (Bearer token required)."""
        if orch:
            orch.breaker_paused = True
            orch.agent_state = AgentState.PAUSED_BREAKER
            await orch.portfolio_service.set_agent_state(AgentState.PAUSED_BREAKER.value)
        else:
            await portfolio_service.set_agent_state(AgentState.PAUSED_BREAKER.value)
        return {"status": "success", "agent_state": AgentState.PAUSED_BREAKER.value}

    @app.post("/kill")
    async def post_kill(_: str = Depends(verify_api_token)) -> dict[str, Any]:
        """Emergency kill switch: close all positions and cancel orders (Bearer token required)."""
        if orch:
            await orch.trigger_kill_switch("API /kill endpoint invoked")
        else:
            kill_path = Path(config.kill_switch_file)
            kill_path.write_text("API /kill endpoint invoked", encoding="utf-8")
            await portfolio_service.set_agent_state(AgentState.PAUSED_KILL.value)
        return {"status": "success", "agent_state": AgentState.PAUSED_KILL.value}

    @app.post("/resume")
    async def post_resume(_: str = Depends(verify_api_token)) -> dict[str, Any]:
        """Resume trading from pause or kill switch (Bearer token required)."""
        if orch:
            try:
                await orch.resume()
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            kill_path = Path(config.kill_switch_file)
            kill_path.unlink(missing_ok=True)
            await portfolio_service.set_agent_state(AgentState.RUNNING.value)
        return {"status": "success", "agent_state": AgentState.RUNNING.value}

    return app


# Default ASGI application instance for uvicorn app.dashboard.api:app
app = create_api_app()
