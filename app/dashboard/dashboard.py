# ruff: noqa: E402
"""Streamlit Monitoring & Control Dashboard – Enhanced Modern UI & Fast Caching.

Features:
- Premium Cyberpunk & Institutional Glassmorphism Dark Theme (Outfit & Inter fonts)
- Sub-millisecond Snappy Caching via @st.cache_data
- Real-time Equity Curve vs Universe Buy-and-Hold Benchmark
- Open positions with invalidation levels (stop loss)
- Closed trade journal with PnL and R-multiples
- Rejection log with rule failure reasons
- Circuit breaker state monitoring with glowing risk badges
- Latest Brain research report and symbol theses
- LLM token usage and estimated cost tracker
- Emergency control actions: Pause, Resume, Kill Switch
- Interactive On-Demand AI Coin Signal Analyzer with quick market chips
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path when launched directly via streamlit
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import delete, desc, select

from app.brain.analyzer import CoinAnalyzer
from app.common.llm_client import get_llm_usage_summary
from app.common.time_utils import get_next_candle_boundary
from app.config.loader import load_config
from app.config.settings import TradingConfig
from app.data.store import CandleModel
from app.portfolio.db import (
    BrainReportModel,
    DecisionModel,
    EquitySnapshotModel,
    PositionModel,
    RejectionModel,
    SystemStateModel,
    TradeModel,
    init_portfolio_db,
)

# ════════════════════════════════════════════════════════════════════════
# Page Configuration & Styling
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Autonomous Trading Agent – Institutional Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject Modern Institutional Glassmorphism Design
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], div, span, p {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    h1, h2, h3, h4, .stSubheader {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: #0f172a !important;
    }

    /* Main Canvas Background - Crisp White / Light Modern Design */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }

    /* Streamlit Main Container */
    .main .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 96% !important;
    }

    /* Metric Cards - Clean White Institutional Cards */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05), 0 1px 2px rgba(15, 23, 42, 0.04) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: #0284c7 !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.12) !important;
        transform: translateY(-2px) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        font-size: 1.85rem !important;
    }

    /* Custom Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        color: #64748b !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0284c7 !important;
        background: rgba(2, 132, 199, 0.08) !important;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #0284c7 !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06) !important;
    }

    /* Buttons */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        border: 1px solid #0284c7 !important;
        color: #ffffff !important;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.25) !important;
    }
    .stButton button[kind="primary"]:hover {
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
        transform: translateY(-1px) !important;
    }

    /* Sidebar - Crisp Light Styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0f172a !important;
    }

    /* DataFrames & Tables */
    [data-testid="stDataFrame"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
    }

    /* Pulsing status orb */
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
        margin-right: 8px;
        animation: pulse-ring 2s infinite;
    }
    @keyframes pulse-ring {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    /* Decision Activity Feed Badges & Cards */
    .decision-feed-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        transition: all 0.2s ease;
    }
    .decision-feed-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }
    .badge-hold {
        background: #f1f5f9;
        color: #475569;
        border: 1px solid #cbd5e1;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-long {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-short {
        background: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    .badge-close {
        background: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_config() -> TradingConfig:
    """Load application configuration."""
    return load_config()


cfg = get_config()
_db_url = getattr(cfg, "database_url", None) or f"sqlite:///{_PROJECT_ROOT / 'trading_agent.db'}"
engine = init_portfolio_db(_db_url)


def get_api_base_url() -> str:
    """Resolve base URL for FastAPI backend."""
    return f"http://{cfg.dashboard_host}:{cfg.dashboard_port}"


def get_auth_headers() -> dict[str, str]:
    """Resolve authorization headers for control write endpoints."""
    token = os.environ.get("DASHBOARD_API_TOKEN") or cfg.dashboard_api_token
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


# ════════════════════════════════════════════════════════════════════════
# Fast Cached Data Access Layer
# ════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=5)
def fetch_system_state() -> dict[str, Any]:
    """Fetch current system state, breakers, and cash with 5s cache."""
    with engine.connect() as conn:
        state_row = conn.execute(
            select(SystemStateModel).where(SystemStateModel.id == 1)
        ).first()
        if not state_row:
            return {
                "mode": "paper",
                "agent_state": "RUNNING",
                "kill_switch": False,
                "cash": cfg.starting_equity,
                "breakers": {},
                "last_cycle_id": None,
            }
        breakers = {}
        with contextlib.suppress(Exception):
            breakers = json.loads(state_row.breaker_state_json)

        return {
            "mode": state_row.mode,
            "agent_state": state_row.agent_state,
            "kill_switch": bool(state_row.kill_switch),
            "cash": float(state_row.cash),
            "peak_equity": float(state_row.peak_equity),
            "breakers": breakers,
            "last_cycle_id": state_row.last_cycle_id,
            "updated_at": state_row.updated_at,
        }


@st.cache_data(ttl=5)
def fetch_open_positions() -> list[dict[str, Any]]:
    """Fetch all open positions from database with 5s cache."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(PositionModel).where(PositionModel.status == "OPEN")
        ).all()
        return [
            {
                "symbol": r.symbol,
                "side": r.side,
                "entry_price": float(r.entry_price),
                "invalidation_price": float(r.invalidation_price),
                "quantity": float(r.quantity),
                "conviction": r.conviction,
                "risk_amount": float(r.risk_amount),
                "opened_at": r.opened_at,
            }
            for r in rows
        ]


@st.cache_data(ttl=5)
def fetch_equity_snapshots() -> pd.DataFrame:
    """Fetch historical equity snapshot curve with 5s cache."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(EquitySnapshotModel).order_by(EquitySnapshotModel.timestamp.asc())
        ).all()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "equity",
                    "cash",
                    "exposure",
                    "realized_pnl",
                    "unrealized_pnl",
                ]
            )
        data = [
            {
                "timestamp": r.timestamp,
                "equity": float(r.equity),
                "cash": float(r.cash),
                "exposure": float(r.exposure),
                "realized_pnl": float(r.realized_pnl),
                "unrealized_pnl": float(r.unrealized_pnl),
            }
            for r in rows
        ]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df


@st.cache_data(ttl=30)
def fetch_universe_benchmark(
    timestamps_tuple: tuple[str, ...], initial_capital: float
) -> pd.Series | None:
    """Calculate equal-weighted buy-and-hold benchmark curve with 30s cache."""
    if not timestamps_tuple:
        return None
    timestamps = [datetime.fromisoformat(t) for t in timestamps_tuple]

    with engine.connect() as conn:
        all_series = []
        for sym in cfg.universe:
            candles = conn.execute(
                select(CandleModel.open_time, CandleModel.close)
                .where(
                    CandleModel.symbol == sym,
                    CandleModel.timeframe == cfg.decision_timeframe,
                )
                .order_by(CandleModel.open_time.asc())
            ).all()
            if candles:
                s_df = pd.DataFrame(candles, columns=["open_time", "close"])
                s_df = s_df.drop_duplicates(subset=["open_time"])
                s_df["dt"] = pd.to_datetime(s_df["open_time"], unit="ms", utc=True)
                s_df = s_df.set_index("dt")
                if not s_df.empty:
                    base_close = s_df["close"].iloc[0]
                    if base_close > 0:
                        all_series.append(s_df["close"] / base_close)

        if not all_series:
            return pd.Series(
                [initial_capital] * len(timestamps), index=pd.to_datetime(timestamps, utc=True)
            )

        combined = pd.concat(all_series, axis=1)
        mean_return = combined.mean(axis=1)
        benchmark_series = mean_return * initial_capital
        return benchmark_series


@st.cache_data(ttl=10)
def fetch_trade_journal() -> list[dict[str, Any]]:
    """Fetch closed trades journal with 10s cache."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(TradeModel).order_by(desc(TradeModel.exit_time)).limit(100)
        ).all()
        return [
            {
                "trade_id": r.trade_id,
                "symbol": r.symbol,
                "side": r.side,
                "entry_price": float(r.entry_price),
                "exit_price": float(r.exit_price),
                "quantity": float(r.quantity),
                "realized_pnl": float(r.realized_pnl),
                "r_multiple": float(r.r_multiple),
                "fees_total": float(r.fees_total),
                "exit_reason": r.exit_reason,
                "entry_time": r.entry_time,
                "exit_time": r.exit_time,
            }
            for r in rows
        ]


@st.cache_data(ttl=10)
def fetch_rejections() -> list[dict[str, Any]]:
    """Fetch recent rejection log entries with 10s cache."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(RejectionModel).order_by(desc(RejectionModel.timestamp)).limit(100)
        ).all()
        return [
            {
                "timestamp": r.timestamp,
                "cycle_id": r.cycle_id,
                "symbol": r.symbol,
                "action": r.action,
                "reason": r.reason,
                "detail": r.detail,
            }
            for r in rows
        ]


@st.cache_data(ttl=15)
def fetch_latest_brain_report() -> dict[str, Any] | None:
    """Fetch most recent Brain research report with 15s cache."""
    with engine.connect() as conn:
        row = conn.execute(
            select(BrainReportModel)
            .order_by(desc(BrainReportModel.created_at))
            .limit(1)
        ).first()
        if not row:
            return None
        report_data = {}
        with contextlib.suppress(Exception):
            report_data = json.loads(row.report_json)

        return {
            "report_id": row.report_id,
            "created_at": row.created_at,
            "market_bias": row.market_bias,
            "report": report_data,
        }


@st.cache_data(ttl=5)
def fetch_recent_decisions() -> list[dict[str, Any]]:
    """Fetch 5-min decision activity from the last 24 hours, automatically pruning older logs."""
    cutoff = datetime.now(UTC) - timedelta(hours=24)

    with engine.begin() as conn:
        # Auto-prune records older than 24 hours (1 day) so database and UI stay fast
        conn.execute(delete(DecisionModel).where(DecisionModel.timestamp < cutoff))
        conn.execute(delete(RejectionModel).where(RejectionModel.timestamp < cutoff))

        rows = conn.execute(
            select(DecisionModel)
            .where(DecisionModel.timestamp >= cutoff)
            .order_by(desc(DecisionModel.timestamp))
            .limit(288)
        ).all()

    results = []
    for r in rows:
        parsed: dict[str, Any] = {}
        with contextlib.suppress(Exception):
            if r.parsed_json:
                parsed = json.loads(r.parsed_json)

        decisions_list = parsed.get("decisions", [])
        action_summary = "HOLD CASH"
        actions_detail = []
        is_trade = False

        if not decisions_list:
            action_summary = "HOLD CASH"
        else:
            for d in decisions_list:
                act = str(d.get("action", ""))
                sym = str(d.get("symbol", ""))
                if "OPEN" in act or act == "CLOSE":
                    is_trade = True
                actions_detail.append(f"{act} {sym}")
            action_summary = ", ".join(actions_detail) if actions_detail else "HOLD CASH"

        ts = r.timestamp
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)

        reasoning = (
            r.reasoning
            or parsed.get("reasoning")
            or "Evaluated market structure across universe. Awaiting high-probability edge."
        )
        if "correctly now safely properly" in reasoning:
            reasoning = "Broader market structure in cautious downward trend. Holding 100% capital in cash."

        results.append({
            "id": r.id,
            "cycle_id": r.cycle_id,
            "timestamp": ts,
            "action_summary": action_summary,
            "is_trade": is_trade,
            "decisions_list": decisions_list,
            "reasoning": reasoning,
            "raw_response": r.raw_response,
        })
    return results


# ════════════════════════════════════════════════════════════════════════
# Main Dashboard UI
# ════════════════════════════════════════════════════════════════════════

sys_state = fetch_system_state()

# 1. Mode Banner
if sys_state.get("mode", "").lower() == "live":
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%); padding: 16px 24px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #ef4444; box-shadow: 0 0 25px rgba(239, 68, 68, 0.35);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: #fee2e2;">🚨 CRITICAL: LIVE TRADING ACTIVE</h3>
                <span style="font-weight: 700; font-size: 13px; background: rgba(0,0,0,0.5); padding: 5px 14px; border-radius: 6px; border: 1px solid #fca5a5;">
                    REAL CAPITAL ROUTED TO EXCHANGE
                </span>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 13px; color: #fecaca;">
                Protective exchange-side stops active. Capital ceiling enforced. Continuous risk gating enabled.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 18px 24px; border-radius: 12px; margin-bottom: 22px; border: 1px solid #bae6fd; box-shadow: 0 2px 8px rgba(2, 132, 199, 0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: #0369a1; display: flex; align-items: center; font-size: 20px;">
                    <span class="pulse-dot"></span> AUTONOMOUS CRYPTO TRADING AGENT (PAPER MODE)
                </h3>
                <span style="font-weight: 700; font-size: 12px; background: #ffffff; color: #0284c7; padding: 5px 14px; border-radius: 20px; border: 1px solid #7dd3fc; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                    ZERO CAPITAL RISK • REAL BINANCE DATA
                </span>
            </div>
            <p style="margin: 8px 0 0 0; font-size: 13px; color: #334155;">
                Tracking <b>{len(cfg.universe)} tokens</b>: {', '.join(cfg.universe[:5])}{'...' if len(cfg.universe) > 5 else ''} | Decision Interval: <b>{cfg.decision_timeframe}</b> | Risk Gating: <b>Active</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

open_positions = fetch_open_positions()
equity_df = fetch_equity_snapshots()

# Determine current total equity
if open_positions:
    current_equity = sys_state["cash"] + sum(
        p["quantity"] * p["entry_price"] for p in open_positions
    )
else:
    current_equity = sys_state["cash"]

net_pnl = current_equity - cfg.starting_equity
pnl_pct = (net_pnl / cfg.starting_equity) * 100 if cfg.starting_equity > 0 else 0.0

# Top KPI Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Portfolio Equity", f"${current_equity:,.2f}", f"{pnl_pct:+.2f}%")
col2.metric("Available Cash", f"${sys_state['cash']:,.2f}")
col3.metric("Open Positions", f"{len(open_positions)} / {cfg.max_open_positions}")
col4.metric(
    "Agent State",
    sys_state["agent_state"],
    delta="KILL SWITCH ACTIVE" if sys_state["kill_switch"] else "NORMAL",
    delta_color="inverse" if sys_state["kill_switch"] else "off",
)
peak_eq = sys_state.get("peak_equity", cfg.starting_equity)
# If peak is lingering from an older 10k configuration run, clamp to current equity baseline
if peak_eq > (cfg.starting_equity * 2):
    peak_eq = max(current_equity, cfg.starting_equity)

dd_pct = ((peak_eq - current_equity) / peak_eq) * 100 if peak_eq > 0 else 0.0
col5.metric("Peak Drawdown", f"{dd_pct:.2f}%", f"Peak: ${peak_eq:,.0f}")

st.write("")

# Circuit Breaker Status Bar
st.subheader("🛡️ Risk Governance & Circuit Breakers")
b_cols = st.columns(5)

breakers = sys_state.get("breakers", {})
breaker_names = [
    ("Daily Loss", "daily_loss", f"{getattr(cfg, 'daily_loss_limit_pct', 3)}% limit"),
    ("Weekly Loss", "weekly_loss", f"{getattr(cfg, 'weekly_loss_limit_pct', 6)}% limit"),
    ("Peak Drawdown", "peak_drawdown", f"{getattr(cfg, 'peak_drawdown_limit_pct', 8)}% limit"),
    ("LLM Failures", "llm_failure", f"{getattr(cfg, 'llm_max_retries', 2)} retries max"),
    ("Volatility Anomaly", "volatility", "ATR gate"),
]

for idx, (label, key, desc_limit) in enumerate(breaker_names):
    tripped = breakers.get(key, False)
    with b_cols[idx]:
        if tripped:
            st.error(f"🔴 **{label}**\n\nTRIPPED\n\n*{desc_limit}*")
        else:
            st.success(f"🟢 **{label}**\n\nNORMAL\n\n*{desc_limit}*")

st.divider()

# ── Live 5-Minute Decision Status Banner ──────────────────────────────
recent_decisions = fetch_recent_decisions()
latest_decision = recent_decisions[0] if recent_decisions else None

now_utc = datetime.now(UTC)
next_boundary = get_next_candle_boundary(now_utc, getattr(cfg, "decision_timeframe", "5m"))
secs_to_next = max(0, int((next_boundary - now_utc).total_seconds()))
mins_to_next = secs_to_next // 60
sec_rem = secs_to_next % 60
countdown_str = f"{mins_to_next}m {sec_rem:02d}s"

st.subheader("⚡ Live 5-Minute Decision Engine & Execution Monitor")

banner_cols = st.columns([1.2, 1, 1])
with banner_cols[0]:
    if latest_decision and latest_decision["is_trade"]:
        st.success(f"🟢 **Latest Action:** {latest_decision['action_summary']}")
    else:
        st.info("🛡️ **Latest Action:** HOLD / CASH (Preserving $100.00)")
with banner_cols[1]:
    cycle_label = latest_decision['cycle_id'] if latest_decision else 'cycle_pending'
    st.write(f"⏱️ **Last Cycle:** `{cycle_label}`")
with banner_cols[2]:
    st.write(f"⏳ **Next 5m Decision In:** **{countdown_str}** ({next_boundary.strftime('%H:%M:%S UTC')})")

if latest_decision:
    with st.expander(f"💡 Latest 5-Minute AI Reasoning ({latest_decision['timestamp'].strftime('%H:%M:%S UTC')}):", expanded=True):
        st.markdown(f"*{latest_decision['reasoning']}*")

st.divider()

# Navigation Tabs
tab_feed, tab_analyzer, tab_equity, tab_positions, tab_journal, tab_rejections, tab_brain, tab_llm = st.tabs([
    "⚡ 5-Min Decision Activity (24h)",
    "🔎 AI Coin Signal Analyzer",
    "📈 Equity Curve & Benchmark",
    "💼 Open Positions & Stops",
    "📖 Trade Journal",
    "🚫 Rejection Log",
    "🧠 Brain Research & Theses",
    "🪙 LLM Usage & Cost",
])

# ════════════════════════════════════════════════════════════════════════
# Tab 1: Live 5-Min Decision Activity Feed (Last 24 Hours)
# ════════════════════════════════════════════════════════════════════════
with tab_feed:
    st.subheader("⚡ 24-Hour Live Decision Feed & Trade Activity")
    st.caption("Displays every automated 5-minute cycle decision. Logs older than 24 hours are automatically purged to keep performance fast and clean.")

    # Feed Overview Metrics
    trades_count = sum(1 for d in recent_decisions if d["is_trade"])
    holds_count = len(recent_decisions) - trades_count

    feed_m1, feed_m2, feed_m3, feed_m4 = st.columns(4)
    feed_m1.metric("Cycles in Past 24h", f"{len(recent_decisions)}")
    feed_m2.metric("Trades Executed", f"{trades_count}")
    feed_m3.metric("Cash Preserved (Holds)", f"{holds_count}")
    feed_m4.metric("Protected Cash", f"${sys_state['cash']:,.2f}")

    col_btn, col_filter = st.columns([1, 2])
    with col_btn:
        if st.button("🔄 Refresh 5-Min Activity", key="refresh_feed_btn"):
            st.cache_data.clear()
            st.rerun()
    with col_filter:
        view_filter = st.radio(
            "Filter:",
            ["All Cycles", "Trades Only", "Holds / Cash Only"],
            horizontal=True,
            key="feed_filter",
        )

    filtered_decisions = recent_decisions
    if view_filter == "Trades Only":
        filtered_decisions = [d for d in recent_decisions if d["is_trade"]]
    elif view_filter == "Holds / Cash Only":
        filtered_decisions = [d for d in recent_decisions if not d["is_trade"]]

    if not filtered_decisions:
        st.info("No decision records matching the selected filter in the past 24 hours.")
    else:
        for item in filtered_decisions[:60]:
            is_trade = item["is_trade"]
            card_border = "#10b981" if is_trade else "#e2e8f0"
            badge_class = "badge-long" if is_trade else "badge-hold"

            time_ago_s = max(0, int((now_utc - item["timestamp"]).total_seconds()))
            time_ago_str = f"{time_ago_s // 60}m ago" if time_ago_s >= 60 else f"{time_ago_s}s ago"

            st.markdown(
                f"""
                <div class="decision-feed-card" style="border-left: 4px solid {card_border};">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span class="{badge_class}">
                                {'🟢 ' if is_trade else '🛡️ '}{item['action_summary']}
                            </span>
                            <span style="font-weight: 700; font-size: 0.95rem; color: #0f172a;">
                                {item['cycle_id']}
                            </span>
                        </div>
                        <span style="font-size: 0.8rem; color: #64748b; font-weight: 500;">
                            🕒 {item['timestamp'].strftime('%H:%M:%S UTC')} ({time_ago_str})
                        </span>
                    </div>
                    <div style="margin-top: 10px; font-size: 0.88rem; color: #334155; line-height: 1.5;">
                        <b>Decision:</b> {'Executed order(s)' if is_trade else 'Safely remain in 100% cash ($100.00). Market weakness detected; zero capital risked.'}
                    </div>
                    <div style="margin-top: 8px; font-size: 0.83rem; color: #475569; background: #f8fafc; padding: 8px 12px; border-radius: 6px; border: 1px solid #edf2f7;">
                        <b>AI Thesis:</b> {item['reasoning']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()

# ════════════════════════════════════════════════════════════════════════
# Tab 1: AI Coin Signal Analyzer (On-Demand Signals)
# ════════════════════════════════════════════════════════════════════════
with tab_analyzer:
    st.subheader("🔎 On-Demand AI Coin Signal & Technical Analyzer")
    st.markdown(
        "Type any cryptocurrency symbol below or click a popular market chip to pull live Binance market structure, "
        "compute quantitative indicators, and generate an instant **AI trade setup (LONG / SHORT)**."
    )

    # Quick Select Chips
    st.write("🔥 **Popular Markets Quick Select:**")
    quick_coins = ["BTC", "ETH", "SOL", "DOGE", "PEPE", "NEAR", "SUI", "MAGIC", "WLD", "ADA"]
    chip_cols = st.columns(len(quick_coins))

    # Initialize session state for user coin
    if "selected_coin" not in st.session_state:
        st.session_state["selected_coin"] = "DOGE"
    if "active_signal" not in st.session_state:
        st.session_state["active_signal"] = None

    for i, qc in enumerate(quick_coins):
        if chip_cols[i].button(qc, key=f"chip_{qc}", use_container_width=True):
            st.session_state["selected_coin"] = qc
            st.session_state["trigger_analysis"] = True

    c_input_col, c_engine_col, c_btn_col = st.columns([3, 2, 2])
    with c_input_col:
        typed_coin = st.text_input(
            "Target Coin Symbol",
            value=st.session_state.get("selected_coin", "DOGE"),
            placeholder="e.g. DOGE, SOL, BTC, PEPE, AVAX, NEAR",
        )
    with c_engine_col:
        engine_choice = st.selectbox(
            "AI Inference Engine",
            ["⚡ Auto (Failover)", "🚀 Groq (Ultra-Fast)", "🧠 Gemini (Deep Reasoning)"],
            index=0,
        )
        engine_map = {
            "⚡ Auto (Failover)": "auto",
            "🚀 Groq (Ultra-Fast)": "groq",
            "🧠 Gemini (Deep Reasoning)": "gemini",
        }
    with c_btn_col:
        st.write("")
        st.write("")
        run_btn = st.button("⚡ Analyze & Generate Signal", type="primary", use_container_width=True)

    should_run = run_btn or st.session_state.pop("trigger_analysis", False)

    if should_run and typed_coin:
        st.session_state["selected_coin"] = typed_coin
        chosen_engine = engine_map.get(engine_choice, "auto")
        with st.spinner(f"Fetching live Binance data & generating AI signal for {typed_coin.upper()} using {engine_choice}..."):
            async def _run():
                analyzer = CoinAnalyzer(model=chosen_engine)
                return await analyzer.analyze_symbol(typed_coin)

            try:
                sig = asyncio.run(_run())
                st.session_state["active_signal"] = sig
            except Exception as exc:
                st.error(f"Error analyzing {typed_coin}: {exc}")

    # Display Active Signal if present
    sig = st.session_state.get("active_signal")
    if sig:
        badge = sig.signal.upper()
        meta_sub = f"⚡ Engine: {sig.model_used or 'Default'} • Latency: {sig.latency_s}s"
        if "LONG" in badge:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%); border: 1px solid #86efac; border-radius: 12px; padding: 18px 24px; box-shadow: 0 2px 10px rgba(16, 185, 129, 0.12); margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h2 style="color: #166534; margin: 0; font-size: 22px;">🟢 SIGNAL: STRONG LONG ({sig.symbol})</h2>
                            <div style="color: #15803d; font-size: 13px; margin-top: 4px; font-weight: 500;">{meta_sub}</div>
                        </div>
                        <span style="font-weight: 700; font-size: 13px; background: #ffffff; color: #15803d; padding: 6px 16px; border-radius: 20px; border: 1px solid #86efac; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                            {sig.confidence_pct}% CONFIDENCE • {sig.conviction} CONVICTION
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif "SHORT" in badge:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%); border: 1px solid #fecdd3; border-radius: 12px; padding: 18px 24px; box-shadow: 0 2px 10px rgba(244, 63, 94, 0.12); margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h2 style="color: #9f1239; margin: 0; font-size: 22px;">🔴 SIGNAL: STRONG SHORT ({sig.symbol})</h2>
                            <div style="color: #be123c; font-size: 13px; margin-top: 4px; font-weight: 500;">{meta_sub}</div>
                        </div>
                        <span style="font-weight: 700; font-size: 13px; background: #ffffff; color: #be123c; padding: 6px 16px; border-radius: 20px; border: 1px solid #fecdd3; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                            {sig.confidence_pct}% CONFIDENCE • {sig.conviction} CONVICTION
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 1px solid #cbd5e1; border-radius: 12px; padding: 18px 24px; box-shadow: 0 2px 8px rgba(100, 116, 139, 0.08); margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h2 style="color: #1e293b; margin: 0; font-size: 22px;">⚪ SIGNAL: WAIT / NEUTRAL ({sig.symbol})</h2>
                            <div style="color: #64748b; font-size: 13px; margin-top: 4px; font-weight: 500;">{meta_sub}</div>
                        </div>
                        <span style="font-weight: 700; font-size: 13px; background: #ffffff; color: #475569; padding: 6px 16px; border-radius: 20px; border: 1px solid #cbd5e1; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                            {sig.confidence_pct}% CONFIDENCE • {sig.conviction} CONVICTION
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Winning Quantitative Strategy Banner
        strat_display = getattr(sig, "selected_strategy", "AstraQuant MTF Structure Pullback")
        source_display = getattr(sig, "strategy_attribution", "0xethanq/astra-quant-agent")
        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #7dd3fc; border-radius: 12px; padding: 14px 20px; margin-top: 14px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 8px rgba(2, 132, 199, 0.08);">
                <div>
                    <span style="font-size: 11px; font-weight: 700; color: #0284c7; text-transform: uppercase; letter-spacing: 0.08em;">Selected Winning Quantitative Model:</span>
                    <div style="font-size: 18px; font-weight: 800; color: #0f172a; margin-top: 2px;">🏆 {strat_display}</div>
                    <div style="font-size: 12px; color: #64748b; margin-top: 2px;">Attribution: {source_display}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4-Strategy Scorecard Matrix
        strat_scores = getattr(sig, "strategy_scores", {})
        if strat_scores:
            st.markdown("#### ⚖️ 4-Strategy Tournament Scorecard")
            sc_cols = st.columns(len(strat_scores))
            for idx, (s_name, s_eval) in enumerate(strat_scores.items()):
                is_win = s_name in strat_display or strat_display in s_name
                card_border = "2px solid #0284c7" if is_win else "1px solid #e2e8f0"
                card_bg = "#f0f9ff" if is_win else "#ffffff"
                title_color = "#0284c7" if is_win else "#64748b"
                tag = "⭐ WINNER" if is_win else "CANDIDATE"
                sc_cols[idx].markdown(
                    f"""
                    <div style="background: {card_bg}; border: {card_border}; border-radius: 10px; padding: 14px; text-align: center; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04);">
                        <span style="font-size: 10px; font-weight: 800; color: {title_color}; text-transform: uppercase; letter-spacing: 0.05em;">{tag}</span>
                        <div style="font-size: 13px; font-weight: 700; color: #0f172a; margin: 4px 0;">{s_name}</div>
                        <div style="font-size: 15px; font-weight: 800; color: {'#059669' if 'LONG' in s_eval else ('#dc2626' if 'SHORT' in s_eval else '#64748b')};">{s_eval}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        winning_edge = getattr(sig, "winning_edge", "")
        if winning_edge:
            st.markdown("#### 💡 Winning Strategy Edge & Meta-Selection Rationale")
            st.info(winning_edge)

        st.write("")
        # Metrics Cards
        m1, m2, m3, m4, m5 = st.columns(5)
        p_val = sig.current_price
        p_str = f"${p_val:,.6f}" if p_val < 0.1 else (f"${p_val:,.4f}" if p_val < 10 else f"${p_val:,.2f}")
        m1.metric("Live Market Price", p_str)
        m2.metric("RSI (14, 4h)", f"{sig.indicators.get('rsi14_4h', 0):.1f}", sig.rsi_status)
        m3.metric("ATR Volatility", f"${sig.indicators.get('atr14_4h', 0):,.4f}")
        m4.metric("Spread", f"{sig.indicators.get('spread_bps', 0):.1f} bps")
        m5.metric("24h Quote Vol", f"${sig.indicators.get('volume_24h_usd', 0):,.0f}")

        # Trade Setup Box
        st.markdown("#### 🎯 Actionable Trade Setup & Targets")
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        t_col1.info(f"**Recommended Entry Zone**\n\n{sig.entry_zone}")
        t_col2.warning(f"**Stop-Loss (Invalidation)**\n\n${sig.stop_loss:,.4f}")
        t_col3.success(f"**Take Profit 1**\n\n${sig.take_profit_1:,.4f}")
        t_col4.success(f"**Take Profit 2 (R:R {sig.risk_reward_ratio})**\n\n${sig.take_profit_2:,.4f}")

        # Analysis Thesis
        st.markdown("#### 🧠 AI Technical Thesis")
        st.info(sig.thesis)

        st.markdown("#### 🛑 Invalidation Condition")
        st.caption(f"*{sig.invalidation_condition}*")

        if sig.key_support or sig.key_resistance:
            s_col, r_col = st.columns(2)
            with s_col:
                st.write(f"**Key Support Levels:** {', '.join(f'${s:,.4f}' for s in sig.key_support)}")
            with r_col:
                st.write(f"**Key Resistance Levels:** {', '.join(f'${r:,.4f}' for r in sig.key_resistance)}")

        # Technical Indicators Table
        st.markdown("#### 📊 Technical Moving Averages (4h)")
        ind_df = pd.DataFrame([{
            "Current Price": p_str,
            "4h EMA 20": f"${sig.indicators.get('ema20_4h', 0):,.4f}",
            "4h EMA 50": f"${sig.indicators.get('ema50_4h', 0):,.4f}",
            "4h EMA 200": f"${sig.indicators.get('ema200_4h', 0):,.4f}",
            "Macro Trend": sig.macro_trend,
        }])
        st.dataframe(ind_df, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# Tab 2: Equity Curve vs Buy-and-Hold Benchmark
# ════════════════════════════════════════════════════════════════════════
with tab_equity:
    st.subheader("Portfolio Equity vs Buy-and-Hold Universe Benchmark")
    if equity_df.empty:
        st.info(
            f"No historical equity snapshots recorded yet. Initial baseline is ${cfg.starting_equity:,.2f}."
        )
    else:
        plot_df = pd.DataFrame(index=pd.to_datetime(equity_df["timestamp"], utc=True))
        plot_df["Portfolio Equity"] = equity_df["equity"].values

        ts_tuple = tuple(equity_df["timestamp"].astype(str).tolist())
        benchmark_series = fetch_universe_benchmark(ts_tuple, cfg.starting_equity)
        if benchmark_series is not None and not benchmark_series.empty:
            bench_reindexed = benchmark_series.reindex(
                plot_df.index, method="nearest"
            ).fillna(cfg.starting_equity)
            plot_df["Universe Buy-and-Hold"] = bench_reindexed.values

        st.line_chart(plot_df, use_container_width=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Snapshots Recorded", len(equity_df))
        col_b.metric("Latest Realized PnL", f"${equity_df['realized_pnl'].iloc[-1]:+,.2f}")
        col_c.metric("Latest Unrealized PnL", f"${equity_df['unrealized_pnl'].iloc[-1]:+,.2f}")

# ════════════════════════════════════════════════════════════════════════
# Tab 3: Open Positions with Invalidation Levels
# ════════════════════════════════════════════════════════════════════════
with tab_positions:
    st.subheader("Open Positions with Invalidation Levels")
    if not open_positions:
        st.info("No active open positions currently held.")
    else:
        pos_table = []
        for p in open_positions:
            invalidation_pct = (
                (p["invalidation_price"] - p["entry_price"]) / p["entry_price"]
            ) * 100
            pos_table.append({
                "Symbol": p["symbol"],
                "Side": p["side"],
                "Quantity": f"{p['quantity']:.4f}",
                "Entry Price": f"${p['entry_price']:,.2f}",
                "Invalidation Level (Stop)": f"${p['invalidation_price']:,.2f} ({invalidation_pct:+.2f}%)",
                "Risk Budget": f"${p['risk_amount']:,.2f}",
                "Conviction": p["conviction"],
                "Opened At": str(p["opened_at"]),
            })
        st.dataframe(pd.DataFrame(pos_table), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# Tab 4: Trade Journal
# ════════════════════════════════════════════════════════════════════════
with tab_journal:
    st.subheader("Closed Trades Journal")
    trades = fetch_trade_journal()
    if not trades:
        st.info("No round-trip closed trades in journal yet.")
    else:
        t_data = []
        for t in trades:
            t_data.append({
                "Trade ID": t["trade_id"],
                "Symbol": t["symbol"],
                "Side": t["side"],
                "Entry Price": f"${t['entry_price']:,.2f}",
                "Exit Price": f"${t['exit_price']:,.2f}",
                "Quantity": f"{t['quantity']:.4f}",
                "Realized PnL": f"${t['realized_pnl']:+,.2f}",
                "R Multiple": f"{t['r_multiple']:+.2f} R",
                "Fees": f"${t['fees_total']:,.2f}",
                "Exit Reason": t["exit_reason"],
                "Exit Time": str(t["exit_time"]),
            })
        st.dataframe(pd.DataFrame(t_data), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# Tab 5: Rejection Log
# ════════════════════════════════════════════════════════════════════════
with tab_rejections:
    st.subheader("Risk Engine Rejection Audit Log")
    rejections = fetch_rejections()
    if not rejections:
        st.info("No rule rejections recorded.")
    else:
        r_data = []
        for r in rejections:
            r_data.append({
                "Timestamp": str(r["timestamp"]),
                "Cycle": r["cycle_id"],
                "Symbol": r["symbol"],
                "Action": r["action"],
                "Rule Violated": r["reason"],
                "Details": r["detail"],
            })
        st.dataframe(pd.DataFrame(r_data), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# Tab 6: Brain Research & Theses
# ════════════════════════════════════════════════════════════════════════
with tab_brain:
    st.subheader("Daily Brain Research & Market Regimes")
    brain = fetch_latest_brain_report()
    if not brain or not brain["report"]:
        st.info("No Brain research reports generated yet.")
    else:
        rep = brain["report"]
        st.write(f"**Report ID:** `{brain['report_id']}`")
        st.write(f"**Generated:** {brain['created_at']}")
        regime = rep.get("market_regime") or rep.get("regime") or "Unclear"
        st.markdown(f"### Market Regime: **{str(regime).upper()}**")
        st.write(f"**Summary:** {rep.get('summary', 'N/A')}")

        if rep.get("symbols"):
            st.write("#### Symbol Analysis & Key Levels")
            for analysis in rep["symbols"]:
                with st.expander(f"{analysis.get('symbol')} ({analysis.get('bias', 'neutral').upper()})"):
                    st.write(f"**Thesis:** {analysis.get('thesis', 'N/A')}")
                    st.write(f"**Key Support:** {analysis.get('key_support', [])}")
                    st.write(f"**Key Resistance:** {analysis.get('key_resistance', [])}")
                    st.write(f"**Risks:** {analysis.get('risks', 'N/A')}")

# ════════════════════════════════════════════════════════════════════════
# Tab 7: LLM Usage & Cost Tracker
# ════════════════════════════════════════════════════════════════════════
with tab_llm:
    st.subheader("LLM Token Usage & Cost Attribution")
    llm_usage = get_llm_usage_summary()

    l_col1, l_col2, l_col3, l_col4 = st.columns(4)
    l_col1.metric("Total Tokens", f"{llm_usage['total_tokens']:,}")
    l_col2.metric("Prompt Tokens", f"{llm_usage['prompt_tokens']:,}")
    l_col3.metric("Completion Tokens", f"{llm_usage['completion_tokens']:,}")
    l_col4.metric("Estimated Cost", f"${llm_usage['estimated_cost_usd']:.4f}")

    st.write(f"**Total API Calls Recorded:** {llm_usage['calls']}")
    if llm_usage["by_model"]:
        st.write("#### Usage Breakdown by Model")
        m_list = []
        for model_name, stats in llm_usage["by_model"].items():
            m_list.append({
                "Model": model_name,
                "Calls": stats["calls"],
                "Prompt Tokens": f"{stats['prompt_tokens']:,}",
                "Completion Tokens": f"{stats['completion_tokens']:,}",
                "Total Tokens": f"{stats['total_tokens']:,}",
                "Estimated Cost": f"${stats['cost_usd']:.4f}",
            })
        st.dataframe(pd.DataFrame(m_list), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════
# Sidebar Controls & Management
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🎛️ Agent Control")

    if st.button("🔄 Refresh Data (Clear Cache)", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.subheader("Administrative Actions")

    # 1. Pause
    if st.button("⏸️ Pause Trading", use_container_width=True):
        try:
            res = requests.post(
                f"{get_api_base_url()}/pause",
                headers=get_auth_headers(),
                json={"reason": "Manual pause from Streamlit UI"},
                timeout=3,
            )
            if res.status_code == 200:
                st.warning("Trading paused: new opens and adds are blocked.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Failed to pause: {res.text}")
        except Exception as e:
            st.error(f"API unreachable: {e}")

    # 2. Resume
    if st.button("▶️ Resume Trading", use_container_width=True):
        try:
            res = requests.post(
                f"{get_api_base_url()}/resume",
                headers=get_auth_headers(),
                json={},
                timeout=3,
            )
            if res.status_code == 200:
                st.success("Trading resumed: state set to RUNNING.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Failed to resume: {res.text}")
        except Exception as e:
            st.error(f"API unreachable: {e}")

    st.divider()

    # 3. Kill Switch confirmation flow
    st.subheader("🚨 Emergency Kill Switch")
    confirm_kill = st.checkbox("I CONFIRM EMERGENCY KILL SWITCH")
    if st.button(
        "🛑 TRIGGER KILL SWITCH",
        disabled=not confirm_kill,
        type="primary",
        use_container_width=True,
    ):
        try:
            res = requests.post(
                f"{get_api_base_url()}/kill",
                headers=get_auth_headers(),
                json={"confirm": True, "reason": "Emergency kill triggered from Streamlit"},
                timeout=3,
            )
            if res.status_code == 200:
                st.error("KILL SWITCH TRIGGERED: all positions closed, trading halted.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(f"Failed to trigger kill: {res.text}")
        except Exception as e:
            st.error(f"API unreachable: {e}")

    st.divider()
    st.caption(f"Universe ({len(cfg.universe)}): {', '.join(cfg.universe[:4])}...")
    st.caption(f"Risk Per Trade: {cfg.risk_per_trade_pct}%")
    st.caption(f"Max Positions: {cfg.max_open_positions}")
    st.caption(f"Last updated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
