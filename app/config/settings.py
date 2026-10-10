"""Pydantic v2 settings models loaded from config.yaml + env vars."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator

from app.common.enums import ConvictionTier, TradingMode


class LiveConfig(BaseModel):
    """Configuration for Live Trading Adapter – Phase 9."""

    model_config = {"extra": "forbid"}

    max_capital: Annotated[float, Field(gt=0)] = 10_000.0
    exact_confirmation_phrase: str = "I CONFIRM LIVE TRADING WITH REAL CAPITAL"


class TradingConfig(BaseModel):
    """Root configuration – every field has a sensible default matching the
    master project context.  Ranges are validated at load time so the agent
    fails fast on bad config.
    """

    model_config = {"extra": "forbid"}

    # ── General ──────────────────────────────────────────────────────
    mode: TradingMode = TradingMode.PAPER

    # ── Universe & Exchange ──────────────────────────────────────────
    universe: list[str] = Field(
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        min_length=1,
    )
    exchange_id: str = "binance"

    # ── Timeframes ───────────────────────────────────────────────────
    decision_timeframe: str = "4h"
    context_timeframe: str = "1d"

    # ── Capital ──────────────────────────────────────────────────────
    starting_equity: Annotated[float, Field(gt=0)] = 10_000.0

    # ── Position Sizing ──────────────────────────────────────────────
    risk_per_trade_pct: Annotated[float, Field(ge=0.1, le=2.0)] = 0.75
    conviction_multipliers: dict[ConvictionTier, float] = Field(
        default={ConvictionTier.A: 1.0, ConvictionTier.B: 0.6},
    )

    # ── Exposure Limits ──────────────────────────────────────────────
    max_open_positions: Annotated[int, Field(ge=1, le=50)] = 5
    max_total_exposure_pct: Annotated[float, Field(gt=0, le=90)] = 70.0
    max_single_position_pct: Annotated[float, Field(gt=0, le=50)] = 20.0

    # ── Stop / Invalidation ─────────────────────────────────────────
    min_stop_atr: Annotated[float, Field(gt=0)] = 1.0
    max_stop_atr: Annotated[float, Field(gt=0)] = 4.0

    # ── Market Quality Gates ─────────────────────────────────────────
    max_spread_bps: Annotated[float, Field(ge=0)] = 10.0
    min_24h_quote_volume: Annotated[float, Field(ge=0)] = 50_000_000.0

    # ── Fees & Slippage ──────────────────────────────────────────────
    fees_taker_bps: Annotated[float, Field(ge=0)] = 10.0
    slippage_bps: Annotated[float, Field(ge=0)] = 5.0

    # ── Circuit Breakers ─────────────────────────────────────────────
    daily_loss_limit_pct: Annotated[float, Field(gt=0, le=100)] = 3.0
    weekly_loss_limit_pct: Annotated[float, Field(gt=0, le=100)] = 6.0
    peak_drawdown_limit_pct: Annotated[float, Field(gt=0, le=100)] = 8.0

    # ── Short Selling ────────────────────────────────────────────────
    shorts_enabled: bool = False
    min_equity_for_shorts: Annotated[float, Field(gt=0)] = 25_000.0

    # ── LLM ──────────────────────────────────────────────────────────
    decision_llm_timeout_s: Annotated[int, Field(gt=0)] = 60
    llm_max_retries: Annotated[int, Field(ge=0)] = 2

    # ── Features ─────────────────────────────────────────────────────
    news_enabled: bool = False

    # ── Orchestrator & Loop ──────────────────────────────────────────
    candle_close_delay_seconds: Annotated[int, Field(ge=0)] = 20
    invalidation_monitor_interval_seconds: Annotated[int, Field(gt=0)] = 30
    invalidation_data_timeout_seconds: Annotated[int, Field(gt=0)] = 120
    invalidation_close_on_timeout: bool = False
    equity_snapshot_interval_seconds: Annotated[int, Field(gt=0)] = 300
    lock_file_path: str = ".trading_agent.lock"
    shutdown_marker_path: str = ".clean_shutdown"
    kill_switch_file: str = "KILL_SWITCH"
    breaker_action: str = "block"  # "block" or "flatten"
    database_url: str = "sqlite:///trading_agent.db"

    # ── Monitoring & Control ─────────────────────────────────────────
    telegram_bot_token: str | None = None
    telegram_allowed_chat_ids: list[int | str] = Field(default_factory=list)
    discord_webhook_url: str | None = None
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8000
    dashboard_api_token: str = "secret-token"

    # ── Live Trading ─────────────────────────────────────────────────
    live: LiveConfig = Field(default_factory=lambda: LiveConfig())

    # ── Cross-field validators ───────────────────────────────────────

    @model_validator(mode="after")
    def _check_cross_field_constraints(self) -> TradingConfig:
        if self.min_stop_atr > self.max_stop_atr:
            raise ValueError(
                f"min_stop_atr ({self.min_stop_atr}) must be <= max_stop_atr ({self.max_stop_atr})"
            )
        if self.max_single_position_pct > self.max_total_exposure_pct:
            raise ValueError(
                f"max_single_position_pct ({self.max_single_position_pct}) "
                f"must be <= max_total_exposure_pct ({self.max_total_exposure_pct})"
            )
        return self

    @field_validator("conviction_multipliers")
    @classmethod
    def _check_conviction_values(
        cls, v: dict[ConvictionTier, float]
    ) -> dict[ConvictionTier, float]:
        for tier, mult in v.items():
            if mult <= 0 or mult > 2.0:
                raise ValueError(
                    f"Conviction multiplier for {tier} must be in (0, 2.0], got {mult}"
                )
        # Both tiers must be present
        for tier in ConvictionTier:
            if tier not in v:
                raise ValueError(f"Missing conviction multiplier for tier {tier.value}")
        return v
