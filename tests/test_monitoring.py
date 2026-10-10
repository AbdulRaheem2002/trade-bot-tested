"""Tests for Phase 6 monitoring and control: Telegram bot, FastAPI, Streamlit, and alerts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.alerts.telegram_bot import TelegramBotService, format_alert_message
from app.common.alerts import Alert, AlertLevel, AlertManager
from app.common.enums import Action, AgentState, ConvictionTier
from app.common.llm_client import MockLLM
from app.common.models import Decision, DecisionItem
from app.config.settings import TradingConfig
from app.dashboard.api import create_api_app
from app.data.store import CandleStore
from app.execution.paper_broker import PaperBroker
from app.orchestrator import Orchestrator
from app.portfolio.db import init_portfolio_db
from app.portfolio.service import PortfolioService

# ── Test Helpers & Mocks ─────────────────────────────────────────────────────


class MockChat:
    def __init__(self, chat_id: int | str):
        self.id = chat_id


class MockMessage:
    def __init__(self, text: str = ""):
        self.text = text


class MockTelegramUpdate:
    def __init__(self, chat_id: int | str, text: str = ""):
        self.effective_chat = MockChat(chat_id)
        self.message = MockMessage(text)


def _populate_candles(store: CandleStore, symbol: str = "BTC/USDT") -> None:
    """Helper to populate historical candles."""
    base_dt = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
    time_future = 9999999999999

    candles_1d: list[list[float]] = []
    p = 60_000.0
    for d in range(35):
        dt = base_dt + timedelta(days=d)
        ts_ms = int(dt.timestamp() * 1000)
        p += 100.0
        candles_1d.append([ts_ms, p, p + 200, p - 200, p + 50, 100_000_000.0])
    store.upsert_candles(symbol, "1d", candles_1d, now_ms=time_future)

    start_4h = base_dt + timedelta(days=20)
    candles_4h: list[list[float]] = []
    p4 = 64_000.0
    for i in range(70):
        dt = start_4h + timedelta(hours=i * 4)
        ts_ms = int(dt.timestamp() * 1000)
        p4 += 30.0
        candles_4h.append([ts_ms, p4, p4 + 500, p4 - 500, p4 + 20, 25_000_000.0])
    store.upsert_candles(symbol, "4h", candles_4h, now_ms=time_future)


async def _create_test_orchestrator(tmp_path: Path) -> Orchestrator:
    """Create an isolated Orchestrator with initialized broker, store, and portfolio."""
    db_url = f"sqlite:///{tmp_path / 'monitor_test.db'}"
    engine = init_portfolio_db(db_url)
    config = TradingConfig(
        universe=["BTC/USDT"],
        starting_equity=10_000.0,
        dashboard_api_token="test-secret-token-123",
        telegram_allowed_chat_ids=["11111", "22222"],
    )

    store = CandleStore(db_url)
    _populate_candles(store, "BTC/USDT")

    broker = PaperBroker(initial_cash=10_000.0)
    await broker.start()

    service = PortfolioService(
        engine=engine,
        broker=broker,
        config=config,
    )
    await service.start()

    canned = [
        Decision(
            reasoning="Testing setup",
            decisions=[
                DecisionItem(
                    action=Action.OPEN_LONG,
                    symbol="BTC/USDT",
                    invalidation_price=64_000.0,
                    conviction=ConvictionTier.A,
                    confidence=0.85,
                    thesis="Test long",
                    rationale="Test rationale",
                )
            ],
        )
    ]
    llm = MockLLM(canned_responses=canned)

    orch = Orchestrator(
        config=config,
        portfolio_service=service,
        broker=broker,
        candle_store=store,
        llm_client=llm,
        engine=engine,
    )
    return orch


# ════════════════════════════════════════════════════════════════════════
# 1. Telegram Whitelist Tests
# ════════════════════════════════════════════════════════════════════════


class TestTelegramWhitelist:
    """Verify that unauthorized chat IDs get no data, no responses, and cannot trigger commands."""

    @pytest.mark.asyncio
    async def test_unauthorized_chat_ignored_completely(self, tmp_path: Path) -> None:
        orch = await _create_test_orchestrator(tmp_path)
        bot = TelegramBotService(
            token="dummy-token",
            allowed_chat_ids=["11111", "22222"],
            orchestrator=orch,
        )

        unauth_update = MockTelegramUpdate(chat_id="99999", text="Hello")

        # 1. Verification of whitelist helper
        assert bot.is_chat_allowed("99999") is False
        assert bot.is_chat_allowed("11111") is True
        assert bot.is_chat_allowed("22222") is True
        assert bot.is_chat_allowed(None) is False

        # 2. Try each command from unauthorized chat
        await bot.cmd_status(unauth_update)
        await bot.cmd_positions(unauth_update)
        await bot.cmd_pnl(unauth_update)
        await bot.cmd_report(unauth_update)
        await bot.cmd_pause(unauth_update)
        await bot.cmd_resume(unauth_update)
        await bot.cmd_kill(unauth_update)
        await bot.handle_message(unauth_update)

        # 3. Assert zero messages sent to unauthorized user
        assert len(bot.sent_messages) == 0
        assert "99999" not in bot._pending_confirmations
        assert orch.agent_state == AgentState.RUNNING
        assert orch.portfolio_service.kill_switch is False

        await orch.portfolio_service.stop()
        await orch.broker.stop()

    @pytest.mark.asyncio
    async def test_authorized_chat_receives_status_and_pnl(self, tmp_path: Path) -> None:
        orch = await _create_test_orchestrator(tmp_path)
        bot = TelegramBotService(
            token="dummy-token",
            allowed_chat_ids=["11111"],
            orchestrator=orch,
        )

        auth_update = MockTelegramUpdate(chat_id="11111")

        # Run /status
        await bot.cmd_status(auth_update)
        assert len(bot.sent_messages) == 1
        assert bot.sent_messages[0]["chat_id"] == "11111"
        assert "PORTFOLIO STATUS" in bot.sent_messages[0]["text"]
        assert "$10,000.00" in bot.sent_messages[0]["text"]

        # Run /positions
        await bot.cmd_positions(auth_update)
        assert len(bot.sent_messages) == 2
        assert "No open positions" in bot.sent_messages[1]["text"]

        # Run /pnl
        await bot.cmd_pnl(auth_update)
        assert len(bot.sent_messages) == 3
        assert "PORTFOLIO PNL SUMMARY" in bot.sent_messages[2]["text"]

        await orch.portfolio_service.stop()
        await orch.broker.stop()


# ════════════════════════════════════════════════════════════════════════
# 2. Confirmation Flow Tests (/kill and /resume)
# ════════════════════════════════════════════════════════════════════════


class TestConfirmationFlows:
    """Verify that destructive operations strictly require interactive confirmation."""

    @pytest.mark.asyncio
    async def test_kill_switch_requires_confirmation(self, tmp_path: Path) -> None:
        orch = await _create_test_orchestrator(tmp_path)
        bot = TelegramBotService(
            token="dummy-token",
            allowed_chat_ids=["11111"],
            orchestrator=orch,
        )

        update = MockTelegramUpdate(chat_id="11111")

        # Step 1: User types /kill
        await bot.cmd_kill(update)
        assert len(bot.sent_messages) == 1
        assert "CONFIRMATION" in bot.sent_messages[0]["text"]
        assert bot._pending_confirmations.get("11111") == "kill"
        # Invariant: kill switch has NOT executed yet
        assert orch.portfolio_service.kill_switch is False
        assert orch.agent_state == AgentState.RUNNING

        # Step 2: User confirms via /confirm_kill
        await bot.cmd_confirm_kill(update)
        assert len(bot.sent_messages) == 2
        assert "KILL SWITCH EXECUTED" in bot.sent_messages[1]["text"]
        # Invariant: kill switch has now executed
        assert orch.portfolio_service.kill_switch is True
        assert orch.agent_state == AgentState.PAUSED_KILL
        assert "11111" not in bot._pending_confirmations

        await orch.portfolio_service.stop()
        await orch.broker.stop()

    @pytest.mark.asyncio
    async def test_kill_switch_confirmed_via_text_yes(self, tmp_path: Path) -> None:
        orch = await _create_test_orchestrator(tmp_path)
        bot = TelegramBotService(
            token="dummy-token",
            allowed_chat_ids=["11111"],
            orchestrator=orch,
        )

        # User triggers /kill
        await bot.cmd_kill(MockTelegramUpdate(chat_id="11111"))
        assert bot._pending_confirmations.get("11111") == "kill"

        # User replies "YES"
        reply_update = MockTelegramUpdate(chat_id="11111", text="YES")
        await bot.handle_message(reply_update)

        assert orch.portfolio_service.kill_switch is True
        assert orch.agent_state == AgentState.PAUSED_KILL
        assert "11111" not in bot._pending_confirmations

        await orch.portfolio_service.stop()
        await orch.broker.stop()

    @pytest.mark.asyncio
    async def test_resume_requires_confirmation(self, tmp_path: Path) -> None:
        orch = await _create_test_orchestrator(tmp_path)
        bot = TelegramBotService(
            token="dummy-token",
            allowed_chat_ids=["11111"],
            orchestrator=orch,
        )

        # First pause the agent
        await bot.cmd_pause(MockTelegramUpdate(chat_id="11111"))
        assert orch.agent_state == AgentState.PAUSED_BREAKER

        # User initiates /resume
        update = MockTelegramUpdate(chat_id="11111")
        await bot.cmd_resume(update)
        assert bot._pending_confirmations.get("11111") == "resume"
        # Invariant: still paused until confirmed
        assert orch.agent_state == AgentState.PAUSED_BREAKER

        # User confirms via /confirm_resume
        await bot.cmd_confirm_resume(update)
        assert orch.agent_state == AgentState.RUNNING
        assert "11111" not in bot._pending_confirmations

        await orch.portfolio_service.stop()
        await orch.broker.stop()

    @pytest.mark.asyncio
    async def test_unprompted_confirmation_ignored(self, tmp_path: Path) -> None:
        orch = await _create_test_orchestrator(tmp_path)
        bot = TelegramBotService(
            token="dummy-token",
            allowed_chat_ids=["11111"],
            orchestrator=orch,
        )

        update = MockTelegramUpdate(chat_id="11111")
        # Direct /confirm_kill without /kill
        await bot.cmd_confirm_kill(update)
        assert "No pending kill command" in bot.sent_messages[0]["text"]
        assert orch.agent_state == AgentState.RUNNING

        # Direct /confirm_resume without /resume
        await bot.cmd_confirm_resume(update)
        assert "No pending resume command" in bot.sent_messages[1]["text"]

        await orch.portfolio_service.stop()
        await orch.broker.stop()


# ════════════════════════════════════════════════════════════════════════
# 3. FastAPI Token Enforcement & Monitoring Endpoints
# ════════════════════════════════════════════════════════════════════════


class TestFastAPIMonitoringAndAuth:
    """Verify read-only endpoints are accessible and write endpoints enforce Bearer token."""

    @pytest.fixture
    def client_and_orch(self, tmp_path: Path):
        db_url = f"sqlite:///{tmp_path / 'fastapi_test.db'}"
        engine = init_portfolio_db(db_url)
        config = TradingConfig(
            universe=["BTC/USDT"],
            starting_equity=10_000.0,
            dashboard_api_token="super-secret-auth-token",
        )
        broker = PaperBroker(initial_cash=10_000.0)
        service = PortfolioService(engine=engine, broker=broker, config=config)
        orch = Orchestrator(
            config=config,
            portfolio_service=service,
            broker=broker,
            engine=engine,
        )

        app = create_api_app(orchestrator=orch)
        client = TestClient(app)
        return client, orch

    def test_read_endpoints_accessible_without_token(self, client_and_orch: tuple[TestClient, Orchestrator]) -> None:
        client, _ = client_and_orch

        # 1. /status
        r = client.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "paper"
        assert data["agent_state"] == "RUNNING"
        assert data["equity"] == 10_000.0

        # 2. /positions
        r = client.get("/positions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

        # 3. /trades
        r = client.get("/trades")
        assert r.status_code == 200

        # 4. /equity-curve
        r = client.get("/equity-curve")
        assert r.status_code == 200

        # 5. /decisions
        r = client.get("/decisions")
        assert r.status_code == 200

        # 6. /brain-reports
        r = client.get("/brain-reports")
        assert r.status_code == 200

        # 7. /metrics
        r = client.get("/metrics")
        assert r.status_code == 200
        assert "equity" in r.json()

        # 8. /llm-cost
        r = client.get("/llm-cost")
        assert r.status_code == 200
        assert "estimated_total_cost_usd" in r.json()

    def test_write_endpoints_reject_missing_token(self, client_and_orch: tuple[TestClient, Orchestrator]) -> None:
        client, orch = client_and_orch

        # POST /pause without Authorization header
        r = client.post("/pause", json={})
        assert r.status_code == 401
        assert "Missing Authorization header" in r.json()["detail"]
        assert orch.agent_state == AgentState.RUNNING

        # POST /kill without Authorization header
        r = client.post("/kill", json={})
        assert r.status_code == 401
        assert orch.portfolio_service.kill_switch is False

        # POST /resume without Authorization header
        r = client.post("/resume", json={})
        assert r.status_code == 401

    def test_write_endpoints_reject_invalid_token(self, client_and_orch: tuple[TestClient, Orchestrator]) -> None:
        client, orch = client_and_orch
        bad_headers = {"Authorization": "Bearer totally-wrong-token"}

        r = client.post("/pause", headers=bad_headers, json={})
        assert r.status_code == 403
        assert "Forbidden: Invalid API bearer token" in r.json()["detail"]
        assert orch.agent_state == AgentState.RUNNING

        r = client.post("/kill", headers=bad_headers, json={})
        assert r.status_code == 403
        assert orch.portfolio_service.kill_switch is False

        r = client.post("/resume", headers=bad_headers, json={})
        assert r.status_code == 403

    def test_write_endpoints_succeed_with_valid_token(self, client_and_orch: tuple[TestClient, Orchestrator]) -> None:
        client, orch = client_and_orch
        valid_headers = {"Authorization": "Bearer super-secret-auth-token"}

        # 1. /pause succeeds
        r = client.post("/pause", headers=valid_headers, json={})
        assert r.status_code == 200
        assert r.json()["status"] == "success"
        assert orch.agent_state == AgentState.PAUSED_BREAKER

        # 2. /resume succeeds
        r = client.post("/resume", headers=valid_headers, json={})
        assert r.status_code == 200
        assert r.json()["status"] == "success"
        assert orch.agent_state == AgentState.RUNNING

        # 3. /kill succeeds
        r = client.post("/kill", headers=valid_headers, json={})
        assert r.status_code == 200
        assert r.json()["status"] == "success"
        assert orch.agent_state == AgentState.PAUSED_KILL
        assert orch.portfolio_service.kill_switch is True


# ════════════════════════════════════════════════════════════════════════
# 4. Alert Formatting & Dispatch Tests
# ════════════════════════════════════════════════════════════════════════


class TestAlertFormattingAndBroadcast:
    """Verify alert formatting, emoji decoration, and broadcast sinks."""

    def test_alert_formatting_emojis(self) -> None:
        # Standard info alert
        a1 = Alert(
            level=AlertLevel.INFO,
            title="System Booted",
            message="Trading agent initialized successfully in PAPER mode.",
            data={"mode": "paper", "starting_equity": 10000.0},
        )
        msg1 = format_alert_message(a1)
        assert "ℹ️ *System Booted*" in msg1
        assert "• `mode`: paper" in msg1
        assert "• `starting_equity`: 10000.0" in msg1

        # Position opened alert
        a2 = Alert(
            level=AlertLevel.INFO,
            title="Position Opened",
            message="Long 0.15 BTC/USDT filled at $67,500.",
            data={"symbol": "BTC/USDT", "side": "BUY"},
        )
        msg2 = format_alert_message(a2)
        assert "🟢 *Position Opened*" in msg2

        # Invalidation closed alert
        a3 = Alert(
            level=AlertLevel.WARNING,
            title="Position Closed (invalidation hit)",
            message="Stop touched at $64,000.",
            data={"reason": "invalidation"},
        )
        msg3 = format_alert_message(a3)
        assert "🔴 *Position Closed (invalidation hit)*" in msg3

        # Kill switch alert
        a4 = Alert(
            level=AlertLevel.CRITICAL,
            title="Emergency Kill Switch Triggered",
            message="All open orders canceled and positions closed.",
            data={"triggered_by": "user_api"},
        )
        msg4 = format_alert_message(a4)
        assert "🛑 *Emergency Kill Switch Triggered*" in msg4

        # Breaker tripped alert
        a5 = Alert(
            level=AlertLevel.CRITICAL,
            title="Circuit Breaker Tripped: Daily Drawdown",
            message="Loss reached 3.2% daily limit.",
        )
        msg5 = format_alert_message(a5)
        assert "⚡ *Circuit Breaker Tripped: Daily Drawdown*" in msg5

    @pytest.mark.asyncio
    async def test_alert_manager_and_discord_broadcast(self) -> None:
        mgr = AlertManager()
        bot = TelegramBotService(
            allowed_chat_ids=["10001", "10002"],
            alert_manager=mgr,
            discord_webhook_url="https://discord.com/api/webhooks/dummy/test",
        )
        mgr.add_handler(bot.on_alert_event)

        with patch.object(bot, "send_discord_alert", new_callable=AsyncMock) as mock_discord:
            await mgr.critical(
                title="Circuit Breaker Tripped",
                message="Loss reached threshold.",
                data={"breaker": "daily_drawdown"},
            )

            # Assert broadcast delivered to both whitelisted chats
            assert len(bot.sent_messages) == 2
            recipients = {m["chat_id"] for m in bot.sent_messages}
            assert recipients == {"10001", "10002"}
            assert "Circuit Breaker Tripped" in bot.sent_messages[0]["text"]

            # Assert secondary Discord sink was called
            mock_discord.assert_awaited_once()
