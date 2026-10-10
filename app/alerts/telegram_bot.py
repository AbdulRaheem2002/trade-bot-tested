"""Telegram bot monitoring and control service – Rule / Phase 6.

Features:
- Whitelist enforcement via TELEGRAM_ALLOWED_CHAT_IDS (fails closed; ignores & logs unauthorized chats).
- Commands:
    /status    - Operating mode, equity, cash, exposure %, breaker state, last cycle.
    /positions - Open positions with entry, current price, invalidation, unrealized R.
    /pnl       - Realized PnL, unrealized PnL, win rate, closed trade summary.
    /report    - Today's Brain report (market regime, thesis, key levels).
    /pause     - Block new opens & adds (close and hold remain allowed).
    /resume    - Requires confirmation reply before unpausing.
    /kill      - Requires confirmation reply, then closes all positions and sets PAUSED_KILL.
- Push alerts for:
    Orders opened/closed, rejection summary, breaker trip, kill switch,
    LLM failure, stale data, and critical monitor errors.
- Daily report at 00:10 UTC.
- Optional Discord webhook as secondary alert sink (alerts only).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.common.alerts import Alert, AlertLevel, AlertManager
from app.common.enums import AgentState
from app.common.logging import get_logger

if TYPE_CHECKING:
    from app.orchestrator import Orchestrator

logger = get_logger(__name__)


def format_alert_message(alert: Alert) -> str:
    """Format structured alert into a readable Markdown message with emojis."""
    level = alert.level
    if level == AlertLevel.CRITICAL:
        emoji = "🚨"
    elif level == AlertLevel.WARNING:
        emoji = "⚠️"
    else:
        emoji = "ℹ️"

    # Specialized emojis for specific titles
    title_lower = alert.title.lower()
    if "order approved" in title_lower or "position opened" in title_lower:
        emoji = "🟢"
    elif "position closed" in title_lower or "invalidation hit" in title_lower:
        emoji = "🔴"
    elif "kill switch" in title_lower:
        emoji = "🛑"
    elif "breaker" in title_lower:
        emoji = "⚡"

    ts_str = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"{emoji} *{alert.title}*",
        f"_{ts_str}_",
        "",
        alert.message,
    ]

    if alert.data:
        lines.append("")
        for k, v in alert.data.items():
            lines.append(f"• `{k}`: {v}")

    return "\n".join(lines)


class TelegramBotService:
    """Telegram bot controller and alert sink."""

    def __init__(
        self,
        token: str | None = None,
        allowed_chat_ids: list[int | str] | None = None,
        orchestrator: Orchestrator | None = None,
        alert_manager: AlertManager | None = None,
        discord_webhook_url: str | None = None,
    ) -> None:
        self.token = token
        raw_ids = allowed_chat_ids or []
        self.allowed_chat_ids: set[str] = {str(cid).strip() for cid in raw_ids if str(cid).strip()}
        self.orchestrator = orchestrator
        self.alert_manager = alert_manager
        self.discord_webhook_url = discord_webhook_url

        # Pending interactive confirmations: {chat_id: "kill" | "resume"}
        self._pending_confirmations: dict[str, str] = {}

        # Bot application handles
        self._app: Application | None = None
        self._is_running = False
        self._daily_report_task: asyncio.Task[None] | None = None

        # In-memory sent message history for test inspection
        self.sent_messages: list[dict[str, Any]] = []

    def is_chat_allowed(self, chat_id: int | str | None) -> bool:
        """Check if chat_id is explicitly whitelisted."""
        if chat_id is None:
            return False
        return str(chat_id).strip() in self.allowed_chat_ids

    # ── Dispatch / Broadcast Helpers ────────────────────────────────────────

    async def send_message(self, chat_id: int | str, text: str) -> None:
        """Send message to a specific chat id (or record in mock history)."""
        cid_str = str(chat_id).strip()
        self.sent_messages.append({"chat_id": cid_str, "text": text, "timestamp": datetime.now(UTC)})

        if self._app and self._app.bot:
            try:
                await self._app.bot.send_message(
                    chat_id=int(cid_str) if cid_str.lstrip("-").isdigit() else cid_str,
                    text=text,
                    parse_mode="Markdown",
                )
            except Exception as exc:
                logger.error("Failed to send Telegram message to chat %s: %s", cid_str, exc)

    async def broadcast(self, text: str) -> None:
        """Broadcast text to all whitelisted chat IDs and secondary Discord webhook."""
        for cid in self.allowed_chat_ids:
            await self.send_message(cid, text)

        if self.discord_webhook_url:
            await self.send_discord_alert(text)

    async def send_discord_alert(self, text: str) -> None:
        """Send alert message to configured Discord webhook."""
        if not self.discord_webhook_url:
            return

        try:
            # Strip markdown formatting for cleaner Discord display if needed
            clean_text = text.replace("*", "**").replace("`", "")
            payload = {"content": clean_text[:2000]}
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(self.discord_webhook_url, json=payload)
                if resp.status_code >= 400:
                    logger.warning("Discord webhook returned %d: %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.error("Error sending alert to Discord webhook: %s", exc)

    # ── Alert Hook ───────────────────────────────────────────────────────────

    async def on_alert_event(self, alert: Alert) -> None:
        """Callback invoked whenever AlertManager emits an alert."""
        text = format_alert_message(alert)
        await self.broadcast(text)

    # ── Command Handlers ─────────────────────────────────────────────────────

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """Handle /status command."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not self.is_chat_allowed(chat_id):
            logger.warning("Unauthorized /status attempt from chat_id: %s", chat_id)
            return

        orch = self.orchestrator
        if not orch:
            await self.send_message(chat_id, "⚠️ Orchestrator not initialized.")
            return

        state = orch.portfolio_service.get_state()
        cash = state.cash
        equity = state.equity
        peak = state.peak_equity
        exposure = (sum(p.notional for p in state.positions.values()) / equity * 100.0) if equity > 0 else 0.0
        mode = orch.config.mode.value.upper()
        agent_state = orch.agent_state.value
        last_cycle = orch.portfolio_service.last_cycle_id or "None"

        lines = [
            f"📊 *PORTFOLIO STATUS ({mode} MODE)*",
            f"• State: `{agent_state}`",
            f"• Equity: `${equity:,.2f}`",
            f"• Cash: `${cash:,.2f}`",
            f"• Peak Equity: `${peak:,.2f}`",
            f"• Total Exposure: `{exposure:.1f}%`",
            f"• Open Positions: `{len(state.positions)}`",
            f"• Breaker Paused: `{orch.breaker_paused}`",
            f"• Kill Switch: `{orch.portfolio_service.kill_switch}`",
            f"• Last Cycle: `{last_cycle}`",
        ]
        await self.send_message(chat_id, "\n".join(lines))

    async def cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """Handle /positions command."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not self.is_chat_allowed(chat_id):
            logger.warning("Unauthorized /positions attempt from chat_id: %s", chat_id)
            return

        orch = self.orchestrator
        if not orch:
            await self.send_message(chat_id, "⚠️ Orchestrator not initialized.")
            return

        positions = orch.portfolio_service.get_state().positions
        if not positions:
            await self.send_message(chat_id, "ℹ️ No open positions currently.")
            return

        lines = ["💼 *OPEN POSITIONS*"]
        for sym, pos in positions.items():
            curr_p = await orch.get_latest_price(sym)
            u_pnl = pos.unrealized_pnl(curr_p)
            u_r = pos.unrealized_r(curr_p)
            sign = "+" if u_pnl >= 0 else ""
            lines.extend([
                f"\n📌 *{sym}* ({pos.side.value.upper()})",
                f"• Size: `{pos.quantity}` | Notional: `${pos.notional:,.2f}`",
                f"• Entry: `${pos.entry_price:,.2f}` | Current: `${curr_p:,.2f}`",
                f"• Invalidation: `${pos.invalidation_price:,.2f}`",
                f"• Unrealized: `{sign}${u_pnl:,.2f}` (`{sign}{u_r:.2f} R`)",
            ])

        await self.send_message(chat_id, "\n".join(lines))

    async def cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """Handle /pnl command."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not self.is_chat_allowed(chat_id):
            logger.warning("Unauthorized /pnl attempt from chat_id: %s", chat_id)
            return

        orch = self.orchestrator
        if not orch:
            await self.send_message(chat_id, "⚠️ Orchestrator not initialized.")
            return

        state = orch.portfolio_service.get_state()
        initial = orch.config.starting_equity
        net_pnl = state.equity - initial
        return_pct = (net_pnl / initial * 100.0) if initial > 0 else 0.0

        # Query closed trades
        from sqlalchemy import select

        from app.portfolio.db import TradeModel

        with orch.portfolio_service.session_factory() as session:
            trades = session.scalars(select(TradeModel)).all()

        realized = sum(t.realized_pnl for t in trades)
        wins = sum(1 for t in trades if t.realized_pnl > 0)
        win_rate = (wins / len(trades) * 100.0) if trades else 0.0

        lines = [
            "📈 *PORTFOLIO PNL SUMMARY*",
            f"• Current Equity: `${state.equity:,.2f}`",
            f"• Total Return: `{net_pnl:+,.2f}` (`{return_pct:+.2f}%`)",
            f"• Realized PnL: `${realized:+,.2f}`",
            f"• Closed Trades: `{len(trades)}`",
            f"• Win Rate: `{win_rate:.1f}%`",
        ]
        await self.send_message(chat_id, "\n".join(lines))

    async def cmd_report(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """Handle /report command (returns today's latest BrainReport)."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not self.is_chat_allowed(chat_id):
            logger.warning("Unauthorized /report attempt from chat_id: %s", chat_id)
            return

        orch = self.orchestrator
        if not orch:
            await self.send_message(chat_id, "⚠️ Orchestrator not initialized.")
            return

        from sqlalchemy import select

        from app.portfolio.db import BrainReportModel

        with orch.portfolio_service.session_factory() as session:
            stmt = select(BrainReportModel).order_by(BrainReportModel.created_at.desc()).limit(1)
            report_row = session.scalar(stmt)

        if not report_row:
            await self.send_message(chat_id, "ℹ️ No Brain report available yet.")
            return

        import json
        rep_dict = json.loads(report_row.report_json)
        regime = rep_dict.get("market_regime", "unclear")
        summary = rep_dict.get("summary", "")
        perf = rep_dict.get("performance_review", "")

        lines = [
            "🧠 *LATEST BRAIN RESEARCH REPORT*",
            f"• Market Regime: `{regime}`",
            f"• Created: `{report_row.created_at.strftime('%Y-%m-%d %H:%M UTC')}`",
            "",
            f"📝 *Summary*:\n{summary}",
            "",
            f"🔍 *Performance Review*:\n{perf}",
        ]
        await self.send_message(chat_id, "\n".join(lines))

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """Handle /pause command."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not self.is_chat_allowed(chat_id):
            logger.warning("Unauthorized /pause attempt from chat_id: %s", chat_id)
            return

        orch = self.orchestrator
        if not orch:
            return

        orch.breaker_paused = True
        orch.agent_state = AgentState.PAUSED_BREAKER
        await orch.portfolio_service.set_agent_state(AgentState.PAUSED_BREAKER.value)

        await self.send_message(
            chat_id,
            "⏸ *AGENT PAUSED*\nNew position opens and adds are blocked. Close and hold remain allowed."
        )

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """Handle /resume command initiation."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not self.is_chat_allowed(chat_id):
            logger.warning("Unauthorized /resume attempt from chat_id: %s", chat_id)
            return

        cid_str = str(chat_id)
        self._pending_confirmations[cid_str] = "resume"
        await self.send_message(
            chat_id,
            "⚠️ *CONFIRM RESUME*\nAre you sure you want to resume normal trading? "
            "Reply with `/confirm_resume` or `YES` to proceed."
        )

    async def cmd_confirm_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """Confirm /resume execution."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not self.is_chat_allowed(chat_id):
            return

        cid_str = str(chat_id)
        if self._pending_confirmations.get(cid_str) != "resume":
            await self.send_message(chat_id, "ℹ️ No pending resume command to confirm.")
            return

        del self._pending_confirmations[cid_str]
        orch = self.orchestrator
        if not orch:
            return

        try:
            await orch.resume()
            await self.send_message(chat_id, "✅ *RESUMED*: Trading agent returned to `RUNNING` state.")
        except Exception as exc:
            await self.send_message(chat_id, f"❌ Failed to resume: {exc}")

    async def cmd_kill(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """Handle /kill command initiation."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not self.is_chat_allowed(chat_id):
            logger.warning("Unauthorized /kill attempt from chat_id: %s", chat_id)
            return

        cid_str = str(chat_id)
        self._pending_confirmations[cid_str] = "kill"
        await self.send_message(
            chat_id,
            "🚨 *EMERGENCY KILL SWITCH CONFIRMATION*\n"
            "This will immediately close ALL open positions at market, cancel all orders, and pause the agent. "
            "Reply with `/confirm_kill` or `YES` to confirm."
        )

    async def cmd_confirm_kill(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """Confirm /kill execution."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not self.is_chat_allowed(chat_id):
            return

        cid_str = str(chat_id)
        if self._pending_confirmations.get(cid_str) != "kill":
            await self.send_message(chat_id, "ℹ️ No pending kill command to confirm.")
            return

        del self._pending_confirmations[cid_str]
        orch = self.orchestrator
        if not orch:
            return

        await orch.trigger_kill_switch("Telegram /kill confirmation")
        await self.send_message(
            chat_id,
            "🛑 *KILL SWITCH EXECUTED*\nAll open positions closed. Agent set to `PAUSED_KILL`."
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE | None = None) -> None:
        """Handle incoming text messages to check for affirmative confirmation."""
        chat_id = update.effective_chat.id if update.effective_chat else None
        if not self.is_chat_allowed(chat_id):
            logger.warning("Unauthorized text message attempt from chat_id: %s", chat_id)
            return

        text = (update.message.text or "").strip().upper() if update.message else ""
        cid_str = str(chat_id)

        if cid_str in self._pending_confirmations and text in ("YES", "CONFIRM", "Y"):
            action = self._pending_confirmations[cid_str]
            if action == "kill":
                await self.cmd_confirm_kill(update, context)
            elif action == "resume":
                await self.cmd_confirm_resume(update, context)

    # ── Daily Scheduled Report Loop (00:10 UTC) ──────────────────────────────

    async def _daily_report_loop(self) -> None:
        """Broadcast summary daily report at 00:10 UTC."""
        while self._is_running:
            try:
                now = datetime.now(UTC)
                target = now.replace(hour=0, minute=10, second=0, microsecond=0)
                if now >= target:
                    target += timedelta(days=1)

                sleep_seconds = max(1.0, (target - now).total_seconds())
                await asyncio.sleep(sleep_seconds)

                if not self._is_running:
                    break

                await self.send_daily_summary_report()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in Telegram daily report loop: %s", exc)
                await asyncio.sleep(60)

    async def send_daily_summary_report(self) -> None:
        """Compile and send daily report."""
        orch = self.orchestrator
        if not orch:
            return

        state = orch.portfolio_service.get_state()
        initial = orch.config.starting_equity
        net_pnl = state.equity - initial

        # Query 24h closed trades
        from sqlalchemy import select

        from app.portfolio.db import RejectionModel, TradeModel

        yesterday = datetime.now(UTC) - timedelta(days=1)
        with orch.portfolio_service.session_factory() as session:
            trades = session.scalars(
                select(TradeModel).where(TradeModel.exit_time >= yesterday)
            ).all()
            rejections = session.scalars(
                select(RejectionModel).where(RejectionModel.timestamp >= yesterday)
            ).all()

        daily_realized = sum(t.realized_pnl for t in trades)

        lines = [
            "🌅 *DAILY PORTFOLIO SUMMARY (00:10 UTC)*",
            f"• Equity: `${state.equity:,.2f}`",
            f"• 24h Realized PnL: `${daily_realized:+,.2f}`",
            f"• Cumulative PnL: `${net_pnl:+,.2f}`",
            f"• Open Positions: `{len(state.positions)}`",
            f"• 24h Closed Trades: `{len(trades)}`",
            f"• 24h Rejections: `{len(rejections)}`",
            f"• Status: `{orch.agent_state.value}`",
        ]
        await self.broadcast("\n".join(lines))

    # ── Service Lifecycle ────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start bot polling and background daily reporting."""
        if self._is_running:
            return
        self._is_running = True

        # Attach alert callback
        if self.alert_manager:
            self.alert_manager.add_handler(self.on_alert_event)

        # Launch daily report task
        self._daily_report_task = asyncio.create_task(
            self._daily_report_loop(), name="telegram_daily_report"
        )

        # If live token is configured, initialize python-telegram-bot Application
        if self.token:
            try:
                self._app = Application.builder().token(self.token).build()
                self._app.add_handler(CommandHandler("status", self.cmd_status))
                self._app.add_handler(CommandHandler("positions", self.cmd_positions))
                self._app.add_handler(CommandHandler("pnl", self.cmd_pnl))
                self._app.add_handler(CommandHandler("report", self.cmd_report))
                self._app.add_handler(CommandHandler("pause", self.cmd_pause))
                self._app.add_handler(CommandHandler("resume", self.cmd_resume))
                self._app.add_handler(CommandHandler("confirm_resume", self.cmd_confirm_resume))
                self._app.add_handler(CommandHandler("kill", self.cmd_kill))
                self._app.add_handler(CommandHandler("confirm_kill", self.cmd_confirm_kill))
                self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

                await self._app.initialize()
                await self._app.start()
                if self._app.updater:
                    await self._app.updater.start_polling()
                logger.info("Telegram Bot service started with %d whitelisted chats.", len(self.allowed_chat_ids))
            except Exception as exc:
                logger.warning("Could not start live Telegram polling: %s", exc)

    async def stop(self) -> None:
        """Stop bot polling and detach handlers."""
        if not self._is_running:
            return
        self._is_running = False

        if self._daily_report_task:
            self._daily_report_task.cancel()
            self._daily_report_task = None

        if self.alert_manager:
            self.alert_manager.remove_handler(self.on_alert_event)

        if self._app:
            try:
                if self._app.updater:
                    await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as exc:
                logger.debug("Telegram app shutdown notice: %s", exc)
            self._app = None

        logger.info("Telegram Bot service stopped.")
