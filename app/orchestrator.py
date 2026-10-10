"""Autonomous trading agent orchestrator – Rule / Phase 5.

Orchestrates:
1. Single-instance lock (.trading_agent.lock) & clean shutdown marker (.clean_shutdown).
2. Restart recovery: state loading, broker reconciliation, missed-cycle detection (no retro-trading),
   unclean shutdown detection & alerts.
3. Background tasks:
   - Candle-close scheduler (triggers Reflex cycle 20s after each 4h close; guards against duplicate cycle_id)
   - Daily Brain job at 00:05 UTC
   - Invalidation monitor (every 30s: checks open position prices vs stop levels, closes immediately on breach)
   - Equity snapshot task (every 5 minutes)
   - Kill-switch watcher (KILL_SWITCH file or system_state: closes all positions, cancels orders, enters PAUSED_KILL)
4. Ordered cycle pipeline:
   refresh data -> build snapshots -> check breakers & kill switch -> run_reflex ->
   apply approved orders via Broker -> update protective stops -> persist everything -> send alert.
5. Replay mode runner for simulated multi-day runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.engine import Engine

from app.brain.brain import run_brain
from app.common.alerts import AlertManager, get_alert_manager
from app.common.enums import Action, AgentState, ConvictionTier, Side, TradingMode
from app.common.errors import (
    InstanceAlreadyRunningError,
    OrchestratorError,
)
from app.common.llm_client import LLMClient
from app.common.logging import get_logger
from app.common.models import (
    ApprovedOrder,
    CycleState,
    MarketSnapshot,
    Rejection,
)
from app.common.time_utils import (
    get_latest_closed_candle_time,
    get_next_candle_boundary,
)
from app.config.loader import load_config
from app.config.settings import TradingConfig
from app.data.exchange_client import ExchangeClient
from app.data.news import NewsProvider, NullNewsProvider
from app.data.snapshot import build_snapshot
from app.data.store import CandleStore
from app.execution.base import Broker
from app.execution.paper_broker import PaperBroker
from app.portfolio.db import init_portfolio_db
from app.portfolio.service import PortfolioService, ReconciliationReport
from app.reflex.reflex import run_reflex
from app.risk.breakers import compute_breaker_state

logger = get_logger(__name__)


class SingleInstanceLock:
    """Guarantees only a single instance of the trading agent can run concurrently."""

    def __init__(self, lock_path: Path | str = ".trading_agent.lock") -> None:
        self.lock_path = Path(lock_path)
        self.acquired = False

    def acquire(self) -> None:
        """Acquire lock file or raise InstanceAlreadyRunningError."""
        if self.lock_path.exists():
            try:
                content = self.lock_path.read_text().strip()
                pid = int(content)
                if pid != os.getpid() and self._is_pid_running(pid):
                    raise InstanceAlreadyRunningError(
                        f"Another instance of the trading agent is already running (PID {pid})."
                    )
            except (ValueError, OSError):
                # Stale or corrupted lock file; safe to overwrite
                pass

        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_text(str(os.getpid()))
        self.acquired = True
        logger.debug("Acquired single-instance lock: %s (PID %d)", self.lock_path, os.getpid())

    def release(self) -> None:
        """Release the lock file if owned by the current process."""
        if self.acquired and self.lock_path.exists():
            try:
                content = self.lock_path.read_text().strip()
                if int(content) == os.getpid():
                    self.lock_path.unlink(missing_ok=True)
            except Exception:
                pass
            self.acquired = False
            logger.debug("Released single-instance lock: %s", self.lock_path)

    @staticmethod
    def _is_pid_running(pid: int) -> bool:
        """Check if process with given PID exists."""
        if pid <= 0:
            return False
        try:
            import psutil

            return psutil.pid_exists(pid)
        except Exception:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False


class ShutdownMarker:
    """Tracks clean vs unclean agent shutdowns across restarts."""

    def __init__(self, marker_path: Path | str = ".clean_shutdown") -> None:
        self.marker_path = Path(marker_path)

    def is_unclean(self) -> bool:
        """Return True if marker is absent (indicating previous crash or kill)."""
        return not self.marker_path.exists()

    def mark_running(self) -> None:
        """Remove clean marker while agent is actively running."""
        self.marker_path.unlink(missing_ok=True)

    def mark_clean(self) -> None:
        """Create clean shutdown marker upon graceful shutdown."""
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        self.marker_path.write_text(f"CLEAN_SHUTDOWN {datetime.now(UTC).isoformat()}\n")


def detect_missed_cycles(last_cycle_id: str | None, now: datetime) -> list[str]:
    """Detect and return missed 4-hour cycle IDs between last_cycle_id and current time."""
    if not last_cycle_id:
        return []

    try:
        parts = last_cycle_id.split("_")
        if len(parts) >= 3:
            dt_str = f"{parts[1]}_{parts[2]}"
            last_dt = datetime.strptime(dt_str, "%Y%m%d_%H%M").replace(tzinfo=UTC)
        else:
            return []
    except Exception:
        return []

    missed: list[str] = []
    curr = last_dt + timedelta(hours=4)
    latest_closed = get_latest_closed_candle_time(now, "4h")

    while curr <= latest_closed:
        cycle_str = f"cycle_{curr.strftime('%Y%m%d_%H%M')}"
        missed.append(cycle_str)
        curr += timedelta(hours=4)

    return missed


class Orchestrator:
    """Master orchestrator for autonomous agent loops, recovery, and cycle execution."""

    def __init__(
        self,
        config: TradingConfig | None = None,
        engine: Engine | str | None = None,
        broker: Broker | None = None,
        portfolio_service: PortfolioService | None = None,
        candle_store: CandleStore | None = None,
        llm_client: LLMClient | None = None,
        exchange_client: ExchangeClient | None = None,
        news_provider: NewsProvider | None = None,
        alert_manager: AlertManager | None = None,
        lock_path: Path | str | None = None,
        shutdown_marker_path: Path | str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.engine = init_portfolio_db(engine or "sqlite:///trading_agent.db")
        if broker is not None:
            self.broker = broker
        elif self.config.mode == TradingMode.LIVE:
            from app.execution.live_broker import LiveBroker

            self.broker = LiveBroker(
                exchange_client=self.exchange_client,
                config=self.config,
                alert_manager=self.alert_manager,
                max_capital=getattr(self.config.live, "max_capital", 10_000.0),
            )
        else:
            self.broker = PaperBroker(
                initial_cash=self.config.starting_equity,
                slippage_bps=self.config.slippage_bps,
                fees_taker_bps=self.config.fees_taker_bps,
                shorts_enabled=self.config.shorts_enabled,
            )
        self.portfolio_service = portfolio_service or PortfolioService(
            engine=self.engine,
            broker=self.broker,
            config=self.config,
        )
        self.candle_store = candle_store or CandleStore(self.engine)
        self.llm_client = llm_client or LLMClient(
            model=os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini"),
            timeout_s=self.config.decision_llm_timeout_s,
            max_retries=self.config.llm_max_retries,
        )
        self.exchange_client = exchange_client
        self.news_provider = news_provider or NullNewsProvider()
        self.alert_manager = alert_manager or get_alert_manager()

        # Guards & markers
        self.lock = SingleInstanceLock(lock_path or self.config.lock_file_path)
        self.shutdown_marker = ShutdownMarker(
            shutdown_marker_path or self.config.shutdown_marker_path
        )

        # Operational state
        self.agent_state = AgentState.RUNNING
        self.breaker_paused = False
        self.completed_cycles: set[str] = set()
        self._running_cycle = False
        self._is_running = False

        # Background task handles
        self._tasks: list[asyncio.Task[Any]] = []

    # ── Startup & Recovery ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Start orchestrator: lock acquisition, restart recovery, and background tasks."""
        if self._is_running:
            return

        # 1. Single-instance guard
        self.lock.acquire()

        # 2. Check for unclean shutdown marker from previous run
        now = datetime.now(UTC)
        if self.shutdown_marker.is_unclean() and self.portfolio_service.last_cycle_id:
            await self.alert_manager.warning(
                title="Unclean Shutdown Detected",
                message="Previous agent instance terminated abnormally without writing clean shutdown marker.",
            )

        # Mark agent actively running
        self.shutdown_marker.mark_running()

        # 3. Load persisted state from DB and start PortfolioService
        if self.portfolio_service and self.portfolio_service._is_running:
            await self.portfolio_service.stop()

        self.portfolio_service = PortfolioService.load_from_db(
            engine=self.engine,
            broker=self.broker,
            config=self.config,
        )
        await self.portfolio_service.start()

        # In Paper Mode, synchronize in-memory broker state with recovered DB state
        if isinstance(self.broker, PaperBroker):
            self.broker.cash = self.portfolio_service.cash
            self.broker._positions = dict(self.portfolio_service._positions)

        if self.portfolio_service.last_cycle_id:
            self.completed_cycles.add(self.portfolio_service.last_cycle_id)

        # 5. Reconcile state with broker (startup check)
        try:
            await self.reconcile_state()
        except Exception as exc:
            logger.warning("Startup reconciliation check encountered warning: %s", exc)

        # 6. Detect missed cycles
        missed = detect_missed_cycles(self.portfolio_service.last_cycle_id, now)
        if missed:
            logger.warning(
                "Detected %d missed 4h cycles since last run (%s to %s). NOT retroactively trading.",
                len(missed),
                missed[0],
                missed[-1],
            )
            await self.alert_manager.info(
                title="Missed Cycles Detected",
                message=f"Detected {len(missed)} missed cycles since last run. Skipping retroactive execution.",
                data={"missed_cycles": missed},
            )

        # 7. Check if kill-switch file is active
        await self.check_kill_switch()

        self._is_running = True

        # 8. Launch background tasks
        self._tasks.append(
            asyncio.create_task(
                self._candle_close_scheduler_loop(), name="candle_close_scheduler"
            )
        )
        self._tasks.append(
            asyncio.create_task(self._daily_brain_loop(), name="daily_brain_job")
        )
        self._tasks.append(
            asyncio.create_task(
                self._invalidation_monitor_loop(), name="invalidation_monitor"
            )
        )
        self._tasks.append(
            asyncio.create_task(
                self._equity_snapshot_loop(), name="equity_snapshot_task"
            )
        )
        self._tasks.append(
            asyncio.create_task(
                self._kill_switch_watcher_loop(), name="kill_switch_watcher"
            )
        )
        self._tasks.append(
            asyncio.create_task(
                self._reconciliation_loop(), name="reconciliation_loop"
            )
        )

        logger.info(
            "Orchestrator successfully started (Mode: %s, State: %s)",
            self.config.mode.value,
            self.agent_state.value,
        )

    async def stop(self) -> None:
        """Gracefully shut down all tasks, flush state, and record clean shutdown marker."""
        if not self._is_running:
            self.lock.release()
            return

        logger.info("Initiating graceful orchestrator shutdown...")
        self._is_running = False

        # 1. Cancel background tasks
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # 2. Wait if a cycle step is currently in progress
        while self._running_cycle:
            logger.info("Waiting for active cycle step to finish...")
            await asyncio.sleep(0.5)

        # 3. Stop portfolio service single-writer queue
        await self.portfolio_service.stop()

        # 4. Write clean shutdown marker
        self.shutdown_marker.mark_clean()

        # 5. Release single-instance lock
        self.lock.release()
        logger.info("Graceful shutdown complete.")

    async def __aenter__(self) -> Orchestrator:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()

    # ── Kill Switch & Resume ─────────────────────────────────────────────────

    async def check_kill_switch(self) -> None:
        """Inspect kill-switch file or database flag and trigger emergency pause if set."""
        kill_file = Path(self.config.kill_switch_file)
        triggered = (
            kill_file.exists()
            or self.portfolio_service.kill_switch
            or (self.agent_state == AgentState.PAUSED_KILL)
        )
        if triggered and self.agent_state != AgentState.PAUSED_KILL:
            await self.trigger_kill_switch("Kill switch file or database flag detected")

    async def trigger_kill_switch(self, reason: str = "") -> None:
        """Close all positions, cancel all orders, set PAUSED_KILL, and alert."""
        logger.critical("KILL SWITCH TRIGGERED: %s", reason)
        self.agent_state = AgentState.PAUSED_KILL
        self.portfolio_service.kill_switch = True
        await self.portfolio_service.set_agent_state(AgentState.PAUSED_KILL.value)

        # 1. Close all open positions at once via Broker
        positions = list(self.portfolio_service.get_state().positions.keys())
        for symbol in positions:
            try:
                fill = await self.broker.close_position(symbol, reason="kill_switch")
                if fill:
                    await self.portfolio_service.apply_close_fill(
                        fill, reason="kill_switch"
                    )
            except Exception as exc:
                logger.error(
                    "Error closing position %s during kill switch: %s", symbol, exc
                )

        # 2. Cancel all resting orders / stops
        try:
            await self.broker.cancel_all()
        except Exception as exc:
            logger.error("Error canceling orders during kill switch: %s", exc)

        # 3. Send critical alert
        await self.alert_manager.critical(
            title="KILL SWITCH ACTIVATED",
            message=f"Agent set to PAUSED_KILL. All positions closed and orders cancelled. Reason: {reason}",
            data={"reason": reason, "closed_positions": positions},
        )

    async def resume(self) -> None:
        """Explicitly resume agent from PAUSED_KILL or paused breaker."""
        kill_file = Path(self.config.kill_switch_file)
        if kill_file.exists():
            raise OrchestratorError(
                f"Cannot resume while kill switch file '{self.config.kill_switch_file}' exists. "
                "Remove the file before issuing resume."
            )

        self.agent_state = AgentState.RUNNING
        self.portfolio_service.kill_switch = False
        self.breaker_paused = False
        await self.portfolio_service.set_agent_state(AgentState.RUNNING.value)

        await self.alert_manager.info(
            title="Agent Resumed",
            message="Trading agent resumed normal RUNNING state.",
        )
        logger.info("Trading agent resumed to RUNNING state.")

    # ── Reconciliation (Startup & Every 5 Minutes) ───────────────────────────

    async def reconcile_state(self) -> ReconciliationReport:
        """Execute reconciliation check and pause trading if mismatch found.

        Never auto-corrects silently. Discrepancies pause new orders and alert operator.
        """
        report = await self.portfolio_service.reconcile(strict=False)
        if not report.is_matched:
            logger.critical(
                "Reconciliation mismatch detected! Pausing new orders. Discrepancies: %s",
                report.mismatches,
            )
            self.breaker_paused = True
            self.agent_state = AgentState.PAUSED_RECONCILIATION
            await self.portfolio_service.set_agent_state(AgentState.PAUSED_RECONCILIATION.value)
            await self.alert_manager.critical(
                title="CRITICAL: Broker Reconciliation Mismatch",
                message=(
                    "Discrepancies found between broker and internal state:\n"
                    + "\n".join(f"• {m}" for m in report.mismatches)
                    + "\n\nNew orders are paused. Manual resolution required."
                ),
                data={"mismatches": report.mismatches},
            )
        return report

    async def _reconciliation_loop(self) -> None:
        """Periodic background reconciliation every 5 minutes (300 seconds)."""
        interval = 300
        while self._is_running:
            try:
                await asyncio.sleep(interval)
                if not self._is_running:
                    break
                await self.reconcile_state()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in reconciliation background loop: %s", exc)

    # ── Invalidation Monitor (Pure Code, Every 30s) ──────────────────────────

    async def check_invalidations(self) -> None:
        """Check all open positions against their invalidation levels and close immediately on breach."""
        state = self.portfolio_service.get_state()
        positions = state.positions
        if not positions:
            return

        for symbol, pos in list(positions.items()):
            # 1. Fetch latest price
            try:
                price = await self.get_latest_price(symbol)
            except Exception as exc:
                await self.alert_manager.critical(
                    title="Invalidation Monitor: Price Data Unavailable",
                    message=f"Failed to fetch market price for {symbol}: {exc}",
                    data={"symbol": symbol, "error": str(exc)},
                )
                if self.config.invalidation_close_on_timeout:
                    fill = await self.broker.close_position(symbol, reason="data_timeout")
                    if fill:
                        await self.portfolio_service.apply_close_fill(
                            fill, reason="data_timeout"
                        )
                continue

            # 2. Invalidation check
            is_breached = (pos.side == Side.LONG and price <= pos.invalidation_price) or (
                pos.side == Side.SHORT and price >= pos.invalidation_price
            )

            if is_breached:
                logger.warning(
                    "Position %s hit invalidation! Current: %.4f, Invalidation: %.4f. Closing at once.",
                    symbol,
                    price,
                    pos.invalidation_price,
                )
                fill = await self.broker.close_position(symbol, reason="invalidation")
                if fill:
                    await self.portfolio_service.apply_close_fill(
                        fill, reason="invalidation"
                    )

                await self.alert_manager.critical(
                    title=f"Position Invalidation Hit: {symbol}",
                    message=(
                        f"Closed {pos.side.value.upper()} position in {symbol} at {price:.2f}. "
                        f"Invalidation stop was {pos.invalidation_price:.2f}."
                    ),
                    data={
                        "symbol": symbol,
                        "side": pos.side.value,
                        "exit_price": price,
                        "invalidation_price": pos.invalidation_price,
                    },
                )

    async def get_latest_price(self, symbol: str) -> float:
        """Get latest price for symbol via broker or exchange client."""
        if isinstance(self.broker, PaperBroker):
            return await self.broker.get_market_price(symbol)
        if self.exchange_client:
            ticker = await self.exchange_client.fetch_ticker(symbol)
            return float(ticker.get("last", 0.0))
        raise OrchestratorError(f"No price provider available for {symbol}")

    # ── Cycle Pipeline ───────────────────────────────────────────────────────

    async def execute_cycle(
        self,
        cycle_id: str,
        cycle_time: datetime | None = None,
    ) -> tuple[list[ApprovedOrder], list[Rejection]]:
        """Run ordered Reflex cycle pipeline.

        Order:
        refresh data -> build snapshots -> check breakers & kill switch ->
        run_reflex -> apply approved orders through Broker -> update protective stops ->
        persist everything -> send alert.
        """
        # Guard against double-running the same cycle_id
        if cycle_id in self.completed_cycles:
            logger.info("Cycle %s already completed. Skipping double-run.", cycle_id)
            return [], []

        if self._running_cycle:
            logger.warning("Another cycle is currently running. Skipping %s.", cycle_id)
            return [], []

        self._running_cycle = True
        cycle_dt = cycle_time or datetime.now(UTC)

        try:
            # ── 1. Refresh data ──────────────────────────────────────────────
            if self.exchange_client:
                tf = getattr(self.config, "decision_timeframe", "15m")
                context_tf = getattr(self.config, "context_timeframe", "4h")
                for sym in self.config.universe:
                    try:
                        await self.candle_store.update_to_latest(
                            symbol=sym,
                            timeframe="4h",
                            client=self.exchange_client,
                        )
                        if tf != "4h":
                            await self.candle_store.update_to_latest(
                                symbol=sym,
                                timeframe=tf,
                                client=self.exchange_client,
                            )
                        if context_tf not in ("4h", tf):
                            await self.candle_store.update_to_latest(
                                symbol=sym,
                                timeframe=context_tf,
                                client=self.exchange_client,
                            )
                    except Exception as exc:
                        logger.warning("Error refreshing data for %s: %s", sym, exc)

            # ── 2. Build snapshots ───────────────────────────────────────────
            snapshots: dict[str, MarketSnapshot] = {}
            for sym in self.config.universe:
                try:
                    snap = await build_snapshot(
                        symbol=sym,
                        store=self.candle_store,
                        client=self.exchange_client,
                        config=self.config,
                        now=cycle_dt,
                    )
                    snapshots[sym] = snap
                except Exception as exc:
                    logger.warning("Could not build snapshot for %s: %s", sym, exc)

            # ── 3. Check breakers and kill switch ────────────────────────────
            await self.check_kill_switch()
            kill_switch_active = (
                self.agent_state == AgentState.PAUSED_KILL
                or self.portfolio_service.kill_switch
            )

            # Circuit breakers
            current_prices = {s: snap.current_price for s, snap in snapshots.items()}
            port_state = self.portfolio_service.get_state(prices=current_prices)
            breaker_state = compute_breaker_state(
                current_equity=port_state.equity,
                peak_equity=port_state.peak_equity,
                equity_history=port_state.equity_history,
                config=self.config,
                now=cycle_dt,
                previously_paused=self.breaker_paused,
            )

            if breaker_state.is_paused:
                self.breaker_paused = True

            # If breaker tripped and configured to flatten, flatten all positions
            if breaker_state.any_active:
                logger.warning("Circuit breaker active: %s", breaker_state)
                if self.config.breaker_action == "flatten":
                    logger.warning("Breaker action is 'flatten' - flattening positions.")
                    for sym in list(self.portfolio_service.get_state().positions.keys()):
                        fill = await self.broker.close_position(
                            sym, reason="breaker_flatten"
                        )
                        if fill:
                            await self.portfolio_service.apply_close_fill(
                                fill, reason="breaker_flatten"
                            )

            cycle_state = CycleState(
                cycle_time=cycle_dt,
                expected_candle_time=get_latest_closed_candle_time(cycle_dt, "4h"),
                breaker_state=breaker_state,
                kill_switch_active=kill_switch_active,
            )

            # ── 4. Run Reflex ────────────────────────────────────────────────
            approved_orders, rejections = await run_reflex(
                cycle_id=cycle_id,
                client=self.llm_client,
                portfolio_service=self.portfolio_service,
                store=self.candle_store,
                exchange_client=self.exchange_client,
                config=self.config,
                engine=self.engine,
                cycle_state=cycle_state,
                news_provider=self.news_provider,
                now=cycle_dt,
                snapshots=snapshots if snapshots else None,
            )

            # ── 5. Apply approved orders through Broker ──────────────────────
            for order in approved_orders:
                if order.action == Action.CLOSE:
                    fill = await self.broker.close_position(
                        order.symbol,
                        reason=order.rationale or "reflex_close",
                    )
                    if fill:
                        await self.portfolio_service.apply_close_fill(
                            fill, reason="reflex_close"
                        )
                elif order.action in (
                    Action.OPEN_LONG,
                    Action.OPEN_SHORT,
                    Action.ADD,
                ):
                    client_id = f"ord_{cycle_id}_{order.symbol.replace('/', '_')}_{uuid.uuid4().hex[:6]}"
                    fill = await self.broker.place_market_order(
                        symbol=order.symbol,
                        side=order.side,
                        qty=order.quantity,
                        client_order_id=client_id,
                        invalidation_price=order.invalidation_price,
                        conviction=order.conviction,
                        risk_amount=order.risk_amount,
                    )
                    await self.portfolio_service.apply_open_fill(
                        fill=fill,
                        invalidation_price=order.invalidation_price,
                        conviction=order.conviction or ConvictionTier.B,
                        risk_amount=order.risk_amount,
                        action=order.action.value,
                    )

            # ── 6. Update protective stops ───────────────────────────────────
            for pos in self.portfolio_service.get_state().positions.values():
                await self.broker.set_protective_stop(
                    pos.symbol, pos.invalidation_price
                )

            # ── 7. Persist everything ────────────────────────────────────────
            self.completed_cycles.add(cycle_id)
            self.portfolio_service.last_cycle_id = cycle_id

            # ── 8. Send alert ────────────────────────────────────────────────
            await self.alert_manager.info(
                title=f"Reflex Cycle {cycle_id} Completed",
                message=(
                    f"Approved {len(approved_orders)} orders, rejected {len(rejections)} decisions. "
                    f"Equity: ${port_state.equity:,.2f}"
                ),
                data={
                    "cycle_id": cycle_id,
                    "approved_count": len(approved_orders),
                    "rejections_count": len(rejections),
                    "open_positions": list(
                        self.portfolio_service.get_state().positions.keys()
                    ),
                },
            )

            return approved_orders, rejections

        except Exception as exc:
            logger.exception("Unhandled exception in cycle %s: %s", cycle_id, exc)
            await self.alert_manager.critical(
                title=f"Cycle {cycle_id} Execution Failed",
                message=f"Cycle failed with error: {exc}. No orders executed.",
                data={"cycle_id": cycle_id, "error": str(exc)},
            )
            # Fail-safe: unhandled exception results in NO orders for that cycle
            return [], []
        finally:
            self._running_cycle = False

    # ── Background Loops ─────────────────────────────────────────────────────

    async def _candle_close_scheduler_loop(self) -> None:
        """Sleep until the next candle close boundary + delay, then trigger Reflex cycle."""
        delay = self.config.candle_close_delay_seconds
        tf = getattr(self.config, "decision_timeframe", "15m")
        while self._is_running:
            try:
                now = datetime.now(UTC)
                next_close = get_next_candle_boundary(now, tf)

                wake_time = next_close + timedelta(seconds=delay)
                sleep_seconds = max(1.0, (wake_time - datetime.now(UTC)).total_seconds())

                logger.debug(
                    "Scheduler sleeping %.1fs until next %s cycle (%s)",
                    sleep_seconds,
                    tf,
                    wake_time.isoformat(),
                )
                await asyncio.sleep(sleep_seconds)

                if not self._is_running:
                    break

                cycle_time = get_latest_closed_candle_time(datetime.now(UTC), tf)
                cycle_id = f"cycle_{cycle_time.strftime('%Y%m%d_%H%M')}"
                await self.execute_cycle(cycle_id=cycle_id, cycle_time=cycle_time)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in candle close scheduler loop: %s", exc)
                await asyncio.sleep(5)

    async def _daily_brain_loop(self) -> None:
        """Run daily Brain research job at 00:05 UTC."""
        while self._is_running:
            try:
                now = datetime.now(UTC)
                # Next 00:05 UTC
                target = now.replace(hour=0, minute=5, second=0, microsecond=0)
                if now >= target:
                    target += timedelta(days=1)

                sleep_seconds = max(1.0, (target - now).total_seconds())
                logger.debug("Brain job sleeping %.1fs until %s", sleep_seconds, target.isoformat())
                await asyncio.sleep(sleep_seconds)

                if not self._is_running:
                    break

                logger.info("Triggering scheduled daily Brain analysis...")
                report = await run_brain(
                    client=self.llm_client,
                    store=self.candle_store,
                    portfolio_service=self.portfolio_service,
                    engine=self.engine,
                    config=self.config,
                    news_provider=self.news_provider,
                )
                await self.alert_manager.info(
                    title="Daily Brain Research Completed",
                    message=f"Regime: {report.market_regime.value}. {report.summary}",
                    data={"market_regime": report.market_regime.value},
                )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in daily brain job loop: %s", exc)
                await asyncio.sleep(60)

    async def _invalidation_monitor_loop(self) -> None:
        """Poll every invalidation_monitor_interval_seconds for price breach."""
        interval = self.config.invalidation_monitor_interval_seconds
        while self._is_running:
            try:
                await self.check_invalidations()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in invalidation monitor loop: %s", exc)
                await asyncio.sleep(5)

    async def _equity_snapshot_loop(self) -> None:
        """Record periodic equity snapshot every equity_snapshot_interval_seconds."""
        interval = self.config.equity_snapshot_interval_seconds
        while self._is_running:
            try:
                current_prices: dict[str, float] = {}
                for sym in self.portfolio_service.get_state().positions:
                    with contextlib.suppress(Exception):
                        current_prices[sym] = await self.get_latest_price(sym)
                await self.portfolio_service.record_equity_snapshot(prices=current_prices)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in equity snapshot loop: %s", exc)
                await asyncio.sleep(10)

    async def _kill_switch_watcher_loop(self) -> None:
        """Watch for KILL_SWITCH file or flag continuously."""
        while self._is_running:
            try:
                await self.check_kill_switch()
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Error in kill-switch watcher: %s", exc)
                await asyncio.sleep(5)

    # ── Replay Mode ──────────────────────────────────────────────────────────

    async def run_replay(self, replay_dir: Path | str) -> dict[str, Any]:
        """Execute a simulated multi-day run using replayed historical candles."""
        replay_path = Path(replay_dir)
        if not replay_path.exists():
            raise OrchestratorError(f"Replay directory does not exist: {replay_path}")

        logger.info("Starting historical replay from %s...", replay_path)

        # 1. Load replay candles from directory (CSV, JSON, or SQLite)
        loaded_count = self._load_replay_candles(replay_path)
        logger.info("Loaded %d candle records for replay.", loaded_count)

        # 2. Get distinct 4h candle open timestamps sorted ascending
        from sqlalchemy import select

        from app.data.store import CandleModel

        with self.candle_store.Session() as session:
            stmt = (
                select(CandleModel.open_time)
                .where(CandleModel.timeframe == "4h")
                .distinct()
                .order_by(CandleModel.open_time.asc())
            )
            open_times = session.scalars(stmt).all()

        if not open_times:
            raise OrchestratorError("No 4h candles found in replay store.")

        # Ensure we have enough initial history (e.g. at least 60 candles) before starting cycles
        start_idx = min(60, len(open_times) - 1)
        sim_cycles = open_times[start_idx:]

        initial_equity = self.portfolio_service.compute_equity()

        for ts_ms in sim_cycles:
            candle_dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
            cycle_id = f"cycle_{candle_dt.strftime('%Y%m%d_%H%M')}"

            # Update broker prices to candle close prices
            for sym in self.config.universe:
                df = self.candle_store.get_candles(sym, "4h", now_ms=ts_ms + (4 * 3600 * 1000))
                if not df.empty:
                    last_c = float(df.iloc[-1]["close"])
                    low_c = float(df.iloc[-1]["low"])
                    high_c = float(df.iloc[-1]["high"])
                    if isinstance(self.broker, PaperBroker):
                        self.broker.set_market_price(sym, last_c)

                    # Simulate intra-candle invalidation check using candle low / high
                    positions = self.portfolio_service.get_state().positions
                    if sym in positions:
                        pos = positions[sym]
                        is_inv = (
                            pos.side == Side.LONG and low_c <= pos.invalidation_price
                        ) or (
                            pos.side == Side.SHORT and high_c >= pos.invalidation_price
                        )
                        if is_inv:
                            fill = await self.broker.close_position(sym, reason="invalidation")
                            if fill:
                                await self.portfolio_service.apply_close_fill(
                                    fill, reason="invalidation"
                                )

            # Execute reflex cycle
            await self.execute_cycle(cycle_id=cycle_id, cycle_time=candle_dt)

            # Record periodic snapshot
            prices = {
                s: await self.get_latest_price(s)
                for s in self.portfolio_service.get_state().positions
            }
            await self.portfolio_service.record_equity_snapshot(prices=prices, timestamp=candle_dt)

        # 3. Compute metrics summary
        final_equity = self.portfolio_service.compute_equity()
        total_pnl = final_equity - initial_equity
        return_pct = (total_pnl / initial_equity) * 100.0

        # Query closed trades
        from sqlalchemy import select

        from app.portfolio.db import TradeModel

        with self.portfolio_service.session_factory() as session:
            trades = session.scalars(select(TradeModel)).all()

        trade_count = len(trades)
        wins = sum(1 for t in trades if t.realized_pnl > 0)
        win_rate = (wins / trade_count * 100.0) if trade_count > 0 else 0.0

        max_dd = 0.0
        peak = initial_equity
        for pt in self.portfolio_service.get_state().equity_history:
            peak = max(peak, pt.equity)
            if peak > 0:
                dd = ((peak - pt.equity) / peak) * 100.0
                max_dd = max(max_dd, dd)

        metrics = {
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "total_pnl": total_pnl,
            "return_pct": return_pct,
            "trade_count": trade_count,
            "win_rate_pct": win_rate,
            "max_drawdown_pct": max_dd,
            "cycles_run": len(sim_cycles),
        }

        self._print_metrics_summary(metrics)
        return metrics

    def _load_replay_candles(self, replay_dir: Path) -> int:
        """Load JSON, CSV, or SQLite files from replay_dir into CandleStore."""
        import json

        total_loaded = 0

        # Check for JSON files
        json_files = list(replay_dir.glob("*.json"))
        for jf in json_files:
            try:
                data = json.loads(jf.read_text())
                # Expected format: list of candles or {"symbol": ..., "timeframe": ..., "candles": [...]}
                if isinstance(data, dict) and "candles" in data:
                    sym = data.get("symbol", "BTC/USDT")
                    tf = data.get("timeframe", "4h")
                    raw = data["candles"]
                    total_loaded += self.candle_store.upsert_candles(
                        sym, tf, raw, now_ms=int(time.time() * 1000) * 10
                    )
            except Exception as exc:
                logger.warning("Error reading %s: %s", jf, exc)

        # Check for SQLite db
        db_files = list(replay_dir.glob("*.db"))
        for dbf in db_files:
            try:
                from sqlalchemy import create_engine, select

                from app.data.store import CandleModel

                src_eng = create_engine(f"sqlite:///{dbf}")
                with src_eng.connect() as conn:
                    rows = conn.execute(
                        select(
                            CandleModel.symbol,
                            CandleModel.timeframe,
                            CandleModel.open_time,
                            CandleModel.open,
                            CandleModel.high,
                            CandleModel.low,
                            CandleModel.close,
                            CandleModel.volume,
                        )
                    ).all()
                    for r in rows:
                        self.candle_store.upsert_candles(
                            r[0],
                            r[1],
                            [[r[2], r[3], r[4], r[5], r[6], r[7]]],
                            now_ms=r[2] + 1_000_000_000,
                        )
                        total_loaded += 1
            except Exception as exc:
                logger.warning("Error importing from db %s: %s", dbf, exc)

        return total_loaded

    def _print_metrics_summary(self, metrics: dict[str, Any]) -> None:
        """Pretty-print replay performance metrics summary."""
        print("\n" + "=" * 70)
        print("          AUTONOMOUS TRADING AGENT - REPLAY PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"  Cycles Simulated:   {metrics['cycles_run']}")
        print(f"  Initial Equity:     ${metrics['initial_equity']:,.2f}")
        print(f"  Final Equity:       ${metrics['final_equity']:,.2f}")
        print(f"  Net Realized PnL:   ${metrics['total_pnl']:,.2f}")
        print(f"  Total Return:       {metrics['return_pct']:+.2f}%")
        print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
        print(f"  Total Trades:       {metrics['trade_count']}")
        print(f"  Win Rate:           {metrics['win_rate_pct']:.1f}%")
        print("=" * 70 + "\n")
