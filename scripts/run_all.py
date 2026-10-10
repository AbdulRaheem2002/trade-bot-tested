# ruff: noqa: E402
"""Docker-less local runner for Autonomous AI Trading Agent – Rule / Phase 6.

Starts the agent orchestrator, FastAPI control API, Streamlit dashboard,
and Telegram monitoring bot together with one command.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --no-bot
    python scripts/run_all.py --port 8501 --api-port 8000
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.environ["PYTHONPATH"] = str(_PROJECT_ROOT) + (os.pathsep + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "")

from app.config.loader import load_config


def print_banner(cfg: Any, api_port: int, dashboard_port: int, bot_enabled: bool) -> None:
    bot_status = "ENABLED (Polling)" if bot_enabled else "DISABLED / WAITING FOR TOKEN"
    banner = f"""
================================================================================
          AUTONOMOUS CRYPTO TRADING AGENT – UNIFIED LOCAL RUNNER
================================================================================
 [!] EXECUTION MODE:       {cfg.mode.value.upper()} (Zero real capital at risk)
 [*] STREAMLIT DASHBOARD:  http://localhost:{dashboard_port}
 [*] FASTAPI CONTROL API:  http://{cfg.dashboard_host}:{api_port}
 [*] SWAGGER DOCS:         http://{cfg.dashboard_host}:{api_port}/docs
 [*] TELEGRAM BOT:         {bot_status}
 [*] UNIVERSE:             {', '.join(cfg.universe)}
================================================================================
 Press Ctrl+C at any time to gracefully shut down all services.
================================================================================
"""
    print(banner, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified local supervisor for trading agent, dashboard, API and bot."
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="FastAPI server port (default: 8000)",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8501,
        help="Streamlit dashboard port (default: 8501)",
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Do not start the core trading agent process.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Do not start the Streamlit web dashboard.",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Do not start the FastAPI control server.",
    )
    parser.add_argument(
        "--no-bot",
        action="store_true",
        help="Do not start the Telegram bot process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN") or cfg.telegram_bot_token
    bot_enabled = bool(bot_token) and not args.no_bot

    print_banner(cfg, args.api_port, args.dashboard_port, bot_enabled)

    processes: list[tuple[str, subprocess.Popen]] = []

    def cleanup() -> None:
        print("\n[!] Shutting down all child processes...", flush=True)
        for name, proc in processes:
            if proc.poll() is None:
                print(f" -> Stopping {name} (PID {proc.pid})...", flush=True)
                proc.terminate()
        # Wait up to 5 seconds for clean termination
        start = time.time()
        for name, proc in processes:
            try:
                proc.wait(timeout=max(0.1, 5.0 - (time.time() - start)))
            except subprocess.TimeoutExpired:
                print(f" -> Force killing {name} (PID {proc.pid})...", flush=True)
                proc.kill()
        print("[!] All processes stopped. Goodbye.", flush=True)

    # Register signals for graceful termination
    def handle_signal(_signum: int, _frame: Any) -> None:
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    py_exe = sys.executable

    try:
        # 1. FastAPI Control & Monitoring API
        if not args.no_api:
            print(f"[*] Starting FastAPI API on port {args.api_port}...", flush=True)
            api_cmd = [
                py_exe,
                "-m",
                "uvicorn",
                "app.dashboard.api:app",
                "--host",
                cfg.dashboard_host,
                "--port",
                str(args.api_port),
            ]
            api_proc = subprocess.Popen(api_cmd, cwd=str(_PROJECT_ROOT), env=os.environ.copy())
            processes.append(("FastAPI API", api_proc))

        # 2. Streamlit Dashboard
        if not args.no_dashboard:
            print(
                f"[*] Starting Streamlit Dashboard on port {args.dashboard_port}...",
                flush=True,
            )
            st_cmd = [
                py_exe,
                "-m",
                "streamlit",
                "run",
                "app/dashboard/dashboard.py",
                "--server.port",
                str(args.dashboard_port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ]
            st_proc = subprocess.Popen(st_cmd, cwd=str(_PROJECT_ROOT), env=os.environ.copy())
            processes.append(("Streamlit Dashboard", st_proc))

        # 3. Core Trading Agent Orchestrator
        if not args.no_agent:
            print("[*] Starting Trading Agent Orchestrator (PAPER mode)...", flush=True)
            agent_cmd = [py_exe, "-m", "app.main", "--mode", "paper"]
            agent_proc = subprocess.Popen(agent_cmd, cwd=str(_PROJECT_ROOT), env=os.environ.copy())
            processes.append(("Trading Agent", agent_proc))

        # Monitor processes
        print("\n[+] All requested services launched successfully. Monitoring...\n", flush=True)
        while True:
            time.sleep(1.0)
            for name, proc in processes:
                ret = proc.poll()
                if ret is not None:
                    print(f"\n[!] WARNING: Process '{name}' exited with return code {ret}!", flush=True)
                    # If the core agent died, stop others
                    if name == "Trading Agent":
                        print("[!] Core trading agent terminated. Exiting runner...", flush=True)
                        cleanup()
                        sys.exit(ret)

    except KeyboardInterrupt:
        cleanup()
    except Exception as exc:
        print(f"\n[!] Error during execution: {exc}", flush=True)
        cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
