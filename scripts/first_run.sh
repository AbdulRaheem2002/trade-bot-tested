#!/bin/bash
set -e

echo "=================================================="
echo "    AUTONOMOUS CRYPTO TRADING AGENT - SETUP       "
echo "=================================================="

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo "[!] Python 3 is required but not found. Please install Python 3.11+."
    exit 1
fi
echo "[+] Python 3 is installed."

# 2. Check Environment
if [ ! -f .env ]; then
    echo "[+] Creating .env from .env.example..."
    cp .env.example .env
    echo "[!] Please edit .env with your real API keys later."
fi

# 3. Setup Virtual Environment
if [ ! -d .venv ]; then
    echo "[+] Creating virtual environment..."
    python3 -m venv .venv
fi
echo "[+] Activating virtual environment..."
source .venv/bin/activate
# Note: On Windows, it would be .venv/Scripts/activate, but bash scripts generally assume Linux/Mac or Git Bash.
# A safe fallback for Windows bash:
if [ ! -f .venv/bin/activate ]; then
    source .venv/Scripts/activate
fi

echo "[+] Installing dependencies..."
pip install --quiet -e .

# 4. Backfill Data and Create Database
echo "[+] Backfilling historical data for universe (BTC/USDT, ETH/USDT, SOL/USDT)..."
python scripts/backfill.py --symbols BTC/USDT ETH/USDT SOL/USDT --days 30

# 5. Run Dry Run Cycle
echo "[+] Running a dry-run cycle with MockLLM..."
python scripts/dry_run_cycle.py

# 6. Start Agent and Dashboard via Docker Compose
echo "[+] Starting agent and dashboard in paper mode via docker-compose..."
if command -v docker-compose &> /dev/null; then
    docker-compose up -d --build
elif docker compose version &> /dev/null; then
    docker compose up -d --build
else
    echo "[!] Docker Compose not found. Please install Docker to run the services."
    exit 1
fi

echo "=================================================="
echo "[+] Setup Complete!"
echo "    - Dashboard is available at: http://localhost:8000/status"
echo "    - Agent and Dashboard are running in the background."
echo "    - Check logs with: docker-compose logs -f"
echo "=================================================="
