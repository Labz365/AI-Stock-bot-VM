"""
Alpaca + Trading 212 API configuration.

Secrets are loaded from .env so they never get committed to version control.
Set LIVE_MODE=true in .env to switch from paper to live trading.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Live vs Paper mode
# ---------------------------------------------------------------------------
LIVE_MODE = os.getenv("LIVE_MODE", "false").lower() in ("true", "1", "yes")

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
BASE_URL = LIVE_URL if LIVE_MODE else PAPER_URL

# ---------------------------------------------------------------------------
# Alpaca credentials
# ---------------------------------------------------------------------------
API_KEY = os.getenv("ALPACA_API_KEY", "")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

from alpaca_trade_api import REST
api = REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "100000"))

# Quick test
if __name__ == '__main__':
    mode = "LIVE" if LIVE_MODE else "PAPER"
    print(f"Mode: {mode}")
    print(f"Base URL: {BASE_URL}")
    account = api.get_account()
    print(f"Account status: {account.status}")
    print(f"Cash: ${account.cash}")
    print(f"Portfolio value: ${account.portfolio_value}")

    positions = api.list_positions()
    if positions:
        for p in positions:
            print(f"Position: {p.symbol} | {p.qty} shares | P/L: ${p.unrealized_pl}")
    else:
        print("No open positions")
