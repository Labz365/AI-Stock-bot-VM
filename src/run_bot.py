"""
MAIN BOT RUNNER — headless, for Task Scheduler
Intraday mode (INTERVAL='1h'): schedule every 30–60 min, 9:30–16:00 ET on weekdays.
Daily mode   (INTERVAL='1d'): schedule once per day after market close.
Auto-retrains monthly (checks data/last_retrain.txt).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, date
import json
import subprocess

from generate_signals import generate_signals
from execute_trades import execute_trades
from execute_trades_t212 import execute_trades_t212
from config import api
from bot_config import INTERVAL, RETRAIN_INTERVAL_DAYS

LAST_RETRAIN_FILE = 'data/last_retrain.txt'


# ==================== AUTO-RETRAIN ====================

def should_retrain():
    """Return True if it has been >= RETRAIN_INTERVAL_DAYS since last retrain."""
    if not os.path.exists(LAST_RETRAIN_FILE):
        return True
    try:
        with open(LAST_RETRAIN_FILE) as f:
            last = date.fromisoformat(f.read().strip())
        return (date.today() - last).days >= RETRAIN_INTERVAL_DAYS
    except Exception:
        return True


def mark_retrained():
    with open(LAST_RETRAIN_FILE, 'w') as f:
        f.write(date.today().isoformat())


def run_retrain():
    print(f"\n{'='*50}")
    print("AUTO-RETRAIN: starting monthly model refresh")
    print(f"{'='*50}\n")

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python = sys.executable

    steps = [
        ('Download stock data',              [python, 'src/download_data.py']),
        ('Download alt + market context',    [python, 'src/download_all_data.py']),
        ('Build features',                   [python, 'src/build_features.py']),
        ('Train models',                     [python, 'src/train_model.py']),
    ]

    for name, cmd in steps:
        print(f"--- {name} ---")
        result = subprocess.run(cmd, cwd=base)
        if result.returncode != 0:
            print(f"ERROR: '{name}' failed (exit {result.returncode}). Aborting retrain.")
            return False

    mark_retrained()
    print(f"\nAuto-retrain complete. Next retrain in {RETRAIN_INTERVAL_DAYS} days.")
    return True


# ==================== MAIN RUN ====================

def run():
    # Use Alpaca's clock — handles weekends, holidays, and intraday market hours
    try:
        clock = api.get_clock()
        if not clock.is_open:
            next_open = clock.next_open.strftime('%Y-%m-%d %H:%M %Z')
            print(f"Market closed ({INTERVAL} mode). Next open: {next_open}")
            return
    except Exception as e:
        print(f"Clock check failed ({e}), falling back to weekend check")
        if datetime.now().weekday() >= 5:
            print("Weekend -- skipping")
            return

    print(f"{'='*50}")
    print(f"BOT RUN: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}\n")

    # Auto-retrain if due
    if should_retrain():
        success = run_retrain()
        if not success:
            print("Retrain failed — continuing with existing models.\n")

    # Account status
    account = api.get_account()
    print(f"Cash: ${account.cash}")
    print(f"Portfolio value: ${account.portfolio_value}\n")

    # Generate signals (returns confidences and market context — no re-download needed)
    print("--- Generating Signals ---")
    signals, confidences, _market_context = generate_signals()

    # Execute trades on Alpaca (paper trading)
    print("\n--- Executing Trades (Alpaca) ---")
    trade_log = execute_trades(signals, confidences=confidences)

    # Execute trades on Trading 212
    print("\n--- Executing Trades (Trading 212) ---")
    t212_log = []
    try:
        t212_log = execute_trades_t212(signals, confidences=confidences)
    except Exception as e:
        print(f"  T212 execution failed: {e}")

    # Save log
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'cash': account.cash,
        'portfolio_value': account.portfolio_value,
        'signals': signals,
        'confidences': confidences,
        'trades': trade_log,
        'trades_t212': t212_log,
    }

    log_path = 'data/trade_log.json'
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    with open(log_path, 'w') as f:
        json.dump(logs, f, indent=2)

    print(f"\nLog saved to {log_path}")
    print(f"{'='*50}")


if __name__ == '__main__':
    run()
