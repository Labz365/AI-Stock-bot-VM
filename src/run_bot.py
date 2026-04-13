"""
MAIN BOT RUNNER — continuous process
Runs indefinitely, trading when the market is open and idling when closed.
Replaces the old Task Scheduler approach.

Intraday mode (INTERVAL='1h'): executes a trading cycle every LOOP_INTERVAL_SECONDS
                                while the market is open.
Daily mode   (INTERVAL='1d'): executes once per session, then idles until next open.

Auto-retrains monthly (checks data/last_retrain.txt).
Graceful shutdown via Ctrl+C or SIGTERM.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import signal
import time
import tempfile
from datetime import datetime, date
import json
import subprocess

from generate_signals import generate_signals
from execute_trades import execute_trades
from execute_trades_t212 import execute_trades_t212
from config import api
from bot_config import (
    INTERVAL, RETRAIN_INTERVAL_DAYS,
    LOOP_INTERVAL_SECONDS, IDLE_POLL_SECONDS,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'bot.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-7s %(name)-12s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger('stock-bot')

LAST_RETRAIN_FILE = os.path.join(LOG_DIR, 'last_retrain.txt')
TRADE_LOG_PATH = os.path.join(LOG_DIR, 'trade_log.json')

# ---------------------------------------------------------------------------
# Shutdown flag
# ---------------------------------------------------------------------------
_stop = False


def _handle_shutdown(*_):
    global _stop
    log.info("Shutdown requested — finishing current cycle")
    _stop = True


# ==================== MARKET HOURS ====================

def get_market_status():
    """
    Check if the market is open via Alpaca's clock API.
    Returns (is_open: bool, seconds_until_open: float or None).
    """
    try:
        clock = api.get_clock()
        if clock.is_open:
            return True, None

        # Calculate seconds until market opens
        now = clock.timestamp
        next_open = clock.next_open
        delta = (next_open - now).total_seconds()
        return False, max(delta, 0)
    except Exception as e:
        log.warning("Clock check failed (%s), falling back to weekday check", e)
        if datetime.now().weekday() >= 5:
            return False, None
        return True, None  # assume open on weekdays if clock fails


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
    log.info("AUTO-RETRAIN: starting monthly model refresh")

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python = sys.executable

    steps = [
        ('Download stock data',           [python, 'src/download_data.py']),
        ('Download alt + market context', [python, 'src/download_all_data.py']),
        ('Build features',                [python, 'src/build_features.py']),
        ('Train models',                  [python, 'src/train_model.py']),
    ]

    for name, cmd in steps:
        log.info("Retrain step: %s", name)
        result = subprocess.run(cmd, cwd=base)
        if result.returncode != 0:
            log.error("Retrain step '%s' failed (exit %d). Aborting.", name, result.returncode)
            return False

    mark_retrained()
    log.info("Auto-retrain complete. Next retrain in %d days.", RETRAIN_INTERVAL_DAYS)
    return True


# ==================== ATOMIC LOG HELPERS ====================

def _append_trade_log(entry):
    """Append an entry to trade_log.json atomically (temp file + rename)."""
    if os.path.exists(TRADE_LOG_PATH):
        try:
            with open(TRADE_LOG_PATH, 'r') as f:
                logs = json.load(f)
        except (json.JSONDecodeError, OSError):
            logs = []
    else:
        logs = []

    logs.append(entry)

    # Write to a temp file in the same directory, then rename.
    # os.replace is atomic on the same filesystem.
    dir_name = os.path.dirname(TRADE_LOG_PATH)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(logs, f, indent=2)
        os.replace(tmp_path, TRADE_LOG_PATH)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _save_equity_snapshot():
    """Save a portfolio equity snapshot without running a full trading cycle."""
    try:
        account = api.get_account()
        entry = {
            'timestamp': datetime.now().isoformat(),
            'cash': float(account.cash),
            'portfolio_value': float(account.portfolio_value),
            'signals': {},
            'confidences': {},
            'trades': [],
            'trades_t212': [],
        }
        _append_trade_log(entry)
        log.info("Equity snapshot: cash=$%.2f  portfolio=$%.2f",
                 entry['cash'], entry['portfolio_value'])
    except Exception as e:
        log.warning("Equity snapshot failed: %s", e)


# ==================== SINGLE TRADING CYCLE ====================

def run_cycle():
    """Execute one full trading cycle (signals + trades + log)."""
    log.info("=" * 50)
    log.info("TRADING CYCLE: %s", datetime.now().strftime('%Y-%m-%d %H:%M'))
    log.info("=" * 50)

    # Auto-retrain if due
    if should_retrain():
        success = run_retrain()
        if not success:
            log.warning("Retrain failed — continuing with existing models.")

    # Account status
    account = api.get_account()
    log.info("Cash: $%s  Portfolio: $%s", account.cash, account.portfolio_value)

    # Generate signals
    log.info("Generating signals...")
    signals, confidences, _market_context = generate_signals()

    # Execute trades on Alpaca
    log.info("Executing trades (Alpaca)...")
    trade_log = execute_trades(signals, confidences=confidences)

    # Execute trades on Trading 212
    log.info("Executing trades (Trading 212)...")
    t212_log = []
    try:
        t212_log = execute_trades_t212(signals, confidences=confidences)
    except Exception as e:
        log.warning("T212 execution failed: %s", e)

    # Save log entry
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'cash': float(account.cash),
        'portfolio_value': float(account.portfolio_value),
        'signals': signals,
        'confidences': confidences,
        'trades': trade_log,
        'trades_t212': t212_log,
    }

    _append_trade_log(log_entry)
    log.info("Log saved. Cycle complete.")


# ==================== MAIN LOOP ====================

def run_forever():
    """
    Continuous main loop.

    When market is open:  run a trading cycle, sleep LOOP_INTERVAL_SECONDS.
    When market is closed: log status, sleep until market opens (or poll
                           every IDLE_POLL_SECONDS if we can't determine
                           the exact open time).
    Daily mode ('1d'):     run one cycle per session, then idle until next open.
    """
    log.info("Stock bot started — continuous mode (interval=%s, loop=%ds)",
             INTERVAL, LOOP_INTERVAL_SECONDS)

    ran_today = False  # for daily mode: only run once per session

    while not _stop:
        is_open, seconds_until_open = get_market_status()

        if not is_open:
            ran_today = False  # reset daily flag when market closes

            if seconds_until_open and seconds_until_open > 0:
                # Sleep in chunks so we can still respond to shutdown signals
                wait = min(seconds_until_open, IDLE_POLL_SECONDS)
                hrs = seconds_until_open / 3600
                log.info("Market closed. Next open in %.1f hours. Sleeping %ds...",
                         hrs, int(wait))
                time.sleep(wait)
            else:
                log.info("Market closed. Polling again in %ds...", IDLE_POLL_SECONDS)
                time.sleep(IDLE_POLL_SECONDS)
            continue

        # Market is open
        if INTERVAL == '1d' and ran_today:
            # Daily mode: already ran this session, record equity and idle
            _save_equity_snapshot()
            log.info("Daily mode: already ran this session. Idling %ds...",
                     IDLE_POLL_SECONDS)
            time.sleep(IDLE_POLL_SECONDS)
            continue

        try:
            run_cycle()
            ran_today = True
        except Exception as e:
            log.exception("Trading cycle error: %s", e)

        if not _stop:
            log.info("Next cycle in %ds...", LOOP_INTERVAL_SECONDS)
            time.sleep(LOOP_INTERVAL_SECONDS)

    log.info("Stock bot stopped.")


# ==================== ENTRY POINT ====================

def run():
    """Single-shot run (backwards compatible with old Task Scheduler usage)."""
    is_open, _ = get_market_status()
    if not is_open:
        log.info("Market closed (%s mode). Skipping.", INTERVAL)
        return
    run_cycle()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    if '--once' in sys.argv:
        run()  # single-shot for backwards compat
    else:
        run_forever()  # default: continuous
