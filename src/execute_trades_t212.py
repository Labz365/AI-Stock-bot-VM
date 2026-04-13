"""
TRADE EXECUTION — Trading 212
==============================
Drop-in replacement for execute_trades.py, targeting Trading 212's API.

Two modes (set T212_MODE in bot_config.py):

  PIE    — creates/updates a T212 Pie with Kelly-weighted slices for each BUY signal.
           T212 then auto-rebalances the pie holdings. Cleanest for testing.

  DIRECT — places individual fractional market orders for BUY signals and
           handles stop-loss / take-profit exits manually.

NOTE: T212 Invest / ISA accounts do not support short-selling.
      SHORT signals are treated as exit triggers for existing longs only.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
from datetime import datetime

from trading212 import Trading212
from bot_config import (
    TICKERS,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    KELLY_FRACTION, MAX_POSITION_PCT, WIN_LOSS_RATIO,
    T212_API_KEY, T212_DEMO, T212_MODE,
    T212_PIE_NAME, T212_PIE_INVEST_AMOUNT,
)


# ==================== POSITION SIZING ====================

def kelly_weight(confidence: float) -> float:
    """Raw (half-Kelly) fraction for a given up-confidence."""
    p, q, b = confidence, 1 - confidence, WIN_LOSS_RATIO
    return max((p * b - q) / b, 0) * KELLY_FRACTION


def compute_pie_instruments(
    signals: dict,
    confidences: dict,
    t212: Trading212,
) -> list[dict]:
    """
    Build a T212 pie instrument list from BUY signals.
    Weights are Kelly-proportional and normalised to sum to 100 %.
    Returns [] if no actionable signals or none can be mapped to T212.
    """
    buys = {t: confidences.get(t, 0.5) for t, s in signals.items() if s == "BUY"}
    if not buys:
        return []

    raw_weights = {t: kelly_weight(c) for t, c in buys.items()}
    total = sum(raw_weights.values())
    if total == 0:
        return []

    instruments = []
    for ticker, w in raw_weights.items():
        t212_ticker = t212.find_ticker(ticker)
        if t212_ticker is None:
            continue
        pct = round(w / total * 100, 2)
        instruments.append({"ticker": t212_ticker, "target": pct, "_symbol": ticker})

    if not instruments:
        return []

    # Fix rounding so targets sum to exactly 100.0
    diff = round(100.0 - sum(i["target"] for i in instruments), 2)
    instruments[0]["target"] = round(instruments[0]["target"] + diff, 2)

    # Strip helper key before sending to API
    return [{"ticker": i["ticker"], "target": i["target"]} for i in instruments]


# ==================== PIE MODE ====================

def execute_pie_mode(signals: dict, confidences: dict, t212: Trading212) -> list[dict]:
    log = []
    instruments = compute_pie_instruments(signals, confidences, t212)

    if not instruments:
        print("  No BUY signals with valid T212 instruments — pie unchanged.")
        log.append({"action": "PIE_SKIP", "reason": "no BUY signals"})
        return log

    # Print planned slices
    buys = {s: signals[s] for s in signals if signals[s] == "BUY"}
    print(f"\n  Planned pie slices ({len(instruments)} stocks):")
    for inst in instruments:
        print(f"    {inst['ticker']:<22} {inst['target']:>6.2f}%")

    # Find or create pie
    existing = t212.find_pie_by_name(T212_PIE_NAME)

    if existing:
        pie_id = existing.get("id") or existing.get("settings", {}).get("id")
        t212.update_pie(pie_id, T212_PIE_NAME, instruments)
        print(f"\n  Pie '{T212_PIE_NAME}' updated (id={pie_id})")
        log.append({"action": "PIE_UPDATE", "pie_id": pie_id, "instruments": instruments})
    else:
        result = t212.create_pie(T212_PIE_NAME, instruments)
        pie_id = (result.get("settings") or {}).get("id") or result.get("id")
        print(f"\n  Pie '{T212_PIE_NAME}' created (id={pie_id})")
        log.append({"action": "PIE_CREATE", "pie_id": pie_id, "instruments": instruments})

    return log


# ==================== DIRECT MODE ====================

def _live_price(symbol: str) -> float | None:
    """Quick yfinance spot price (reuses infrastructure already in this project)."""
    try:
        data = yf.download(symbol, period="1d", interval="1m", progress=False)
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception:
        pass
    return None


def execute_direct_mode(signals: dict, confidences: dict, t212: Trading212) -> list[dict]:
    log = []

    cash_data = t212.get_cash()
    free_cash  = float(cash_data.get("free",  0))
    total_val  = float(cash_data.get("total", free_cash))

    portfolio  = t212.get_portfolio()
    # Key by T212 ticker for fast lookup
    positions  = {p["ticker"]: p for p in portfolio}

    print(f"  Free cash: ${free_cash:,.2f}   Total: ${total_val:,.2f}")
    print(f"  Open positions: {list(positions.keys()) or 'none'}\n")

    # Pre-map all symbols to T212 tickers once (saves repeated API calls)
    ticker_map = t212.map_tickers(TICKERS)

    for symbol in TICKERS:
        signal     = signals.get(symbol, "HOLD")
        confidence = confidences.get(symbol, 0.5)
        t212_ticker = ticker_map.get(symbol)
        action = "SKIP"

        if t212_ticker is None:
            print(f"  {symbol:<6} {signal:<6}  SKIP (not on T212)")
            continue

        pos = positions.get(t212_ticker)

        # ── has an open position ──
        if pos:
            avg_px  = float(pos.get("averagePrice",  0))
            cur_px  = float(pos.get("currentPrice",  avg_px))
            qty     = float(pos.get("quantity",       0))
            pnl_pct = (cur_px - avg_px) / avg_px if avg_px > 0 else 0

            if pnl_pct <= STOP_LOSS_PCT:
                t212.place_market_sell(t212_ticker, qty)
                action = f"EXIT {qty:.4f}sh (stop-loss {pnl_pct:+.2%})"

            elif pnl_pct >= TAKE_PROFIT_PCT:
                t212.place_market_sell(t212_ticker, qty)
                action = f"EXIT {qty:.4f}sh (take-profit {pnl_pct:+.2%})"

            elif signal == "SHORT":
                # Can't short on Invest — exit long instead
                t212.place_market_sell(t212_ticker, qty)
                action = f"EXIT {qty:.4f}sh (signal → SHORT, closing long)"

            else:
                action = f"HOLD {qty:.4f}sh  P&L={pnl_pct:+.2%}"

        # ── no position, BUY signal ──
        elif signal == "BUY":
            price = _live_price(symbol)
            if price is None:
                action = "SKIP BUY (price unavailable)"
            else:
                raw_kelly = kelly_weight(confidence)
                dollars   = min(raw_kelly * total_val, MAX_POSITION_PCT * total_val)
                qty       = round(dollars / price, 4)   # fractional shares

                if qty < 0.0001:
                    action = "SKIP BUY (position too small)"
                elif free_cash < dollars:
                    action = f"SKIP BUY (need ${dollars:,.2f}, have ${free_cash:,.2f})"
                else:
                    t212.place_market_buy(t212_ticker, qty)
                    free_cash -= dollars   # update running cash total
                    action = (f"BUY {qty:.4f}sh @ ~${price:.2f}  "
                              f"(${dollars:,.0f}  {confidence:.0%} conf  Kelly)")

        # ── SHORT signal, no position ──
        elif signal == "SHORT":
            action = "SKIP SHORT (not supported on T212 Invest/ISA)"

        else:
            action = f"SKIP ({signal})"

        print(f"  {symbol:<6} {signal:<6}  {action}")
        log.append({
            "timestamp":   datetime.now().isoformat(),
            "ticker":      symbol,
            "t212_ticker": t212_ticker,
            "signal":      signal,
            "action":      action,
            "confidence":  round(confidence, 4),
        })

    return log


# ==================== ENTRY POINT ====================

def execute_trades_t212(signals: dict, confidences: dict = None) -> list[dict]:
    """
    Main entry point. Mirrors execute_trades() in execute_trades.py.
    signals:     dict  ticker -> 'BUY' | 'SHORT' | 'HOLD'
    confidences: dict  ticker -> float
    """
    t212 = Trading212(T212_API_KEY, demo=T212_DEMO)
    env  = "DEMO" if T212_DEMO else "LIVE"

    print(f"\n{'='*55}")
    print(f"  T212 Trade Execution  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Mode: {T212_MODE}  |  {env}")
    print(f"{'='*55}")

    if confidences is None:
        confidences = {}

    if T212_MODE == "PIE":
        return execute_pie_mode(signals, confidences, t212)
    else:
        return execute_direct_mode(signals, confidences, t212)


if __name__ == "__main__":
    # Smoke-test: connect and print account info
    t212 = Trading212(T212_API_KEY, demo=T212_DEMO)
    print("=== Account ===")
    print(t212.get_cash())
    print("\n=== Portfolio ===")
    for p in t212.get_portfolio():
        print(f"  {p['ticker']}  qty={p['quantity']}  avg={p['averagePrice']}  cur={p['currentPrice']}")
    print("\n=== Pies ===")
    for pie in t212.get_pies():
        s = pie.get("settings") or pie
        print(f"  [{s.get('id')}] {s.get('name')}")
