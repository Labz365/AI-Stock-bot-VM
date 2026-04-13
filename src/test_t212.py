"""
TRADING 212 CONNECTION TEST
============================
Run this first to verify your API key and demo account are working
before running the full bot.

Usage:
    python src/test_t212.py

Steps it checks:
  1. Auth + account cash
  2. Open portfolio positions
  3. Existing pies
  4. Instrument lookup for all bot tickers
  5. Dry-run of pie generation with fake signals
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading212 import Trading212
from execute_trades_t212 import compute_pie_instruments
from bot_config import T212_API_KEY, T212_DEMO, TICKERS, BUY_THRESHOLD


def separator(title=""):
    print(f"\n{'-'*50}")
    if title:
        print(f"  {title}")
    print(f"{'-'*50}")


def main():
    env = "DEMO" if T212_DEMO else "LIVE ⚠️"
    print(f"\n{'='*50}")
    print(f"  Trading 212 Connection Test  [{env}]")
    print(f"{'='*50}")

    if T212_API_KEY == "YOUR_T212_API_KEY_HERE":
        print("\n  ERROR: Set T212_API_KEY in src/bot_config.py first.")
        return

    t212 = Trading212(T212_API_KEY, demo=T212_DEMO)

    # ── 1. Account cash ──────────────────────────────────────────────────────
    separator("1. Account Cash")
    try:
        cash = t212.get_cash()
        print(f"  Free:     ${cash.get('free',     0):>12,.2f}")
        print(f"  Invested: ${cash.get('invested',  0):>12,.2f}")
        print(f"  Total:    ${cash.get('total',     0):>12,.2f}")
        print(f"  Result:   ${cash.get('result',    0):>12,.2f}")
        print("  Auth OK")
    except Exception as e:
        print(f"  FAILED: {e}")
        return

    # ── 2. Portfolio ─────────────────────────────────────────────────────────
    separator("2. Open Positions")
    try:
        portfolio = t212.get_portfolio()
        if portfolio:
            for p in portfolio:
                avg = float(p.get("averagePrice", 0))
                cur = float(p.get("currentPrice", avg))
                pnl = (cur - avg) / avg * 100 if avg else 0
                print(f"  {p['ticker']:<22} qty={p['quantity']:.4f}  "
                      f"avg={avg:.2f}  cur={cur:.2f}  P&L={pnl:+.2f}%")
        else:
            print("  (no open positions)")
        print("  Portfolio OK")
    except Exception as e:
        print(f"  FAILED: {e}")

    # ── 3. Pies ──────────────────────────────────────────────────────────────
    separator("3. Existing Pies")
    try:
        pies = t212.get_pies()
        if pies:
            for pie in pies:
                s = pie.get("settings") or pie
                print(f"  [{s.get('id')}] {s.get('name')}")
        else:
            print("  (no pies yet)")
        print("  Pies OK")
    except Exception as e:
        print(f"  FAILED: {e}")

    # ── 4. Instrument lookup ─────────────────────────────────────────────────
    separator("4. Instrument Lookup (bot tickers)")
    all_found = True
    mapping = {}
    for symbol in TICKERS:
        t212_ticker = t212.find_ticker(symbol)
        mapping[symbol] = t212_ticker
        status = "OK" if t212_ticker else "NOT FOUND"
        print(f"  {symbol:<8} -> {t212_ticker or '-':<22} {status}")
        if t212_ticker is None:
            all_found = False

    if all_found:
        print("\n  All tickers found on T212")
    else:
        missing = [s for s, t in mapping.items() if t is None]
        print(f"\n  WARNING - Missing tickers: {missing}")
        print("     These will be skipped during execution.")

    # ── 5. Dry-run pie generation ─────────────────────────────────────────────
    separator("5. Dry-run: Pie with fake BUY signals")
    # Simulate: first half of tickers = BUY, confidence slightly above threshold
    fake_signals     = {t: ("BUY" if i < len(TICKERS) // 2 else "HOLD")
                        for i, t in enumerate(TICKERS)}
    fake_confidences = {t: (BUY_THRESHOLD + 0.05 if fake_signals[t] == "BUY" else 0.5)
                        for t in TICKERS}

    try:
        instruments = compute_pie_instruments(fake_signals, fake_confidences, t212)
        if instruments:
            total = sum(i["target"] for i in instruments)
            print(f"  Generated {len(instruments)} slice(s)  (total={total:.2f}%):")
            for inst in instruments:
                print(f"    {inst['ticker']:<22} {inst['target']:>6.2f}%")
            print("  Pie generation OK (not submitted to T212)")
        else:
            print("  (no instruments mapped - check ticker lookup above)")
    except Exception as e:
        print(f"  FAILED: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  Test complete. If all steps show ✓, you are ready to run:")
    print("  python src/execute_trades_t212.py")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
