"""
TRADE EXECUTION
Handles BUY (long), SHORT (short-sell), and exit logic.
Position sizing via Kelly Criterion.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import api
from datetime import datetime

from generate_signals import (
    download_market_context_live,
    build_live_features,
    load_model,
    tickers,
)
from features import DROP_FROM_MODEL
from bot_config import (
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    KELLY_FRACTION, MAX_POSITION_PCT, WIN_LOSS_RATIO,
)


# ==================== POSITION SIZING ====================

def kelly_dollars(confidence: float, portfolio_value: float) -> float:
    """Half-Kelly Criterion: size position proportionally to edge."""
    p, q, b = confidence, 1 - confidence, WIN_LOSS_RATIO
    kelly    = max((p * b - q) / b, 0)
    dollars  = kelly * KELLY_FRACTION * portfolio_value
    return min(dollars, MAX_POSITION_PCT * portfolio_value)


# ==================== ALPACA HELPERS ====================

def get_current_positions():
    positions = {}
    for p in api.list_positions():
        qty = int(p.qty)   # negative = short
        positions[p.symbol] = {
            'qty':          qty,
            'side':         'short' if qty < 0 else 'long',
            'entry_price':  float(p.avg_entry_price),
            'current_price': float(p.current_price),
            'pnl_pct':      float(p.unrealized_plpc),
        }
    return positions


def get_pending_orders():
    return {o.symbol for o in api.list_orders(status='open')}


def cancel_stale_orders(signals):
    for order in api.list_orders(status='open'):
        sig = signals.get(order.symbol, 'HOLD')
        if order.side == 'buy'  and sig not in ('BUY',):
            api.cancel_order(order.id)
            print(f"  Cancelled stale BUY  order: {order.symbol}")
        elif order.side == 'sell' and sig not in ('SHORT',):
            api.cancel_order(order.id)
            print(f"  Cancelled stale SELL order: {order.symbol}")


def get_latest_price(ticker):
    try:
        return float(api.get_latest_trade(ticker).price)
    except Exception as e:
        print(f"  Price fetch failed for {ticker}: {e}")
        return None


# ==================== FALLBACK CONFIDENCES ====================

def get_confidences():
    """Recompute confidences from scratch (fallback when not passed in)."""
    confidences = {}
    print("  Recalculating confidences (fallback)...")
    market_ctx = download_market_context_live()
    for ticker in tickers:
        try:
            model, feature_names = load_model(ticker)
            df = build_live_features(ticker, market_ctx)
            if feature_names:
                for f in feature_names:
                    if f not in df.columns:
                        df[f] = 0
                latest = df[feature_names].iloc[[-1]]
            else:
                cols   = [c for c in df.columns if c not in DROP_FROM_MODEL]
                latest = df[cols].iloc[[-1]]
            confidences[ticker] = float(model.predict_proba(latest)[0][1])
        except Exception as e:
            print(f"  {ticker}: confidence error — {e}")
            confidences[ticker] = 0.5
    return confidences


# ==================== MAIN EXECUTION ====================

def execute_trades(signals, confidences=None):
    """
    Execute trades.
    signals:     dict  ticker -> 'BUY' | 'SHORT' | 'HOLD'
    confidences: dict  ticker -> float  (pass from generate_signals to skip re-download)
    """
    cancel_stale_orders(signals)

    positions = get_current_positions()
    pending   = get_pending_orders()
    if confidences is None:
        confidences = get_confidences()

    account = api.get_account()
    cash    = float(account.cash)
    pv      = float(account.portfolio_value)
    log     = []

    print(f"\n{'='*55}")
    print(f"  Trade Execution  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")
    print(f"  Cash: ${cash:,.2f}   Portfolio: ${pv:,.2f}")
    print(f"  Positions: {list(positions.keys()) or 'none'}")
    print(f"  Pending orders: {list(pending) or 'none'}\n")

    for ticker in tickers:
        signal     = signals.get(ticker, 'HOLD')
        confidence = confidences.get(ticker, 0.5)
        pos        = positions.get(ticker)
        action     = 'SKIP'

        # ── has an existing position ──
        if pos:
            pnl  = pos['pnl_pct']
            qty  = abs(pos['qty'])
            side = pos['side']

            # Exit conditions (same logic for long & short — Alpaca reports P&L correctly)
            if pnl <= STOP_LOSS_PCT:
                exit_side = 'buy' if side == 'short' else 'sell'
                api.submit_order(symbol=ticker, qty=qty, side=exit_side,
                                 type='market', time_in_force='day')
                action = f'EXIT {side.upper()} {qty}sh (stop-loss {pnl:.2%})'

            elif pnl >= TAKE_PROFIT_PCT:
                exit_side = 'buy' if side == 'short' else 'sell'
                api.submit_order(symbol=ticker, qty=qty, side=exit_side,
                                 type='market', time_in_force='day')
                action = f'EXIT {side.upper()} {qty}sh (take-profit {pnl:.2%})'

            # Signal reversed — exit
            elif side == 'long'  and signal == 'SHORT':
                api.submit_order(symbol=ticker, qty=qty, side='sell',
                                 type='market', time_in_force='day')
                action = f'EXIT LONG {qty}sh (signal → SHORT)'

            elif side == 'short' and signal == 'BUY':
                api.submit_order(symbol=ticker, qty=qty, side='buy',
                                 type='market', time_in_force='day')
                action = f'COVER SHORT {qty}sh (signal → BUY)'

            # No exit trigger — keep holding
            else:
                action = f'HOLD {side.upper()} {qty}sh  P&L={pnl:.2%}'

        # ── no position, BUY signal ──
        elif signal == 'BUY':
            if ticker in pending:
                action = 'SKIP (pending order exists)'
            else:
                price = get_latest_price(ticker)
                if price:
                    dollars = kelly_dollars(confidence, pv)
                    qty     = int(dollars // price)
                    if qty > 0 and cash >= qty * price:
                        api.submit_order(symbol=ticker, qty=qty, side='buy',
                                         type='market', time_in_force='day')
                        action = (f'BUY {qty}sh @ ~${price:.2f}  '
                                  f'(${dollars:,.0f}  {confidence:.0%} conf  Kelly)')
                    else:
                        action = f'SKIP BUY (need ${qty*price if price else "?":,.0f}, have ${cash:,.0f})'

        # ── no position, SHORT signal ──
        elif signal == 'SHORT':
            if ticker in pending:
                action = 'SKIP (pending order exists)'
            else:
                price = get_latest_price(ticker)
                if price:
                    dollars = kelly_dollars(1 - confidence, pv)   # confidence in DOWN
                    qty     = int(dollars // price)
                    if qty > 0:
                        api.submit_order(symbol=ticker, qty=qty, side='sell',
                                         type='market', time_in_force='day')
                        action = (f'SHORT {qty}sh @ ~${price:.2f}  '
                                  f'(${dollars:,.0f}  {1-confidence:.0%} conf DOWN  Kelly)')
                    else:
                        action = 'SKIP SHORT (position too small)'

        else:
            action = 'SKIP (no position, HOLD signal)'

        print(f"  {ticker:<6} {signal:<6}  {action}")
        log.append({
            'timestamp':  datetime.now().isoformat(),
            'ticker':     ticker,
            'signal':     signal,
            'action':     action,
            'confidence': round(confidence, 4),
        })

    return log


if __name__ == '__main__':
    test_signals = {t: 'BUY' for t in tickers[:2]}
    test_signals.update({t: 'SHORT' for t in tickers[2:4]})
    execute_trades(test_signals)
