"""
BACKTESTING ENGINE
==================
Simulates the full strategy on held-out historical data (last 20%).
Uses saved models - run retrain.py first to get up-to-date models.

Usage:
    python src/backtest.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional

from features import DROP_FROM_MODEL
from bot_config import (
    TICKERS, BUY_THRESHOLD, SHORT_THRESHOLD,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    KELLY_FRACTION, MAX_POSITION_PCT, WIN_LOSS_RATIO,
    BACKTEST_INITIAL_CAPITAL, BACKTEST_TRAIN_SPLIT,
)


# ==================== DATA STRUCTURES ====================

@dataclass
class Position:
    ticker:      str
    side:        str          # 'long' | 'short'
    qty:         int
    entry_price: float
    entry_date:  object
    high_water:  float = field(init=False)
    low_water:   float = field(init=False)

    def __post_init__(self):
        self.high_water = self.entry_price
        self.low_water  = self.entry_price


@dataclass
class Trade:
    ticker:      str
    side:        str
    entry_date:  object
    exit_date:   object
    entry_price: float
    exit_price:  float
    qty:         int
    exit_reason: str

    @property
    def pnl_pct(self):
        if self.side == 'long':
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price

    @property
    def pnl_dollars(self):
        return self.pnl_pct * self.qty * self.entry_price


# ==================== HELPERS ====================

def load_model(ticker: str):
    path = f'models/{ticker}.pkl'
    if not os.path.exists(path):
        return None, None
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj['model'], obj['feature_names']
    fn = list(obj.feature_names_in_) if hasattr(obj, 'feature_names_in_') else None
    return obj, fn


def kelly_dollars(confidence: float, portfolio_value: float) -> float:
    """Half-Kelly position size in dollars."""
    p, q, b = confidence, 1 - confidence, WIN_LOSS_RATIO
    kelly = max((p * b - q) / b, 0)
    dollars = kelly * KELLY_FRACTION * portfolio_value
    return min(dollars, MAX_POSITION_PCT * portfolio_value)


def portfolio_value(cash: float, positions: dict, prices: dict) -> float:
    val = cash
    for pos in positions.values():
        px = prices.get(pos.ticker, pos.entry_price)
        val += pos.qty * px if pos.side == 'long' else -pos.qty * px
    return val


# ==================== BACKTESTER ====================

class Backtester:

    def __init__(self):
        self.models   = {}   # ticker -> (model, feature_names)
        self.data     = {}   # ticker -> DataFrame (test period, OHLCV + features)
        self.dates    = None # common test dates (pd.DatetimeIndex)

    # ---------- setup ----------

    def _load(self):
        loaded = []
        for ticker in TICKERS:
            model, feat = load_model(ticker)
            if model is None:
                print(f"  {ticker}: no model found - skipping")
                continue
            path = f'data/{ticker}_features.csv'
            if not os.path.exists(path):
                print(f"  {ticker}: no features CSV - skipping")
                continue
            df = pd.read_csv(path, parse_dates=['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            split = int(len(df) * BACKTEST_TRAIN_SPLIT)
            df_test = df.iloc[split:].copy()
            if len(df_test) < 30:
                print(f"  {ticker}: test set too small ({len(df_test)} rows) - skipping")
                continue
            self.models[ticker] = (model, feat)
            self.data[ticker]   = df_test
            loaded.append(ticker)
            print(f"  {ticker}: {len(df_test)} test rows  "
                  f"({df_test['Date'].iloc[0].date()} -> {df_test['Date'].iloc[-1].date()})")

        if not loaded:
            raise RuntimeError("No tickers loaded. Run retrain.py first.")

        # Common date range across all loaded tickers
        start = max(df['Date'].iloc[0] for df in self.data.values())
        end   = min(df['Date'].iloc[-1] for df in self.data.values())
        for t in loaded:
            df = self.data[t]
            self.data[t] = df[(df['Date'] >= start) & (df['Date'] <= end)].copy()

        self.dates = pd.to_datetime(
            sorted(set.intersection(*[set(d['Date']) for d in self.data.values()]))
        )
        print(f"\nCommon test period: {self.dates[0].date()} -> {self.dates[-1].date()}"
              f"  ({len(self.dates)} trading days)\n")

    def _row(self, ticker: str, date) -> Optional[pd.Series]:
        df = self.data[ticker]
        mask = df['Date'] == date
        if not mask.any():
            return None
        return df[mask].iloc[0]

    def _predict(self, ticker: str, row: pd.Series):
        model, feat = self.models[ticker]
        if feat is not None:
            x = {f: row.get(f, 0) for f in feat}
            arr = pd.DataFrame([x])[feat].values
        else:
            cols = [c for c in row.index if c not in DROP_FROM_MODEL]
            arr = row[cols].values.reshape(1, -1)
        prob_up = float(model.predict_proba(arr)[0][1])
        if prob_up >= BUY_THRESHOLD:
            return 'BUY', prob_up
        if prob_up <= SHORT_THRESHOLD:
            return 'SHORT', prob_up
        return 'HOLD', prob_up

    # ---------- main loop ----------

    def run(self) -> dict:
        cash       = float(BACKTEST_INITIAL_CAPITAL)
        positions  = {}   # ticker -> Position
        trades     = []
        equity     = []   # (date, value)

        for date in self.dates:
            prices_today = {}
            rows = {}
            for ticker in self.models:
                r = self._row(ticker, date)
                if r is not None:
                    rows[ticker] = r
                    prices_today[ticker] = float(r['Close'])

            # ── 1. update high/low water for open positions ──
            for ticker, pos in positions.items():
                if ticker in prices_today:
                    px = prices_today[ticker]
                    pos.high_water = max(pos.high_water, px)
                    pos.low_water  = min(pos.low_water,  px)

            # ── 2. check exits ──
            to_close = {}
            for ticker, pos in positions.items():
                if ticker not in rows:
                    continue
                row  = rows[ticker]
                high = float(row['High'])
                low  = float(row['Low'])
                close = prices_today[ticker]

                exit_px     = None
                exit_reason = None

                if pos.side == 'long':
                    stop_level   = pos.entry_price * (1 + STOP_LOSS_PCT)
                    profit_level = pos.entry_price * (1 + TAKE_PROFIT_PCT)
                    if low <= stop_level:
                        exit_px, exit_reason = stop_level, 'stop-loss'
                    elif high >= profit_level:
                        exit_px, exit_reason = profit_level, 'take-profit'
                else:  # short
                    stop_level   = pos.entry_price * (1 - STOP_LOSS_PCT)   # price went up
                    profit_level = pos.entry_price * (1 + STOP_LOSS_PCT - TAKE_PROFIT_PCT - STOP_LOSS_PCT)
                    # simpler: stop if up STOP_LOSS_PCT from entry, profit if down TAKE_PROFIT_PCT
                    stop_level   = pos.entry_price * (1 + abs(STOP_LOSS_PCT))
                    profit_level = pos.entry_price * (1 - TAKE_PROFIT_PCT)
                    if high >= stop_level:
                        exit_px, exit_reason = stop_level, 'stop-loss'
                    elif low <= profit_level:
                        exit_px, exit_reason = profit_level, 'take-profit'

                if exit_px is not None:
                    to_close[ticker] = (exit_px, exit_reason)

            # ── 3. get signals for everything (exits may override) ──
            signals = {}
            for ticker in rows:
                try:
                    sig, conf = self._predict(ticker, rows[ticker])
                    signals[ticker] = (sig, conf)
                except Exception:
                    signals[ticker] = ('HOLD', 0.5)

            # signal reversal closes
            for ticker, pos in positions.items():
                if ticker in to_close:
                    continue
                if ticker not in signals:
                    continue
                sig, _ = signals[ticker]
                if pos.side == 'long'  and sig == 'SHORT':
                    to_close[ticker] = (prices_today.get(ticker, pos.entry_price), 'signal-reversal')
                elif pos.side == 'short' and sig == 'BUY':
                    to_close[ticker] = (prices_today.get(ticker, pos.entry_price), 'signal-reversal')
                elif pos.side == 'long'  and sig == 'HOLD':
                    pass   # hold long through HOLD signal
                elif pos.side == 'short' and sig == 'HOLD':
                    pass   # hold short through HOLD signal

            # ── 4. execute closes ──
            for ticker, (exit_px, reason) in to_close.items():
                pos = positions.pop(ticker)
                if pos.side == 'long':
                    cash += pos.qty * exit_px
                else:
                    cash -= pos.qty * exit_px   # buy back to cover
                trades.append(Trade(
                    ticker=ticker, side=pos.side,
                    entry_date=pos.entry_date, exit_date=date,
                    entry_price=pos.entry_price, exit_price=exit_px,
                    qty=pos.qty, exit_reason=reason,
                ))

            # ── 5. open new positions ──
            pv = portfolio_value(cash, positions, prices_today)
            for ticker, (sig, conf) in signals.items():
                if ticker in positions or sig == 'HOLD':
                    continue
                if ticker not in prices_today:
                    continue
                px      = prices_today[ticker]
                dollars = kelly_dollars(conf, pv)
                qty     = int(dollars // px)
                if qty < 1:
                    continue

                if sig == 'BUY':
                    cost = qty * px
                    if cash >= cost:
                        cash -= cost
                        positions[ticker] = Position('long' if False else ticker,
                                                      'long', qty, px, date)
                        # fix: correct constructor
                        positions[ticker] = Position(ticker=ticker, side='long',
                                                      qty=qty, entry_price=px, entry_date=date)
                elif sig == 'SHORT':
                    # short: receive cash, owe shares
                    cash += qty * px
                    positions[ticker] = Position(ticker=ticker, side='short',
                                                  qty=qty, entry_price=px, entry_date=date)

            # ── 6. record equity ──
            equity.append({
                'date':            date,
                'portfolio_value': portfolio_value(cash, positions, prices_today),
                'cash':            cash,
                'n_positions':     len(positions),
            })

        # close all remaining positions at last price
        last_date = self.dates[-1]
        for ticker, pos in list(positions.items()):
            px = prices_today.get(ticker, pos.entry_price)
            if pos.side == 'long':
                cash += pos.qty * px
            else:
                cash -= pos.qty * px
            trades.append(Trade(
                ticker=ticker, side=pos.side,
                entry_date=pos.entry_date, exit_date=last_date,
                entry_price=pos.entry_price, exit_price=px,
                qty=pos.qty, exit_reason='end-of-backtest',
            ))

        equity_df = pd.DataFrame(equity).set_index('date')
        return {'equity': equity_df, 'trades': trades, 'final_cash': cash}

    # ---------- metrics ----------

    @staticmethod
    def metrics(results: dict) -> dict:
        eq     = results['equity']['portfolio_value']
        trades = results['trades']

        daily_ret  = eq.pct_change().dropna()
        total_ret  = eq.iloc[-1] / eq.iloc[0] - 1
        n_years    = len(eq) / 252
        cagr       = (1 + total_ret) ** (1 / max(n_years, 1e-6)) - 1
        rf_daily   = 0.04 / 252
        excess     = daily_ret - rf_daily
        sharpe     = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
        downside   = excess[excess < 0]
        sortino    = excess.mean() / downside.std() * np.sqrt(252) if len(downside) > 1 else 0

        roll_max   = eq.cummax()
        drawdown   = (eq - roll_max) / roll_max
        max_dd     = drawdown.min()

        completed  = [t for t in trades if t.exit_reason != 'end-of-backtest']
        wins       = [t for t in completed if t.pnl_pct > 0]
        losses     = [t for t in completed if t.pnl_pct <= 0]
        win_rate   = len(wins) / len(completed) if completed else 0
        avg_win    = np.mean([t.pnl_pct for t in wins])   if wins   else 0
        avg_loss   = np.mean([t.pnl_pct for t in losses]) if losses else 0
        gross_win  = sum(t.pnl_dollars for t in wins)
        gross_loss = abs(sum(t.pnl_dollars for t in losses))
        pf         = gross_win / gross_loss if gross_loss > 0 else float('inf')

        longs  = [t for t in completed if t.side == 'long']
        shorts = [t for t in completed if t.side == 'short']

        return dict(
            total_return=total_ret, cagr=cagr,
            sharpe=sharpe, sortino=sortino, max_drawdown=max_dd,
            n_trades=len(completed), win_rate=win_rate,
            avg_win=avg_win, avg_loss=avg_loss, profit_factor=pf,
            n_longs=len(longs), n_shorts=len(shorts),
            long_win_rate=len([t for t in longs if t.pnl_pct>0])/len(longs) if longs else 0,
            short_win_rate=len([t for t in shorts if t.pnl_pct>0])/len(shorts) if shorts else 0,
        )

    # ---------- plot ----------

    @staticmethod
    def plot(results: dict):
        import matplotlib.ticker as mticker
        import matplotlib.patches as mpatches

        eq     = results['equity']['portfolio_value']
        trades = results['trades']
        done   = [t for t in trades if t.exit_reason != 'end-of-backtest']

        BLUE   = '#2196F3'
        GREEN  = '#4CAF50'
        RED    = '#F44336'
        GREY   = '#9E9E9E'
        ORANGE = '#FF9800'
        PURPLE = '#9C27B0'
        BG     = '#FAFAFA'

        # ── SPY benchmark ──
        spy_bh = None
        if os.path.exists('data/SPY.csv'):
            spy = pd.read_csv('data/SPY.csv', parse_dates=[0])
            spy = spy.rename(columns={spy.columns[0]: 'Date'}).set_index('Date')['Close']
            spy = spy.reindex(eq.index, method='ffill').dropna()
            if len(spy) > 10:
                spy_bh = spy / spy.iloc[0] * float(BACKTEST_INITIAL_CAPITAL)

        # ── derived series ──
        daily_ret  = eq.pct_change().dropna() * 100
        roll_max   = eq.cummax()
        dd         = (eq - roll_max) / roll_max * 100
        rolling_sh = (daily_ret.rolling(30).mean() / daily_ret.rolling(30).std()
                      * np.sqrt(252)).dropna()

        # monthly returns for heatmap
        monthly = eq.resample('ME').last().pct_change().dropna() * 100
        monthly_df = pd.DataFrame({
            'year':  monthly.index.year,
            'month': monthly.index.month,
            'ret':   monthly.values,
        })

        # per-ticker P&L
        ticker_pnl = {}
        for t in done:
            ticker_pnl.setdefault(t.ticker, []).append(t.pnl_dollars)
        ticker_totals = {k: sum(v) for k, v in ticker_pnl.items()}

        # exit reason breakdown
        reason_counts = {}
        for t in done:
            reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1

        # ── figure layout: 3×2 grid ──
        plt.style.use('seaborn-v0_8-whitegrid')
        fig = plt.figure(figsize=(18, 14), facecolor=BG)
        fig.suptitle('AI Stock Bot - Backtest Dashboard', fontsize=16,
                     fontweight='bold', y=0.98)

        gs = gridspec.GridSpec(3, 2, figure=fig,
                               height_ratios=[2.2, 1.2, 1.2],
                               hspace=0.45, wspace=0.32)

        # ── [0,0] Equity curve ──
        ax_eq = fig.add_subplot(gs[0, 0])
        ax_eq.set_facecolor(BG)
        ax_eq.plot(eq.index, eq.values, color=BLUE, linewidth=2, label='Strategy', zorder=3)
        if spy_bh is not None:
            ax_eq.plot(spy_bh.index, spy_bh.values, color=GREY,
                       linewidth=1.3, linestyle='--', alpha=0.8, label='SPY B&H', zorder=2)
        # shade above/below initial capital
        ax_eq.axhline(BACKTEST_INITIAL_CAPITAL, color=GREY, linewidth=0.6,
                      linestyle=':', alpha=0.5)
        ax_eq.fill_between(eq.index, eq.values, BACKTEST_INITIAL_CAPITAL,
                           where=eq.values >= BACKTEST_INITIAL_CAPITAL,
                           alpha=0.08, color=GREEN, interpolate=True)
        ax_eq.fill_between(eq.index, eq.values, BACKTEST_INITIAL_CAPITAL,
                           where=eq.values < BACKTEST_INITIAL_CAPITAL,
                           alpha=0.08, color=RED, interpolate=True)
        # winning/losing trade markers
        for t in done:
            try:
                y = eq.asof(pd.Timestamp(t.exit_date))
                c = GREEN if t.pnl_pct > 0 else RED
                m = '^' if t.side == 'long' else 'v'
                ax_eq.scatter(t.exit_date, y, color=c, marker=m, s=22, zorder=5, alpha=0.7)
            except Exception:
                pass
        ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax_eq.set_title('Equity Curve', fontweight='bold')
        ax_eq.set_ylabel('Portfolio Value ($)')
        leg_patches = [
            mpatches.Patch(color=GREEN, label='Win exit'),
            mpatches.Patch(color=RED,   label='Loss exit'),
        ]
        handles, labels = ax_eq.get_legend_handles_labels()
        ax_eq.legend(handles=handles + leg_patches, fontsize=8, loc='upper left')

        # ── [0,1] Monthly returns heatmap ──
        ax_heat = fig.add_subplot(gs[0, 1])
        ax_heat.set_facecolor(BG)
        if len(monthly_df) >= 3:
            years  = sorted(monthly_df['year'].unique())
            months = list(range(1, 13))
            month_abbr = ['Jan','Feb','Mar','Apr','May','Jun',
                          'Jul','Aug','Sep','Oct','Nov','Dec']
            heat = np.full((len(years), 12), np.nan)
            for _, row in monthly_df.iterrows():
                yi = years.index(row['year'])
                mi = int(row['month']) - 1
                heat[yi, mi] = row['ret']
            vmax = max(abs(np.nanmin(heat)), abs(np.nanmax(heat)), 1)
            im = ax_heat.imshow(heat, cmap='RdYlGn', vmin=-vmax, vmax=vmax,
                                aspect='auto')
            ax_heat.set_xticks(range(12))
            ax_heat.set_xticklabels(month_abbr, fontsize=8)
            ax_heat.set_yticks(range(len(years)))
            ax_heat.set_yticklabels(years, fontsize=8)
            for yi in range(len(years)):
                for mi in range(12):
                    val = heat[yi, mi]
                    if not np.isnan(val):
                        ax_heat.text(mi, yi, f'{val:.1f}%', ha='center', va='center',
                                     fontsize=6.5, color='black',
                                     fontweight='bold' if abs(val) > vmax * 0.6 else 'normal')
            plt.colorbar(im, ax=ax_heat, shrink=0.8, label='Return %')
        ax_heat.set_title('Monthly Returns Heatmap', fontweight='bold')

        # ── [1,0] Drawdown ──
        ax_dd = fig.add_subplot(gs[1, 0])
        ax_dd.set_facecolor(BG)
        ax_dd.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.35)
        ax_dd.plot(dd.index, dd.values, color=RED, linewidth=0.9)
        ax_dd.set_title('Drawdown (%)', fontweight='bold')
        ax_dd.set_ylabel('%')
        ax_dd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.1f}%'))
        # annotate max drawdown
        min_idx = dd.idxmin()
        ax_dd.annotate(f'{dd.min():.1f}%', xy=(min_idx, dd.min()),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, color=RED,
                       arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))

        # ── [1,1] Rolling 30-day Sharpe ──
        ax_sh = fig.add_subplot(gs[1, 1])
        ax_sh.set_facecolor(BG)
        ax_sh.plot(rolling_sh.index, rolling_sh.values, color=PURPLE, linewidth=1.2)
        ax_sh.axhline(0,   color='black', linewidth=0.6, linestyle='--', alpha=0.4)
        ax_sh.axhline(1,   color=GREEN,   linewidth=0.6, linestyle=':',  alpha=0.6, label='Sharpe=1')
        ax_sh.axhline(-1,  color=RED,     linewidth=0.6, linestyle=':',  alpha=0.6)
        ax_sh.fill_between(rolling_sh.index, rolling_sh.values, 0,
                           where=rolling_sh.values > 0, alpha=0.1, color=GREEN, interpolate=True)
        ax_sh.fill_between(rolling_sh.index, rolling_sh.values, 0,
                           where=rolling_sh.values < 0, alpha=0.1, color=RED, interpolate=True)
        ax_sh.set_title('Rolling 30-Day Sharpe Ratio (annualised)', fontweight='bold')
        ax_sh.legend(fontsize=8)

        # ── [2,0] Trade P&L distribution ──
        ax_pnl = fig.add_subplot(gs[2, 0])
        ax_pnl.set_facecolor(BG)
        if done:
            pnl_pcts = [t.pnl_pct * 100 for t in done]
            wins_v   = [v for v in pnl_pcts if v > 0]
            loss_v   = [v for v in pnl_pcts if v <= 0]
            bins = np.linspace(min(pnl_pcts) - 0.1, max(pnl_pcts) + 0.1, 40)
            ax_pnl.hist(wins_v, bins=bins, color=GREEN, alpha=0.7, label=f'Wins ({len(wins_v)})')
            ax_pnl.hist(loss_v, bins=bins, color=RED,   alpha=0.7, label=f'Losses ({len(loss_v)})')
            ax_pnl.axvline(0, color='black', linewidth=1)
            ax_pnl.axvline(np.mean(pnl_pcts), color=ORANGE, linewidth=1.4,
                           linestyle='--', label=f'Mean {np.mean(pnl_pcts):.2f}%')
        ax_pnl.set_title('Trade P&L Distribution', fontweight='bold')
        ax_pnl.set_xlabel('Return per Trade (%)')
        ax_pnl.legend(fontsize=8)

        # ── [2,1] Per-ticker total P&L bars ──
        ax_tkr = fig.add_subplot(gs[2, 1])
        ax_tkr.set_facecolor(BG)
        if ticker_totals:
            tickers_sorted = sorted(ticker_totals, key=ticker_totals.get)
            vals   = [ticker_totals[t] for t in tickers_sorted]
            colors = [GREEN if v >= 0 else RED for v in vals]
            bars   = ax_tkr.barh(tickers_sorted, vals, color=colors, alpha=0.8, edgecolor='none')
            ax_tkr.axvline(0, color='black', linewidth=0.8)
            for bar, val in zip(bars, vals):
                ax_tkr.text(val + (max(abs(v) for v in vals) * 0.01),
                            bar.get_y() + bar.get_height() / 2,
                            f'${val:+,.0f}', va='center', fontsize=8,
                            color='black')
            ax_tkr.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        ax_tkr.set_title('Total P&L by Ticker', fontweight='bold')

        os.makedirs('results', exist_ok=True)
        plt.savefig('results/backtest_results.png', dpi=150, bbox_inches='tight',
                    facecolor=BG)
        print("Dashboard saved -> results/backtest_results.png")

        # Save individual charts so the GUI can show one at a time
        _chart_exports = [
            (ax_eq,   'chart_equity'),
            (ax_heat, 'chart_heatmap'),
            (ax_dd,   'chart_drawdown'),
            (ax_sh,   'chart_sharpe'),
            (ax_pnl,  'chart_pnl_dist'),
            (ax_tkr,  'chart_ticker_pnl'),
        ]
        for ax, name in _chart_exports:
            try:
                extent = ax.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted())
                fig.savefig(os.path.join('results', f'{name}.png'),
                            bbox_inches=extent.expanded(1.12, 1.25),
                            dpi=150, facecolor=BG)
            except Exception:
                pass

        plt.close()

    # ---------- report ----------

    def report(self, results: dict):
        m  = self.metrics(results)
        eq = results['equity']['portfolio_value']

        print("\n" + "=" * 50)
        print("  BACKTEST RESULTS")
        print("=" * 50)
        print(f"  Period       {self.dates[0].date()} -> {self.dates[-1].date()}")
        print(f"  Tickers      {', '.join(self.models.keys())}")
        print("-" * 50)
        print(f"  Initial      ${BACKTEST_INITIAL_CAPITAL:>12,.0f}")
        print(f"  Final        ${eq.iloc[-1]:>12,.0f}")
        print(f"  Total Return {m['total_return']:>+11.1%}")
        print(f"  CAGR         {m['cagr']:>+11.1%}")
        print("-" * 50)
        print(f"  Sharpe       {m['sharpe']:>12.2f}")
        print(f"  Sortino      {m['sortino']:>12.2f}")
        print(f"  Max Drawdown {m['max_drawdown']:>11.1%}")
        print("-" * 50)
        print(f"  Trades       {m['n_trades']:>12d}  "
              f"(long={m['n_longs']}, short={m['n_shorts']})")
        print(f"  Win Rate     {m['win_rate']:>11.1%}  "
              f"(L={m['long_win_rate']:.0%}  S={m['short_win_rate']:.0%})")
        print(f"  Avg Win      {m['avg_win']:>+11.2%}")
        print(f"  Avg Loss     {m['avg_loss']:>+11.2%}")
        print(f"  Profit Factor{m['profit_factor']:>12.2f}")
        print("=" * 50)

        # Save metrics JSON for GUI display
        os.makedirs('results', exist_ok=True)
        def _safe(v):
            if isinstance(v, float):
                import math
                return None if (math.isinf(v) or math.isnan(v)) else v
            return v
        m_out = {k: _safe(v) for k, v in m.items()}
        m_out['initial']      = float(BACKTEST_INITIAL_CAPITAL)
        m_out['final']        = float(eq.iloc[-1])
        m_out['period_start'] = str(self.dates[0].date())
        m_out['period_end']   = str(self.dates[-1].date())
        import json as _json
        with open(os.path.join('results', 'backtest_metrics.json'), 'w') as _f:
            _json.dump(m_out, _f, indent=2)

        # per-ticker breakdown
        trades = [t for t in results['trades'] if t.exit_reason != 'end-of-backtest']
        print("\n  Per-ticker summary:")
        print(f"  {'Ticker':<8}  {'Trades':>6}  {'Win%':>6}  {'Avg P&L':>8}  {'Total $':>10}")
        print("  " + "-" * 45)
        for ticker in sorted(self.models.keys()):
            tt = [t for t in trades if t.ticker == ticker]
            if not tt:
                continue
            wr  = sum(1 for t in tt if t.pnl_pct > 0) / len(tt)
            avg = np.mean([t.pnl_pct for t in tt])
            tot = sum(t.pnl_dollars for t in tt)
            print(f"  {ticker:<8}  {len(tt):>6}  {wr:>5.0%}  {avg:>+7.2%}  ${tot:>9,.0f}")


# ==================== ENTRY POINT ====================

if __name__ == '__main__':
    bt = Backtester()

    print("Loading models and feature data...")
    bt._load()

    print("Running simulation...")
    results = bt.run()

    bt.report(results)
    bt.plot(results)