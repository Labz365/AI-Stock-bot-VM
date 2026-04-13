"""
AI STOCK BOT — GUI
==================
Monitor performance and edit settings.
Task Scheduler continues to run run_bot.py independently.

Usage:
    python src/gui.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import ttk, messagebox, font
import json
import re
import subprocess
import threading
from datetime import datetime
from PIL import Image, ImageTk

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, 'src', 'bot_config.py')
LOG_PATH    = os.path.join(ROOT, 'data', 'trade_log.json')
BACKTEST_IMG     = os.path.join(ROOT, 'results', 'backtest_results.png')
BACKTEST_METRICS = os.path.join(ROOT, 'results', 'backtest_metrics.json')
BACKTEST_CHART_FILES = [
    (os.path.join(ROOT, 'results', 'chart_equity.png'),     'Equity Curve'),
    (os.path.join(ROOT, 'results', 'chart_heatmap.png'),    'Monthly Returns Heatmap'),
    (os.path.join(ROOT, 'results', 'chart_drawdown.png'),   'Drawdown (%)'),
    (os.path.join(ROOT, 'results', 'chart_sharpe.png'),     'Rolling 30-Day Sharpe'),
    (os.path.join(ROOT, 'results', 'chart_pnl_dist.png'),   'Trade P&L Distribution'),
    (os.path.join(ROOT, 'results', 'chart_ticker_pnl.png'), 'Total P&L by Ticker'),
]

# ── colours ────────────────────────────────────────────────────────────────
BG       = '#1E1E2E'
SURFACE  = '#2A2A3E'
ACCENT   = '#7C3AED'
ACCENT2  = '#06B6D4'
GREEN    = '#22C55E'
RED      = '#EF4444'
ORANGE   = '#F97316'
FG       = '#E2E8F0'
FG_DIM   = '#94A3B8'
BORDER   = '#3F3F5F'


# ==================== CONFIG READ/WRITE ====================

def read_config() -> dict:
    """Parse key=value pairs from bot_config.py."""
    with open(CONFIG_PATH, 'r') as f:
        src = f.read()

    cfg = {}
    patterns = {
        'INTERVAL':               r"INTERVAL\s*=\s*'([^']+)'",
        'TICKERS':                r"TICKERS\s*=\s*\[([^\]]+)\]",
        'BUY_THRESHOLD':          r"BUY_THRESHOLD\s*=\s*([\d.]+)",
        'SHORT_THRESHOLD':        r"SHORT_THRESHOLD\s*=\s*([\d.]+)",
        'STOP_LOSS_PCT':          r"STOP_LOSS_PCT\s*=\s*(-?[\d.]+)",
        'TAKE_PROFIT_PCT':        r"TAKE_PROFIT_PCT\s*=\s*([\d.]+)",
        'KELLY_FRACTION':         r"KELLY_FRACTION\s*=\s*([\d.]+)",
        'MAX_POSITION_PCT':       r"MAX_POSITION_PCT\s*=\s*([\d.]+)",
        'RETRAIN_INTERVAL_DAYS':  r"RETRAIN_INTERVAL_DAYS\s*=\s*(\d+)",
        'BACKTEST_INITIAL_CAPITAL': r"BACKTEST_INITIAL_CAPITAL\s*=\s*([\d_]+)",
        'T212_API_KEY':           r"T212_API_KEY\s*=\s*'([^']+)'",
        'T212_DEMO':              r"T212_DEMO\s*=\s*(True|False)",
        'T212_MODE':              r"T212_MODE\s*=\s*'([^']+)'",
        'T212_PIE_NAME':          r"T212_PIE_NAME\s*=\s*'([^']+)'",
    }
    for key, pat in patterns.items():
        m = re.search(pat, src)
        if m:
            cfg[key] = m.group(1).strip()

    # Clean up TICKERS list
    if 'TICKERS' in cfg:
        tickers = re.findall(r"'([^']+)'", cfg['TICKERS'])
        cfg['TICKERS'] = ', '.join(tickers)

    return cfg


def write_config(updates: dict):
    """Write updated values back into bot_config.py."""
    with open(CONFIG_PATH, 'r') as f:
        src = f.read()

    replacements = {
        'INTERVAL':               (r"(INTERVAL\s*=\s*)'[^']+'",          lambda v: f"'{{v}}'"),
        'BUY_THRESHOLD':          (r"(BUY_THRESHOLD\s*=\s*)[\d.]+",      None),
        'SHORT_THRESHOLD':        (r"(SHORT_THRESHOLD\s*=\s*)[\d.]+",    None),
        'STOP_LOSS_PCT':          (r"(STOP_LOSS_PCT\s*=\s*)-?[\d.]+",    None),
        'TAKE_PROFIT_PCT':        (r"(TAKE_PROFIT_PCT\s*=\s*)[\d.]+",    None),
        'KELLY_FRACTION':         (r"(KELLY_FRACTION\s*=\s*)[\d.]+",     None),
        'MAX_POSITION_PCT':       (r"(MAX_POSITION_PCT\s*=\s*)[\d.]+",   None),
        'RETRAIN_INTERVAL_DAYS':  (r"(RETRAIN_INTERVAL_DAYS\s*=\s*)\d+", None),
        'BACKTEST_INITIAL_CAPITAL': (r"(BACKTEST_INITIAL_CAPITAL\s*=\s*)[\d_]+", None),
        'T212_API_KEY':           (r"(T212_API_KEY\s*=\s*)'[^']+'",      lambda v: f"'{{v}}'"),
        'T212_DEMO':              (r"(T212_DEMO\s*=\s*)(True|False)",     None),
        'T212_MODE':              (r"(T212_MODE\s*=\s*)'[^']+'",         lambda v: f"'{{v}}'"),
        'T212_PIE_NAME':          (r"(T212_PIE_NAME\s*=\s*)'[^']+'",     lambda v: f"'{{v}}'"),
    }

    for key, val in updates.items():
        if key == 'TICKERS':
            # Rebuild list
            tickers = [t.strip().strip("'\"") for t in val.split(',') if t.strip()]
            list_str = ', '.join(f"'{t}'" for t in tickers)
            src = re.sub(
                r"(TICKERS\s*=\s*\[)[^\]]+(\])",
                lambda m: f"{m.group(1)}{list_str}{m.group(2)}",
                src
            )
            continue

        if key not in replacements:
            continue

        pat, fmt = replacements[key]
        if fmt:
            replacement = r'\g<1>' + fmt('').format(v=val)
        else:
            replacement = r'\g<1>' + str(val)
        src = re.sub(pat, replacement, src)

    with open(CONFIG_PATH, 'w') as f:
        f.write(src)


# ==================== LOAD DATA ====================

def load_trade_log() -> list:
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH) as f:
            return json.load(f)
    except Exception:
        return []


# ==================== MAIN GUI ====================

class BotGUI(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('AI Stock Bot')
        self.geometry('1100x720')
        self.configure(bg=BG)
        self.resizable(True, True)

        self._bot_process      = None
        self._backtest_process = None
        self._after_id         = None
        self._backtest_img     = None
        self._chart_index      = 0

        self._build_styles()
        self._build_header()
        self._build_tabs()
        self._refresh()        # initial data load

        self.protocol('WM_DELETE_WINDOW', self._on_close)

    # ── styles ────────────────────────────────────────────────────────────

    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')

        style.configure('.',          background=BG,      foreground=FG,  font=('Segoe UI', 10))
        style.configure('TNotebook',  background=BG,      borderwidth=0)
        style.configure('TNotebook.Tab', background=SURFACE, foreground=FG_DIM,
                        padding=[16, 8], font=('Segoe UI', 10))
        style.map('TNotebook.Tab',
                  background=[('selected', ACCENT)],
                  foreground=[('selected', '#FFFFFF')])
        style.configure('TFrame',     background=BG)
        style.configure('Card.TFrame', background=SURFACE, relief='flat')
        style.configure('TLabel',     background=BG,      foreground=FG)
        style.configure('Card.TLabel', background=SURFACE, foreground=FG)
        style.configure('Dim.TLabel', background=BG,      foreground=FG_DIM, font=('Segoe UI', 9))
        style.configure('Dim2.TLabel', background=SURFACE, foreground=FG_DIM, font=('Segoe UI', 9))
        style.configure('H1.TLabel',  background=BG,      foreground=FG,
                        font=('Segoe UI', 18, 'bold'))
        style.configure('H2.TLabel',  background=SURFACE, foreground=FG,
                        font=('Segoe UI', 12, 'bold'))
        style.configure('TEntry',     fieldbackground=SURFACE, foreground=FG,
                        insertcolor=FG, bordercolor=BORDER)
        style.configure('Treeview',   background=SURFACE, foreground=FG,
                        fieldbackground=SURFACE, rowheight=26,
                        font=('Segoe UI', 9))
        style.configure('Treeview.Heading', background=ACCENT, foreground='white',
                        font=('Segoe UI', 9, 'bold'), relief='flat')
        style.map('Treeview', background=[('selected', ACCENT)])
        style.configure('TScrollbar', background=SURFACE, troughcolor=BG,
                        arrowcolor=FG_DIM)

    # ── header ────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self, bg=SURFACE, pady=10)
        hdr.pack(fill='x', padx=0)

        tk.Label(hdr, text='AI Stock Bot', bg=SURFACE, fg=FG,
                 font=('Segoe UI', 16, 'bold')).pack(side='left', padx=20)

        self._status_lbl = tk.Label(hdr, text='', bg=SURFACE, fg=FG_DIM,
                                    font=('Segoe UI', 9))
        self._status_lbl.pack(side='left', padx=10)

        # Buttons right-aligned
        btn_frame = tk.Frame(hdr, bg=SURFACE)
        btn_frame.pack(side='right', padx=16)

        self._run_btn      = self._btn(btn_frame, 'Run Now',  self._run_bot,      ACCENT)
        self._backtest_btn = self._btn(btn_frame, 'Backtest', self._run_backtest, ACCENT2)
        self._run_btn.pack(side='right', padx=4)
        self._backtest_btn.pack(side='right', padx=4)
        self._btn(btn_frame, 'Refresh', self._refresh, '#374151').pack(side='right', padx=4)

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         bg=color, fg='white', relief='flat',
                         font=('Segoe UI', 9, 'bold'),
                         padx=14, pady=6, cursor='hand2',
                         activebackground=color, activeforeground='white')

    # ── tabs ──────────────────────────────────────────────────────────────

    def _build_tabs(self):
        self._nb = ttk.Notebook(self)
        self._nb.pack(fill='both', expand=True, padx=0, pady=0)

        self._tab_dashboard = ttk.Frame(self._nb)
        self._tab_signals   = ttk.Frame(self._nb)
        self._tab_trades    = ttk.Frame(self._nb)
        self._tab_settings  = ttk.Frame(self._nb)
        self._tab_backtest  = ttk.Frame(self._nb)

        self._nb.add(self._tab_dashboard, text='  Dashboard  ')
        self._nb.add(self._tab_signals,   text='  Signals  ')
        self._nb.add(self._tab_trades,    text='  Trade Log  ')
        self._nb.add(self._tab_settings,  text='  Settings  ')
        self._nb.add(self._tab_backtest,  text='  Backtest  ')

        self._build_dashboard()
        self._build_signals()
        self._build_trades()
        self._build_settings()
        self._build_backtest()

    # ══════════════════════════════════════════════════════════════════════
    # TAB: Dashboard
    # ══════════════════════════════════════════════════════════════════════

    def _build_dashboard(self):
        f = self._tab_dashboard
        f.configure(style='TFrame')

        # ── stat cards row ──
        cards_row = tk.Frame(f, bg=BG)
        cards_row.pack(fill='x', padx=16, pady=(16, 8))

        self._card_last_run   = self._stat_card(cards_row, 'Last Run',        '—')
        self._card_signals    = self._stat_card(cards_row, 'Active Signals',  '—')
        self._card_buys       = self._stat_card(cards_row, 'BUY Signals',     '—', GREEN)
        self._card_shorts     = self._stat_card(cards_row, 'SHORT Signals',   '—', RED)

        for c in (self._card_last_run, self._card_signals,
                  self._card_buys, self._card_shorts):
            c.pack(side='left', fill='both', expand=True, padx=6)

        # ── recent activity ──
        act_frame = tk.Frame(f, bg=SURFACE, bd=0)
        act_frame.pack(fill='both', expand=True, padx=16, pady=8)

        tk.Label(act_frame, text='Recent Activity', bg=SURFACE, fg=FG,
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=14, pady=(10, 4))

        self._activity_text = tk.Text(act_frame, bg=SURFACE, fg=FG,
                                      font=('Consolas', 9), relief='flat',
                                      state='disabled', wrap='word',
                                      highlightthickness=0)
        sb = ttk.Scrollbar(act_frame, command=self._activity_text.yview)
        self._activity_text.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self._activity_text.pack(fill='both', expand=True, padx=8, pady=(0, 10))

    def _stat_card(self, parent, label, value, color=FG):
        frame = tk.Frame(parent, bg=SURFACE, padx=16, pady=12)
        tk.Label(frame, text=label, bg=SURFACE, fg=FG_DIM,
                 font=('Segoe UI', 9)).pack(anchor='w')
        val_lbl = tk.Label(frame, text=value, bg=SURFACE, fg=color,
                           font=('Segoe UI', 20, 'bold'))
        val_lbl.pack(anchor='w')
        frame._val_lbl = val_lbl
        frame._color   = color
        return frame

    def _update_card(self, card, value, color=None):
        card._val_lbl.configure(text=value, fg=color or card._color)

    def _refresh_dashboard(self, logs):
        if not logs:
            return
        last = logs[-1]
        ts   = last.get('timestamp', '')
        try:
            dt = datetime.fromisoformat(ts).strftime('%d %b %Y  %H:%M')
        except Exception:
            dt = ts

        sigs  = last.get('signals', {})
        buys  = sum(1 for s in sigs.values() if s == 'BUY')
        shorts = sum(1 for s in sigs.values() if s == 'SHORT')

        self._update_card(self._card_last_run,  dt)
        self._update_card(self._card_signals,   str(len(sigs)))
        self._update_card(self._card_buys,      str(buys),  GREEN if buys  else FG)
        self._update_card(self._card_shorts,    str(shorts), RED  if shorts else FG)

        # Activity feed
        lines = []
        for entry in reversed(logs[-20:]):
            ts_str = entry.get('timestamp', '')[:16].replace('T', '  ')
            for trade in entry.get('trades', []):
                action = trade.get('action', '')
                ticker = trade.get('ticker', '')
                signal = trade.get('signal', '')
                if 'BUY' in action or 'SHORT' in action or 'EXIT' in action:
                    lines.append(f"[{ts_str}]  {ticker:<6} {signal:<6}  {action}")

        self._activity_text.configure(state='normal')
        self._activity_text.delete('1.0', 'end')
        self._activity_text.insert('end', '\n'.join(lines) if lines else 'No trades yet.')
        self._activity_text.configure(state='disabled')

    # ══════════════════════════════════════════════════════════════════════
    # TAB: Signals
    # ══════════════════════════════════════════════════════════════════════

    def _build_signals(self):
        f = self._tab_signals
        cols = ('Ticker', 'Signal', 'Confidence', 'Timestamp')

        self._sig_tree = self._tree(f, cols)
        self._sig_tree.column('Ticker',     width=100, anchor='center')
        self._sig_tree.column('Signal',     width=100, anchor='center')
        self._sig_tree.column('Confidence', width=120, anchor='center')
        self._sig_tree.column('Timestamp',  width=200, anchor='center')

    def _refresh_signals(self, logs):
        tree = self._sig_tree
        tree.delete(*tree.get_children())
        if not logs:
            return
        last = logs[-1]
        ts   = last.get('timestamp', '')[:16]
        sigs  = last.get('signals', {})
        confs = last.get('confidences', {})

        for ticker in sorted(sigs):
            sig  = sigs[ticker]
            conf = confs.get(ticker, 0)
            tag  = sig.lower()
            tree.insert('', 'end', values=(ticker, sig, f'{conf:.1%}', ts), tags=(tag,))

        tree.tag_configure('buy',   foreground=GREEN)
        tree.tag_configure('short', foreground=RED)
        tree.tag_configure('hold',  foreground=FG_DIM)

    # ══════════════════════════════════════════════════════════════════════
    # TAB: Trade Log
    # ══════════════════════════════════════════════════════════════════════

    def _build_trades(self):
        f = self._tab_trades
        cols = ('Time', 'Ticker', 'Signal', 'Action', 'Confidence')

        self._trade_tree = self._tree(f, cols)
        self._trade_tree.column('Time',       width=155, anchor='center')
        self._trade_tree.column('Ticker',     width=80,  anchor='center')
        self._trade_tree.column('Signal',     width=80,  anchor='center')
        self._trade_tree.column('Action',     width=460, anchor='w')
        self._trade_tree.column('Confidence', width=100, anchor='center')

    def _refresh_trades(self, logs):
        tree = self._trade_tree
        tree.delete(*tree.get_children())
        rows = []
        for entry in logs:
            ts = entry.get('timestamp', '')[:16].replace('T', ' ')
            for trade in entry.get('trades', []):
                rows.append((
                    ts,
                    trade.get('ticker', ''),
                    trade.get('signal', ''),
                    trade.get('action', ''),
                    f"{trade.get('confidence', 0):.1%}",
                ))
        for row in reversed(rows[-500:]):
            action = row[3]
            tag = 'buy' if 'BUY' in action else ('sell' if 'EXIT' in action or 'SHORT' in action else 'hold')
            tree.insert('', 'end', values=row, tags=(tag,))

        tree.tag_configure('buy',  foreground=GREEN)
        tree.tag_configure('sell', foreground=RED)
        tree.tag_configure('hold', foreground=FG_DIM)

    # ══════════════════════════════════════════════════════════════════════
    # TAB: Settings
    # ══════════════════════════════════════════════════════════════════════

    def _build_settings(self):
        f = self._tab_settings

        canvas = tk.Canvas(f, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(f, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        inner = tk.Frame(canvas, bg=BG)
        canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<MouseWheel>', lambda e: canvas.yview_scroll(-1*(e.delta//120), 'units'))

        self._setting_vars = {}

        sections = [
            ('Strategy', [
                ('INTERVAL',          'Bar interval',           'str',   '1d, 1h, 30m'),
                ('BUY_THRESHOLD',     'Buy threshold',          'float', 'P(UP) >= this to BUY'),
                ('SHORT_THRESHOLD',   'Short threshold',        'float', 'P(UP) <= this to SHORT'),
                ('STOP_LOSS_PCT',     'Stop loss %',            'float', 'e.g. -0.015 = -1.5%'),
                ('TAKE_PROFIT_PCT',   'Take profit %',          'float', 'e.g. 0.07 = 7%'),
                ('KELLY_FRACTION',    'Kelly fraction',         'float', '0.3 = 30%-Kelly'),
                ('MAX_POSITION_PCT',  'Max position %',         'float', 'e.g. 0.15 = 15% of portfolio'),
            ]),
            ('Tickers', [
                ('TICKERS',           'Tickers (comma-separated)', 'str', 'e.g. AAPL, NVDA, MSFT'),
            ]),
            ('Training & Backtest', [
                ('RETRAIN_INTERVAL_DAYS',    'Retrain every N days',   'int',   ''),
                ('BACKTEST_INITIAL_CAPITAL', 'Backtest capital ($)',    'str',   ''),
            ]),
            ('Trading 212', [
                ('T212_API_KEY',   'API key',         'str',  ''),
                ('T212_DEMO',      'Demo mode',       'bool', 'True or False'),
                ('T212_MODE',      'Mode',            'str',  'PIE or DIRECT'),
                ('T212_PIE_NAME',  'Pie name',        'str',  ''),
            ]),
        ]

        cfg = read_config()

        for section_title, fields in sections:
            # Section header
            sec = tk.Frame(inner, bg=SURFACE, pady=10)
            sec.pack(fill='x', padx=16, pady=(12, 2))
            tk.Label(sec, text=section_title, bg=SURFACE, fg=FG,
                     font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=12)

            grid = tk.Frame(inner, bg=BG)
            grid.pack(fill='x', padx=16, pady=2)
            grid.columnconfigure(1, weight=1)

            for i, (key, label, dtype, hint) in enumerate(fields):
                tk.Label(grid, text=label, bg=BG, fg=FG,
                         font=('Segoe UI', 9), width=26, anchor='w').grid(
                    row=i, column=0, padx=(8, 4), pady=5, sticky='w')

                var = tk.StringVar(value=cfg.get(key, ''))
                self._setting_vars[key] = var

                entry = tk.Entry(grid, textvariable=var, bg=SURFACE, fg=FG,
                                 insertbackground=FG, relief='flat',
                                 font=('Segoe UI', 9), bd=6)
                entry.grid(row=i, column=1, padx=4, pady=5, sticky='ew')

                if hint:
                    tk.Label(grid, text=hint, bg=BG, fg=FG_DIM,
                             font=('Segoe UI', 8)).grid(
                        row=i, column=2, padx=(4, 8), sticky='w')

        # Save button
        btn_row = tk.Frame(inner, bg=BG)
        btn_row.pack(fill='x', padx=16, pady=16)
        self._btn(btn_row, 'Save Settings', self._save_settings, GREEN).pack(side='left', padx=8)
        self._save_status = tk.Label(btn_row, text='', bg=BG, fg=GREEN,
                                     font=('Segoe UI', 9))
        self._save_status.pack(side='left')

    def _save_settings(self):
        updates = {k: v.get() for k, v in self._setting_vars.items() if v.get()}
        try:
            write_config(updates)
            self._save_status.configure(text='Saved successfully.', fg=GREEN)
            self.after(3000, lambda: self._save_status.configure(text=''))
        except Exception as e:
            messagebox.showerror('Save Error', str(e))

    # ══════════════════════════════════════════════════════════════════════
    # TAB: Backtest
    # ══════════════════════════════════════════════════════════════════════

    def _build_backtest(self):
        f = self._tab_backtest

        # ── Navigation bar ──
        nav = tk.Frame(f, bg=SURFACE, pady=8)
        nav.pack(fill='x')

        self._prev_btn = self._btn(nav, '< Prev', self._chart_prev, '#374151')
        self._prev_btn.pack(side='left', padx=12)

        self._next_btn = self._btn(nav, 'Next >', self._chart_next, '#374151')
        self._next_btn.pack(side='right', padx=12)

        self._chart_title_lbl = tk.Label(nav, text='', bg=SURFACE, fg=FG,
                                         font=('Segoe UI', 12, 'bold'))
        self._chart_title_lbl.pack(side='left', expand=True)

        self._chart_num_lbl = tk.Label(nav, text='', bg=SURFACE, fg=FG_DIM,
                                       font=('Segoe UI', 9))
        self._chart_num_lbl.pack(side='left', padx=12)

        # ── Main area: chart (left) + metrics (right) ──
        main = tk.Frame(f, bg=BG)
        main.pack(fill='both', expand=True, padx=6, pady=4)

        # Chart image area
        chart_frame = tk.Frame(main, bg=BG)
        chart_frame.pack(side='left', fill='both', expand=True)

        self._backtest_lbl = tk.Label(chart_frame, bg=BG,
                                      text='No backtest results found.\nRun a backtest first.',
                                      fg=FG_DIM, font=('Segoe UI', 11))
        self._backtest_lbl.pack(expand=True, fill='both')

        # Metrics panel
        metrics_frame = tk.Frame(main, bg=SURFACE, padx=14, pady=12)
        metrics_frame.pack(side='right', fill='y', padx=(6, 2))

        tk.Label(metrics_frame, text='Performance', bg=SURFACE, fg=FG,
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w', pady=(0, 8))

        self._metrics_text = tk.Text(metrics_frame, bg=SURFACE, fg=FG,
                                     font=('Consolas', 9), relief='flat',
                                     state='disabled', wrap='none',
                                     highlightthickness=0, width=28)
        self._metrics_text.pack(fill='both', expand=True)

        self._load_backtest_image()

    def _chart_prev(self):
        self._chart_index = max(0, self._chart_index - 1)
        self._load_backtest_image()

    def _chart_next(self):
        available = [p for p, _ in BACKTEST_CHART_FILES if os.path.exists(p)]
        self._chart_index = min(len(available) - 1, self._chart_index + 1)
        self._load_backtest_image()

    def _load_backtest_image(self):
        available = [(p, t) for p, t in BACKTEST_CHART_FILES if os.path.exists(p)]

        if not available:
            # Fall back to combined image
            if os.path.exists(BACKTEST_IMG):
                available = [(BACKTEST_IMG, 'Backtest Dashboard')]
            else:
                self._backtest_lbl.configure(image='',
                    text='No backtest results found.\nRun a backtest first.')
                self._chart_title_lbl.configure(text='')
                self._chart_num_lbl.configure(text='')
                return

        self._chart_index = max(0, min(self._chart_index, len(available) - 1))
        path, title = available[self._chart_index]
        n = len(available)

        self._chart_title_lbl.configure(text=title)
        self._chart_num_lbl.configure(text=f'{self._chart_index + 1} / {n}')

        # Disable nav buttons at boundaries
        self._prev_btn.configure(state='normal' if self._chart_index > 0     else 'disabled')
        self._next_btn.configure(state='normal' if self._chart_index < n - 1 else 'disabled')

        try:
            img = Image.open(path)
            w = max(self.winfo_width() - 280, 700)
            h = max(self.winfo_height() - 160, 420)
            ratio = min(w / img.width, h / img.height)
            if ratio < 1.0:
                img = img.resize((int(img.width * ratio), int(img.height * ratio)),
                                 Image.LANCZOS)
            self._backtest_img = ImageTk.PhotoImage(img)
            self._backtest_lbl.configure(image=self._backtest_img, text='')
        except Exception as e:
            self._backtest_lbl.configure(image='', text=f'Could not load image: {e}')

        self._load_metrics()

    def _load_metrics(self):
        if not os.path.exists(BACKTEST_METRICS):
            return
        try:
            with open(BACKTEST_METRICS) as f:
                m = json.load(f)

            def pct(v):  return f'{v:+.1%}' if v is not None else 'N/A'
            def flt(v):  return f'{v:.2f}'  if v is not None else 'N/A'
            def dol(v):  return f'${v:>,.0f}' if v is not None else 'N/A'

            lines = [
                'Period',
                f"  {m.get('period_start', '?')}",
                f"  -> {m.get('period_end', '?')}",
                '',
                'Returns',
                f"  Total   {pct(m.get('total_return'))}",
                f"  CAGR    {pct(m.get('cagr'))}",
                f"  Initial {dol(m.get('initial'))}",
                f"  Final   {dol(m.get('final'))}",
                '',
                'Risk',
                f"  Sharpe  {flt(m.get('sharpe'))}",
                f"  Sortino {flt(m.get('sortino'))}",
                f"  Max DD  {pct(m.get('max_drawdown'))}",
                '',
                'Trades',
                f"  Total   {m.get('n_trades', 0)}",
                f"  Win%    {pct(m.get('win_rate'))}",
                f"  Avg Win {pct(m.get('avg_win'))}",
                f"  Avg Los {pct(m.get('avg_loss'))}",
                f"  PF      {flt(m.get('profit_factor'))}",
                '',
                'By Side',
                f"  Longs  {m.get('n_longs', 0)}  WR {pct(m.get('long_win_rate'))}",
                f"  Shorts {m.get('n_shorts', 0)}  WR {pct(m.get('short_win_rate'))}",
            ]

            self._metrics_text.configure(state='normal')
            self._metrics_text.delete('1.0', 'end')
            self._metrics_text.insert('end', '\n'.join(lines))
            self._metrics_text.configure(state='disabled')
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════
    # SHARED
    # ══════════════════════════════════════════════════════════════════════

    def _tree(self, parent, cols):
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill='both', expand=True, padx=8, pady=8)

        tree = ttk.Treeview(frame, columns=cols, show='headings', selectmode='browse')
        for col in cols:
            tree.heading(col, text=col)
        sb_y = ttk.Scrollbar(frame, orient='vertical',   command=tree.yview)
        sb_x = ttk.Scrollbar(frame, orient='horizontal', command=tree.xview)
        tree.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        sb_y.pack(side='right',  fill='y')
        sb_x.pack(side='bottom', fill='x')
        tree.pack(fill='both', expand=True)
        return tree

    # ══════════════════════════════════════════════════════════════════════
    # REFRESH / RUN
    # ══════════════════════════════════════════════════════════════════════

    def _refresh(self):
        logs = load_trade_log()
        self._refresh_dashboard(logs)
        self._refresh_signals(logs)
        self._refresh_trades(logs)
        self._load_backtest_image()

        ts = datetime.now().strftime('%H:%M:%S')
        self._status_lbl.configure(text=f'Last refreshed: {ts}  |  {len(logs)} log entries')

        # Auto-refresh every 60 s
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(60_000, self._refresh)

    def _run_bot(self):
        if self._bot_process and self._bot_process.poll() is None:
            messagebox.showinfo('Already running', 'The bot is already running.')
            return
        self._run_btn.configure(text='Running...', state='disabled')
        threading.Thread(target=self._subprocess_bot, daemon=True).start()

    def _run_backtest(self):
        if self._backtest_process and self._backtest_process.poll() is None:
            messagebox.showinfo('Already running', 'Backtest is already running.')
            return
        self._backtest_btn.configure(text='Running...', state='disabled')
        threading.Thread(target=self._subprocess_backtest, daemon=True).start()

    def _subprocess_bot(self):
        python = sys.executable
        script = os.path.join(ROOT, 'src', 'run_bot.py')
        self._bot_process = subprocess.Popen(
            [python, script], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._bot_process.wait()
        self.after(0, self._on_bot_done)

    def _subprocess_backtest(self):
        python = sys.executable
        script = os.path.join(ROOT, 'src', 'backtest.py')
        self._backtest_process = subprocess.Popen(
            [python, script], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self._backtest_process.wait()
        self.after(0, self._on_backtest_done)

    def _on_bot_done(self):
        self._run_btn.configure(text='Run Now', state='normal')
        self._refresh()

    def _on_backtest_done(self):
        self._backtest_btn.configure(text='Backtest', state='normal')
        self._chart_index = 0
        self._load_backtest_image()
        self._nb.select(self._tab_backtest)

    def _on_close(self):
        if self._after_id:
            self.after_cancel(self._after_id)
        self.destroy()


# ==================== ENTRY POINT ====================

if __name__ == '__main__':
    app = BotGUI()
    app.mainloop()
