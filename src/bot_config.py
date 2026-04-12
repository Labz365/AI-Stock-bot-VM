"""
CENTRAL BOT CONFIGURATION
All strategy parameters live here. Imported by every other module.
Run `python src/retrain.py` after changing TARGET_DAYS or TARGET_GAIN_PCT.
"""

# ===== BAR SIZE =====
# '1d'  → daily bars,   run once per day
# '1h'  → hourly bars,  run every 30-60 min during market hours (9:30–16:00 ET)
# '30m' → 30-min bars,  run every 15-30 min  (yfinance: max 60 days history)
INTERVAL     = '1h'
BARS_PER_DAY = 7      # approximate trading bars in one session (used for target scaling)

# ===== TICKERS =====
TICKERS = [
    'AAPL', 'GOOGL', 'AMZN', 'NVDA',   # original core
    'MSFT', 'META', 'TSLA', 'AMD',       # high-momentum tech
    'NFLX', 'CRM',                        # growth
]

# ===== SIGNAL THRESHOLDS =====
BUY_THRESHOLD   = 0.65   # model P(UP) >= this → BUY long
SHORT_THRESHOLD = 0.35   # model P(UP) <= this → SHORT (inverse signal)

# ===== EXIT PARAMETERS =====
STOP_LOSS_PCT   = -0.015  # -1.5% hard stop
TAKE_PROFIT_PCT =  0.07   # +7% take profit

# ===== KELLY CRITERION POSITION SIZING =====
# f* = (p·b − q) / b,  actual fraction = f* × KELLY_FRACTION
KELLY_FRACTION   = 0.30   # conservative 30%-Kelly
MAX_POSITION_PCT = 0.15   # hard cap: max 15% of portfolio per position

# Win/loss ratio used in Kelly formula (take-profit ÷ stop-loss distance)
WIN_LOSS_RATIO = abs(TAKE_PROFIT_PCT / STOP_LOSS_PCT)   # ≈ 4.67

# ===== TRAINING TARGET =====
# Predict: will Close N bars from now exceed Close × (1 + GAIN_PCT)?
# For '1d': 5 bars = 5 trading days, 2% target
# For '1h': 4 bars = 4 hours (~half a session), 0.8% target
TARGET_DAYS     = 4 if INTERVAL != '1d' else 5
TARGET_GAIN_PCT = 0.008 if INTERVAL != '1d' else 0.02

# ===== AUTO-RETRAIN =====
RETRAIN_INTERVAL_DAYS = 30

# ===== BACKTEST =====
BACKTEST_INITIAL_CAPITAL = 100_000
BACKTEST_TRAIN_SPLIT     = 0.80    # models trained on first 80%, backtest on last 20%

# ===== TRADING 212 =====
T212_API_KEY   = '47031985ZDbRWZmTBYCTaVLRIUcFQEwbIDMzN'   # Settings → API → Generate key
T212_DEMO      = False                       # True = demo account, False = live (real money!)

# Execution mode:
#   'PIE'    → create/update a T212 Pie weighted by Kelly confidence (recommended for testing)
#   'DIRECT' → place individual market orders per signal
T212_MODE      = 'PIE'

T212_PIE_NAME          = 'AI Bot'           # name of the pie in your T212 account
T212_PIE_INVEST_AMOUNT = 0                  # extra £/$ to invest into pie each run (0 = rebalance only)