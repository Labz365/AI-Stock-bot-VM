"""
SHARED TECHNICAL FEATURE COMPUTATION
=====================================
Single source of truth for all OHLCV-based features.
Used by both build_features.py (training) and generate_signals.py (live).

Input df must have columns: Open, High, Low, Close, Volume, Date
All raw-price-scale intermediates are dropped before returning.
"""

import pandas as pd
import numpy as np


# Columns that are raw-price or raw-volume scale — excluded from model input
# (normalized versions of these ARE kept as features)
RAW_SCALE_COLS = ['ma_10', 'ma_50', 'volume_ma_10', 'volume_ma_50', 'atr_14']

# Base columns never used as model features
BASE_COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# All columns to exclude when building the feature matrix for the model
DROP_FROM_MODEL = BASE_COLS + RAW_SCALE_COLS + ['target']


def strip_tz(series: pd.Series) -> pd.Series:
    """Return a tz-naive datetime Series (strips UTC or any other timezone)."""
    s = pd.to_datetime(series)
    if s.dt.tz is not None:
        s = s.dt.tz_convert('UTC').dt.tz_localize(None)
    return s


def _rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-10)))


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical features to a OHLCV + Date DataFrame.
    Returns the same DataFrame with ~60 new feature columns.
    Intermediate raw-scale columns (ha_open, etc.) are dropped.
    """

    # ===== HEIKIN-ASHI =====
    df['ha_close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['ha_open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    df.loc[df.index[0], 'ha_open'] = df['Open'].iloc[0]
    df['ha_high'] = df[['High', 'ha_open', 'ha_close']].max(axis=1)
    df['ha_low'] = df[['Low', 'ha_open', 'ha_close']].min(axis=1)
    df['ha_return'] = np.log(df['ha_close'] / df['ha_close'].shift(1))
    df['ha_body'] = (df['ha_close'] - df['ha_open']) / df['ha_close']
    df['ha_upper_shadow'] = (
        df['ha_high'] - df[['ha_open', 'ha_close']].max(axis=1)) / df['ha_close']
    df['ha_lower_shadow'] = (
        df[['ha_open', 'ha_close']].min(axis=1) - df['ha_low']) / df['ha_close']

    # ===== LOG RETURNS =====
    df['return_cc'] = np.log(df['Close'] / df['Close'].shift(1))
    df['return_oc'] = np.log(df['Close'] / df['Open'])
    df['overnight_gap'] = np.log(df['Open'] / df['Close'].shift(1))
    df['upside'] = np.log(df['High'] / df['Close'].shift(1))
    df['downside'] = np.log(df['Low'] / df['Close'].shift(1))

    # Lagged returns (autoregressive signals)
    df['lag_1_return'] = df['return_cc'].shift(1)
    df['lag_2_return'] = df['return_cc'].shift(2)
    df['lag_3_return'] = df['return_cc'].shift(3)
    df['lag_5_return'] = df['return_cc'].shift(5)

    # ===== MULTI-TIMEFRAME RETURNS =====
    df['weekly_return'] = np.log(df['Close'] / df['Close'].shift(5))
    df['biweekly_return'] = np.log(df['Close'] / df['Close'].shift(10))
    df['monthly_return'] = np.log(df['Close'] / df['Close'].shift(21))

    # Momentum (price ratio — different signal than log return)
    df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['momentum_21'] = df['Close'] / df['Close'].shift(21) - 1
    df['momentum_63'] = df['Close'] / df['Close'].shift(63) - 1  # 3 months

    # ===== MOVING AVERAGES (smoothed, price-normalized) =====
    smooth_close = df['Close'].ewm(span=5).mean()
    df['ma_10'] = smooth_close.rolling(10).mean()   # kept raw for drop_cols
    df['ma_50'] = smooth_close.rolling(50).mean()   # kept raw for drop_cols
    df['price_vs_ma10'] = (smooth_close - df['ma_10']) / (df['ma_10'] + 1e-10)
    df['price_vs_ma50'] = (smooth_close - df['ma_50']) / (df['ma_50'] + 1e-10)
    df['ma_diff'] = (df['ma_10'] - df['ma_50']) / (df['ma_50'] + 1e-10)
    df['ma_10_slope'] = df['ma_10'].pct_change(5)
    df['ma_50_slope'] = df['ma_50'].pct_change(5)

    # EMA crossovers
    ema5 = df['Close'].ewm(span=5).mean()
    ema20 = df['Close'].ewm(span=20).mean()
    ema50 = df['Close'].ewm(span=50).mean()
    df['ema5_vs_ema20'] = (ema5 - ema20) / (ema20 + 1e-10)
    df['ema20_vs_ema50'] = (ema20 - ema50) / (ema50 + 1e-10)
    df['price_vs_ema5'] = (df['Close'] - ema5) / (ema5 + 1e-10)
    df['price_vs_ema20'] = (df['Close'] - ema20) / (ema20 + 1e-10)

    # ===== BOLLINGER BANDS =====
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df['bb_width'] = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    # BB squeeze: current width vs 90-day average (detects consolidation breakouts)
    df['bb_squeeze'] = df['bb_width'] / (df['bb_width'].rolling(90).mean() + 1e-10)

    # ===== TRUE RANGE & ATR =====
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()               # raw scale — in drop_cols
    df['atr_ratio'] = df['atr_14'] / (df['Close'] + 1e-10)   # normalized ATR
    df['atr_trend'] = df['atr_14'] / (df['atr_14'].rolling(30).mean() + 1e-10)  # rising/falling

    # ===== VOLATILITY =====
    df['volatility_10'] = df['return_cc'].rolling(10).std()
    df['volatility_50'] = df['return_cc'].rolling(50).std()
    df['vol_ratio'] = df['volatility_10'] / (df['volatility_50'] + 1e-10)
    df['volatility_annual'] = df['volatility_10'] * (252 ** 0.5)
    df['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * (np.log(df['High'] / df['Low']) ** 2)
    ).rolling(10).mean()
    df['vol_change'] = df['volatility_10'].pct_change(5)

    # ===== HIGH-LOW =====
    df['intraday_range'] = (df['High'] - df['Low']) / df['Close']
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
    df['range_expansion'] = df['intraday_range'] / (df['intraday_range'].rolling(20).mean() + 1e-10)

    # 52-week (252 trading days) position
    high_252 = df['High'].rolling(252).max()
    low_252 = df['Low'].rolling(252).min()
    df['week52_high_dist'] = (high_252 - df['Close']) / (high_252 + 1e-10)
    df['week52_low_dist'] = (df['Close'] - low_252) / (low_252 + 1e-10)
    df['week52_position'] = (df['Close'] - low_252) / (high_252 - low_252 + 1e-10)

    # ===== RSI (3 periods for different horizons) =====
    df['rsi_7'] = _rsi(df['Close'], 7)
    df['rsi'] = _rsi(df['Close'], 14)
    df['rsi_21'] = _rsi(df['Close'], 21)
    df['rsi_slope'] = df['rsi'].diff(3)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)

    # ===== STOCHASTIC OSCILLATOR =====
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    stoch_k = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_k.rolling(3).mean()
    df['stoch_signal'] = stoch_k - df['stoch_d']   # K-D divergence
    df['stoch_overbought'] = (stoch_k > 80).astype(int)
    df['stoch_oversold'] = (stoch_k < 20).astype(int)

    # ===== WILLIAMS %R =====
    high_14w = df['High'].rolling(14).max()
    low_14w = df['Low'].rolling(14).min()
    df['williams_r'] = -100 * (high_14w - df['Close']) / (high_14w - low_14w + 1e-10)

    # ===== MACD (normalized by price) =====
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    macd_sig = macd.ewm(span=9).mean()
    macd_hist = macd - macd_sig
    df['macd_norm'] = macd / (df['Close'] + 1e-10)
    df['macd_signal_norm'] = macd_sig / (df['Close'] + 1e-10)
    df['macd_hist_norm'] = macd_hist / (df['Close'] + 1e-10)
    df['macd_cross'] = (macd > macd_sig).astype(int)
    df['macd_hist_change'] = macd_hist.diff()

    # ===== ADX / DMI =====
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(14).mean()
    df['adx_di_diff'] = plus_di - minus_di   # positive = uptrend
    df['adx_trend'] = (df['adx'] > 25).astype(int)

    # ===== OBV (On-Balance Volume) =====
    obv = (np.sign(df['return_cc']) * df['Volume']).fillna(0).cumsum()
    obv_ma10 = obv.rolling(10).mean()
    df['obv_trend'] = (obv - obv_ma10) / (obv_ma10.abs() + 1e-10)
    df['obv_slope'] = obv.diff(5) / (obv.abs().rolling(5).mean() + 1e-10)

    # ===== MFI (Money Flow Index) =====
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    mf = typical_price * df['Volume']
    tp_up = typical_price > typical_price.shift(1)
    pos_mf = mf.where(tp_up, 0).rolling(14).sum()
    neg_mf = mf.where(~tp_up, 0).rolling(14).sum()
    df['mfi'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
    df['mfi_overbought'] = (df['mfi'] > 80).astype(int)
    df['mfi_oversold'] = (df['mfi'] < 20).astype(int)

    # ===== CMF (Chaikin Money Flow) =====
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    df['cmf'] = (clv * df['Volume']).rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-10)

    # ===== VOLUME =====
    df['volume_ma_10'] = df['Volume'].rolling(10).mean()   # raw — in drop_cols
    df['volume_ma_50'] = df['Volume'].rolling(50).mean()   # raw — in drop_cols
    df['volume_ratio'] = df['Volume'] / (df['volume_ma_10'] + 1e-10)
    df['volume_trend'] = df['volume_ma_10'] / (df['volume_ma_50'] + 1e-10)
    df['price_volume'] = df['return_cc'] * df['volume_ratio']
    # Volume-Price Trend direction signal
    vpt = (df['return_cc'] * df['Volume']).cumsum()
    df['vpt_signal'] = (vpt > vpt.rolling(10).mean()).astype(int)
    df['vpt_slope'] = vpt.diff(5) / (vpt.abs().rolling(5).mean() + 1e-10)

    # ===== CONSECUTIVE STREAK =====
    up = (df['return_cc'] > 0).astype(int)
    groups = (up != up.shift()).cumsum()
    counts = up.groupby(groups).cumcount() + 1
    df['up_streak'] = counts.where(up == 1, 0)
    df['down_streak'] = counts.where(up == 0, 0)

    # ===== DAY OF WEEK / TIME OF DAY =====
    dt = pd.to_datetime(df['Date'])
    df['day_of_week'] = dt.dt.dayofweek
    df['is_monday']   = (df['day_of_week'] == 0).astype(int)
    df['is_friday']   = (df['day_of_week'] == 4).astype(int)
    # Hour of day (0 for daily bars; 9–15 for hourly bars)
    df['hour_of_day']   = dt.dt.hour
    df['is_open_hour']  = (dt.dt.hour == 9).astype(int)   # 9–10 AM: volatile open
    df['is_close_hour'] = (dt.dt.hour == 15).astype(int)  # 3–4 PM: high-volume close

    # ===== DROP RAW INTERMEDIATES =====
    _drop = ['ha_open', 'ha_high', 'ha_low', 'ha_close']
    df.drop(columns=[c for c in _drop if c in df.columns], inplace=True)

    return df
