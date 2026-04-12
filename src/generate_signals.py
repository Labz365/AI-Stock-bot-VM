"""
LIVE SIGNAL GENERATION
Downloads fresh data, builds features (matching training exactly), runs models.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
import json

from features import compute_technical_features, DROP_FROM_MODEL, strip_tz
from bot_config import TICKERS as tickers, BUY_THRESHOLD, SHORT_THRESHOLD, INTERVAL

# Enough history for all rolling windows regardless of bar size
LIVE_HISTORY_DAYS = '400d'


# ==================== LOAD MODEL ====================

def load_model(ticker):
    """Load model pkl. Supports both new {model, feature_names} dict and old direct-model format."""
    obj = joblib.load(f'models/{ticker}.pkl')
    if isinstance(obj, dict):
        return obj['model'], obj['feature_names']
    # Old format: raw model object
    feature_names = list(obj.feature_names_in_) if hasattr(obj, 'feature_names_in_') else None
    return obj, feature_names


# ==================== MARKET CONTEXT (LIVE) ====================

def download_market_context_live():
    context = pd.DataFrame()

    downloads = {
        '^VIX': 'vix', 'SPY': 'spy', 'QQQ': 'qqq',
        'XLK': 'xlk', 'TLT': 'tlt', 'GLD': 'gld', 'UUP': 'uup',
    }
    raw = {}
    for sym, name in downloads.items():
        try:
            df = yf.download(sym, period=LIVE_HISTORY_DAYS,
                             interval=INTERVAL, progress=False)
            df.columns = df.columns.get_level_values(0)
            df = df.reset_index().rename(columns={df.reset_index().columns[0]: 'Date'})
            df['Date'] = strip_tz(df['Date'])
            raw[name] = df
        except Exception:
            pass

    if 'vix' in raw:
        v = raw['vix'].copy()
        v['vix'] = v['Close']
        v['vix_change'] = v['vix'].pct_change()
        v['vix_ma5'] = v['vix'].rolling(5).mean()
        v['vix_high'] = (v['vix'] > 25).astype(int)
        v['vix_extreme'] = (v['vix'] > 35).astype(int)
        context = v[['Date', 'vix', 'vix_change', 'vix_ma5', 'vix_high', 'vix_extreme']]

    if 'spy' in raw:
        s = raw['spy'].copy()
        s['sp500_return'] = np.log(s['Close'] / s['Close'].shift(1))
        s['sp500_trend'] = (s['Close'].rolling(10).mean() > s['Close'].rolling(50).mean()).astype(int)
        s['sp500_weekly'] = np.log(s['Close'] / s['Close'].shift(5))
        s = s[['Date', 'sp500_return', 'sp500_trend', 'sp500_weekly']]
        context = context.merge(s, on='Date', how='outer') if not context.empty else s

    if 'qqq' in raw:
        q = raw['qqq'].copy()
        q['nasdaq_return'] = np.log(q['Close'] / q['Close'].shift(1))
        q['nasdaq_weekly'] = np.log(q['Close'] / q['Close'].shift(5))
        context = context.merge(q[['Date', 'nasdaq_return', 'nasdaq_weekly']], on='Date', how='outer')

    if 'xlk' in raw and 'spy' in raw:
        m = raw['xlk'][['Date', 'Close']].merge(
            raw['spy'][['Date', 'Close']], on='Date', suffixes=('_xlk', '_spy'))
        m['tech_vs_market'] = np.log(m['Close_xlk'] / m['Close_spy'])
        m['tech_vs_market_change'] = m['tech_vs_market'].diff()
        context = context.merge(m[['Date', 'tech_vs_market', 'tech_vs_market_change']], on='Date', how='outer')

    if 'tlt' in raw:
        t = raw['tlt'].copy()
        t['bond_return'] = np.log(t['Close'] / t['Close'].shift(1))
        t['bond_weekly'] = np.log(t['Close'] / t['Close'].shift(5))
        context = context.merge(t[['Date', 'bond_return', 'bond_weekly']], on='Date', how='outer')

    if 'gld' in raw:
        g = raw['gld'].copy()
        g['gold_return'] = np.log(g['Close'] / g['Close'].shift(1))
        g['gold_weekly'] = np.log(g['Close'] / g['Close'].shift(5))
        context = context.merge(g[['Date', 'gold_return', 'gold_weekly']], on='Date', how='outer')

    if 'uup' in raw:
        u = raw['uup'].copy()
        u['dollar_return'] = np.log(u['Close'] / u['Close'].shift(1))
        u['dollar_weekly'] = np.log(u['Close'] / u['Close'].shift(5))
        context = context.merge(u[['Date', 'dollar_return', 'dollar_weekly']], on='Date', how='outer')

    for fred_file, col_name, extra_fn in [
        ('data/FRED_DGS10.csv', 'treasury_10yr',
         lambda df: df.assign(treasury_10yr_change=df['treasury_10yr'].diff())),
        ('data/FRED_T10Y2Y.csv', 'yield_curve',
         lambda df: df.assign(yield_curve_negative=(df['yield_curve'] < 0).astype(int))),
    ]:
        if os.path.exists(fred_file):
            try:
                fred = pd.read_csv(fred_file, parse_dates=[0])
                fred.columns = ['Date', col_name]
                fred[col_name] = pd.to_numeric(fred[col_name], errors='coerce')
                fred = extra_fn(fred)
                context = context.merge(fred, on='Date', how='outer')
            except Exception:
                pass

    if not context.empty:
        context = context.ffill()

    return context


# ==================== FINNHUB FEATURES (LIVE) ====================

def load_finnhub_features_live(ticker):
    features = {}

    path = f'data/{ticker}_sentiment.json'
    if os.path.exists(path):
        with open(path) as f:
            sent = json.load(f)
        features['news_score'] = sent.get('company_news_score', 0.5)
        features['bullish_pct'] = sent.get('bullish_pct', 0.5)
        features['bearish_pct'] = sent.get('bearish_pct', 0.5)
        features['sentiment_vs_sector'] = (
            sent.get('company_news_score', 0.5) - sent.get('sector_avg_news_score', 0.5))
        features['buzz_ratio'] = (
            sent.get('buzz_articles_week', 0) / max(sent.get('buzz_weekly_avg', 1), 1))

    path = f'data/{ticker}_earnings.csv'
    if os.path.exists(path):
        earnings = pd.read_csv(path)
        if len(earnings) > 0:
            features['last_earnings_surprise'] = earnings.iloc[0].get('surprisePercent', 0)
            features['avg_earnings_surprise'] = earnings['surprisePercent'].mean()
            streak = 0
            for _, row in earnings.iterrows():
                if row.get('surprisePercent', 0) > 0:
                    streak += 1
                else:
                    break
            features['earnings_beat_streak'] = streak

    path = f'data/{ticker}_recommendations.csv'
    if os.path.exists(path):
        recs = pd.read_csv(path)
        if len(recs) > 0:
            latest = recs.iloc[0]
            total = sum(latest.get(k, 0) for k in ['strongBuy', 'buy', 'hold', 'sell', 'strongSell'])
            if total > 0:
                features['analyst_buy_pct'] = (latest.get('strongBuy', 0) + latest.get('buy', 0)) / total
                features['analyst_sell_pct'] = (latest.get('sell', 0) + latest.get('strongSell', 0)) / total
                features['analyst_consensus'] = features['analyst_buy_pct'] - features['analyst_sell_pct']
            if len(recs) > 1:
                prev = recs.iloc[1]
                prev_total = sum(prev.get(k, 0) for k in ['strongBuy', 'buy', 'hold', 'sell', 'strongSell'])
                if prev_total > 0:
                    prev_buy = (prev.get('strongBuy', 0) + prev.get('buy', 0)) / prev_total
                    features['analyst_trend'] = features.get('analyst_buy_pct', 0) - prev_buy

    path = f'data/{ticker}_insider_sentiment.csv'
    if os.path.exists(path):
        isent = pd.read_csv(path)
        if len(isent) > 0:
            features['insider_mspr'] = isent.iloc[-1].get('mspr', 0)
            features['insider_mspr_avg'] = isent['mspr'].mean()
            features['insider_buying'] = int(features['insider_mspr'] > 0)

    daily_news = None
    path = f'data/{ticker}_daily_news.csv'
    if os.path.exists(path):
        daily_news = pd.read_csv(path, parse_dates=['date'])
        daily_news = daily_news.rename(columns={'date': 'Date', 'news_count': 'daily_news_count'})
        daily_news['Date'] = pd.to_datetime(daily_news['Date'])
        daily_news = daily_news.sort_values('Date')
        daily_news['news_count_ma5'] = daily_news['daily_news_count'].rolling(5, min_periods=1).mean()
        daily_news['news_spike'] = (
            daily_news['daily_news_count'] > daily_news['news_count_ma5'] * 2).astype(int)

    return features, daily_news


# ==================== BUILD LIVE FEATURES ====================

def build_live_features(ticker, market_context):
    """Download fresh OHLCV data and compute all features. Must match build_features.py exactly."""
    raw = yf.download(ticker, period=LIVE_HISTORY_DAYS,
                      interval=INTERVAL, progress=False)
    raw.columns = raw.columns.get_level_values(0)
    raw.reset_index(inplace=True)
    raw = raw.rename(columns={raw.columns[0]: 'Date'})
    raw['Date'] = strip_tz(raw['Date'])   # hourly data is UTC-aware; strip for consistent merging

    # All OHLCV technical features (shared with training)
    df = compute_technical_features(raw)

    # Market context
    df['Date'] = df['Date'].dt.normalize()
    if not market_context.empty:
        mc = market_context.copy()
        mc['Date'] = strip_tz(pd.to_datetime(mc['Date'])).dt.normalize()
        df = df.merge(mc, on='Date', how='left')
        context_cols = [c for c in mc.columns if c != 'Date']
        df[context_cols] = df[context_cols].ffill()

    # Relative strength
    if 'sp500_return' in df.columns:
        df['relative_strength'] = df['return_cc'] - df['sp500_return']
        df['relative_strength_5d'] = df['relative_strength'].rolling(5).sum()
    if 'nasdaq_return' in df.columns:
        df['vs_nasdaq'] = df['return_cc'] - df['nasdaq_return']

    # Finnhub features
    finnhub_features, daily_news = load_finnhub_features_live(ticker)
    for name, val in finnhub_features.items():
        df[name] = val
    if daily_news is not None:
        daily_news['Date'] = pd.to_datetime(daily_news['Date']).dt.normalize()
        df = df.merge(daily_news, on='Date', how='left')
        df['daily_news_count'] = df['daily_news_count'].fillna(0)
        df['news_count_ma5'] = df['news_count_ma5'].fillna(0)
        df['news_spike'] = df['news_spike'].fillna(0)

    df.dropna(inplace=True)
    return df


# ==================== GENERATE SIGNALS ====================

def generate_signals():
    """Generate BUY/HOLD signals.
    Returns: signals, confidences, market_context (pass to execute_trades to skip re-download).
    """
    print("Downloading live market context...")
    market_context = download_market_context_live()
    print(f"Market context ready: {len(market_context.columns)} columns\n")

    signals = {}
    confidences = {}

    for ticker in tickers:
        try:
            df = build_live_features(ticker, market_context)
            model, feature_names = load_model(ticker)

            if feature_names is not None:
                # Align to training feature set
                for f in feature_names:
                    if f not in df.columns:
                        df[f] = 0
                latest = df[feature_names].iloc[[-1]]
            else:
                feature_cols = [c for c in df.columns if c not in DROP_FROM_MODEL]
                latest = df[feature_cols].iloc[[-1]]

            probability = model.predict_proba(latest)[0]
            confidence = float(probability[1])
            confidences[ticker] = round(confidence, 4)

            if confidence >= BUY_THRESHOLD:
                signals[ticker] = 'BUY'
            elif confidence <= SHORT_THRESHOLD:
                signals[ticker] = 'SHORT'
            else:
                signals[ticker] = 'HOLD'

            tag = ' [BUY]' if signals[ticker] == 'BUY' else (' [SHORT]' if signals[ticker] == 'SHORT' else '')
            print(f"{ticker}: {signals[ticker]:<5} "
                  f"(DOWN: {probability[0]:.2f}, UP: {confidence:.2f}){tag}")

        except Exception as e:
            print(f"{ticker}: ERROR — {e}")
            signals[ticker] = 'HOLD'
            confidences[ticker] = 0.5

    return signals, confidences, market_context


if __name__ == '__main__':
    signals, confidences, _ = generate_signals()
    print(f"\nFinal signals: {signals}")
    print(f"Confidences:   {confidences}")
