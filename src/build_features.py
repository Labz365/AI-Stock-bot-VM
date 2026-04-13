"""
FEATURE BUILDING — training data
Loads CSVs, computes all technical + alternative features, saves *_features.csv
"""

import pandas as pd
import numpy as np
import os
import json

from features import compute_technical_features, strip_tz
from bot_config import TICKERS as tickers, TARGET_DAYS, TARGET_GAIN_PCT


# ==================== MARKET CONTEXT ====================

def load_market_context():
    context = pd.DataFrame()

    if os.path.exists('data/VIX.csv'):
        vix = pd.read_csv('data/VIX.csv', parse_dates=[0])
        vix.columns = ['Date', 'vix']
        vix['vix_change'] = vix['vix'].pct_change()
        vix['vix_ma5'] = vix['vix'].rolling(5).mean()
        vix['vix_high'] = (vix['vix'] > 25).astype(int)
        vix['vix_extreme'] = (vix['vix'] > 35).astype(int)
        context = vix

    if os.path.exists('data/SPY.csv'):
        spy = pd.read_csv('data/SPY.csv', parse_dates=[0])
        spy = spy.rename(columns={spy.columns[0]: 'Date'})
        spy['sp500_return'] = np.log(spy['Close'] / spy['Close'].shift(1))
        spy['sp500_trend'] = (spy['Close'].rolling(10).mean() > spy['Close'].rolling(50).mean()).astype(int)
        spy['sp500_weekly'] = np.log(spy['Close'] / spy['Close'].shift(5))
        spy = spy[['Date', 'sp500_return', 'sp500_trend', 'sp500_weekly']]
        context = spy if context.empty else context.merge(spy, on='Date', how='outer')

    if os.path.exists('data/QQQ.csv'):
        qqq = pd.read_csv('data/QQQ.csv', parse_dates=[0])
        qqq = qqq.rename(columns={qqq.columns[0]: 'Date'})
        qqq['nasdaq_return'] = np.log(qqq['Close'] / qqq['Close'].shift(1))
        qqq['nasdaq_weekly'] = np.log(qqq['Close'] / qqq['Close'].shift(5))
        qqq = qqq[['Date', 'nasdaq_return', 'nasdaq_weekly']]
        context = context.merge(qqq, on='Date', how='outer')

    if os.path.exists('data/XLK.csv') and os.path.exists('data/SPY.csv'):
        xlk = pd.read_csv('data/XLK.csv', parse_dates=[0]).rename(columns={pd.read_csv('data/XLK.csv').columns[0]: 'Date'})
        spy_f = pd.read_csv('data/SPY.csv', parse_dates=[0]).rename(columns={pd.read_csv('data/SPY.csv').columns[0]: 'Date'})
        xlk = xlk.rename(columns={xlk.columns[0]: 'Date'})
        spy_f = spy_f.rename(columns={spy_f.columns[0]: 'Date'})
        m = xlk[['Date', 'Close']].merge(spy_f[['Date', 'Close']], on='Date', suffixes=('_xlk', '_spy'))
        m['tech_vs_market'] = np.log(m['Close_xlk'] / m['Close_spy'])
        m['tech_vs_market_change'] = m['tech_vs_market'].diff()
        context = context.merge(m[['Date', 'tech_vs_market', 'tech_vs_market_change']], on='Date', how='outer')

    if os.path.exists('data/TLT.csv'):
        tlt = pd.read_csv('data/TLT.csv', parse_dates=[0]).rename(columns={pd.read_csv('data/TLT.csv').columns[0]: 'Date'})
        tlt = tlt.rename(columns={tlt.columns[0]: 'Date'})
        tlt['bond_return'] = np.log(tlt['Close'] / tlt['Close'].shift(1))
        tlt['bond_weekly'] = np.log(tlt['Close'] / tlt['Close'].shift(5))
        context = context.merge(tlt[['Date', 'bond_return', 'bond_weekly']], on='Date', how='outer')

    if os.path.exists('data/GLD.csv'):
        gld = pd.read_csv('data/GLD.csv', parse_dates=[0]).rename(columns={pd.read_csv('data/GLD.csv').columns[0]: 'Date'})
        gld = gld.rename(columns={gld.columns[0]: 'Date'})
        gld['gold_return'] = np.log(gld['Close'] / gld['Close'].shift(1))
        gld['gold_weekly'] = np.log(gld['Close'] / gld['Close'].shift(5))
        context = context.merge(gld[['Date', 'gold_return', 'gold_weekly']], on='Date', how='outer')

    if os.path.exists('data/UUP.csv'):
        uup = pd.read_csv('data/UUP.csv', parse_dates=[0]).rename(columns={pd.read_csv('data/UUP.csv').columns[0]: 'Date'})
        uup = uup.rename(columns={uup.columns[0]: 'Date'})
        uup['dollar_return'] = np.log(uup['Close'] / uup['Close'].shift(1))
        uup['dollar_weekly'] = np.log(uup['Close'] / uup['Close'].shift(5))
        context = context.merge(uup[['Date', 'dollar_return', 'dollar_weekly']], on='Date', how='outer')

    if os.path.exists('data/FRED_DGS10.csv'):
        t10 = pd.read_csv('data/FRED_DGS10.csv', parse_dates=[0])
        t10.columns = ['Date', 'treasury_10yr']
        t10['treasury_10yr'] = pd.to_numeric(t10['treasury_10yr'], errors='coerce')
        t10['treasury_10yr_change'] = t10['treasury_10yr'].diff()
        context = context.merge(t10, on='Date', how='outer')

    if os.path.exists('data/FRED_T10Y2Y.csv'):
        yc = pd.read_csv('data/FRED_T10Y2Y.csv', parse_dates=[0])
        yc.columns = ['Date', 'yield_curve']
        yc['yield_curve'] = pd.to_numeric(yc['yield_curve'], errors='coerce')
        yc['yield_curve_negative'] = (yc['yield_curve'] < 0).astype(int)
        context = context.merge(yc, on='Date', how='outer')

    if not context.empty:
        non_date = [c for c in context.columns if c != 'Date']
        context[non_date] = context[non_date].ffill()

    return context


# ==================== FINNHUB ALTERNATIVE DATA ====================

def load_finnhub_features(ticker):
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
            total = (latest.get('strongBuy', 0) + latest.get('buy', 0) +
                     latest.get('hold', 0) + latest.get('sell', 0) + latest.get('strongSell', 0))
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


# ==================== MAIN ====================

print("Loading market context data...")
market_context = load_market_context()
if not market_context.empty:
    print(f"Market context: {len(market_context)} rows, {len(market_context.columns)} columns\n")
else:
    print("No market context data found. Run download_all_data.py first.\n")

for ticker in tickers:
    print(f"{'='*60}")
    print(f"--- {ticker} ---")
    print(f"{'='*60}")

    df = pd.read_csv(f'data/{ticker}.csv', parse_dates=[0])
    date_col = df.columns[0]
    df = df.rename(columns={date_col: 'Date'})
    # Strip timezone (hourly yfinance data is UTC-aware; daily context CSVs are tz-naive)
    df['Date'] = strip_tz(df['Date'])

    # ===== TECHNICAL FEATURES =====
    df = compute_technical_features(df)

    # ===== FILTER LOW-VOLUME DAYS (after rolling calcs are done) =====
    avg_vol = df['Volume'].rolling(20).mean()
    df = df[df['Volume'] > avg_vol * 0.5].copy()

    # ===== MERGE MARKET CONTEXT =====
    df['Date'] = df['Date'].dt.normalize()
    if not market_context.empty:
        mc = market_context.copy()
        mc['Date'] = strip_tz(pd.to_datetime(mc['Date'])).dt.normalize()
        df = df.merge(mc, on='Date', how='left')
        context_cols = [c for c in mc.columns if c != 'Date']
        df[context_cols] = df[context_cols].ffill()

    # ===== RELATIVE STRENGTH =====
    if 'sp500_return' in df.columns:
        df['relative_strength'] = df['return_cc'] - df['sp500_return']
        df['relative_strength_5d'] = df['relative_strength'].rolling(5).sum()
    if 'nasdaq_return' in df.columns:
        df['vs_nasdaq'] = df['return_cc'] - df['nasdaq_return']

    # ===== FINNHUB ALTERNATIVE DATA =====
    finnhub_features, daily_news = load_finnhub_features(ticker)
    for name, val in finnhub_features.items():
        df[name] = val
    if daily_news is not None:
        daily_news['Date'] = pd.to_datetime(daily_news['Date']).dt.normalize()
        df = df.merge(daily_news, on='Date', how='left')
        df['daily_news_count'] = df['daily_news_count'].fillna(0)
        df['news_count_ma5'] = df['news_count_ma5'].fillna(0)
        df['news_spike'] = df['news_spike'].fillna(0)

    # ===== TARGET: N-day forward return > GAIN_PCT =====
    df['target'] = (df['Close'].shift(-TARGET_DAYS) > df['Close'] * (1 + TARGET_GAIN_PCT)).astype(int)

    # ===== CLEANUP =====
    print(f"Rows before cleanup: {len(df)}")
    nan_counts = df.isna().sum()
    bad_cols = nan_counts[nan_counts > 0]
    if len(bad_cols) > 0:
        print(f"NaN columns:\n{bad_cols}")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Rows after cleanup: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Target balance: {df['target'].value_counts(normalize=True).to_dict()}")

    alt_features = [f for f in finnhub_features.keys() if f in df.columns]
    if alt_features:
        print(f"Finnhub features: {alt_features}")

    df.to_csv(f'data/{ticker}_features.csv', index=False)
    print(f"Saved data/{ticker}_features.csv\n")
