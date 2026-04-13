"""
DOWNLOAD ALL ALTERNATIVE + MARKET CONTEXT DATA
================================================
Run this ONCE before build_features.py

Data sources:
1. Finnhub (free, 60 calls/min): sentiment, insider trades, earnings, recommendations
2. yfinance (free): VIX, sector ETFs, bonds, gold, dollar
3. FRED via pandas_datareader (free): treasury yields, unemployment, CPI

Setup: 
  pip install finnhub-python pandas_datareader
  Get free API key at https://finnhub.io/register
"""

import finnhub
import pandas as pd
import numpy as np
import yfinance as yf
import time
import os
import json
from datetime import datetime, timedelta

# ==================== CONFIG ====================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bot_config import TICKERS as tickers

FINNHUB_API_KEY = 'YOUR_FINNHUB_KEY_HERE'  # Get free at https://finnhub.io/register
START_DATE = '2018-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')


# ==================== PART 1: FINNHUB ALTERNATIVE DATA ====================

def download_finnhub_data():
    """Download sentiment, insider, earnings, and recommendation data from Finnhub."""
    if FINNHUB_API_KEY == 'YOUR_FINNHUB_KEY_HERE':
        print("WARNING: No Finnhub API key set. Skipping Finnhub data.")
        print("Get a free key at https://finnhub.io/register")
        print("Then update FINNHUB_API_KEY in this file.\n")
        return

    client = finnhub.Client(api_key=FINNHUB_API_KEY)

    for ticker in tickers:
        print(f"\n--- {ticker}: Finnhub Data ---")

        # 1. NEWS SENTIMENT
        try:
            sentiment = client.news_sentiment(ticker)
            sentiment_data = {
                'buzz_articles_week': sentiment.get('buzz', {}).get('articlesInLastWeek', 0),
                'buzz_score': sentiment.get('buzz', {}).get('buzz', 0),
                'buzz_weekly_avg': sentiment.get('buzz', {}).get('weeklyAverage', 0),
                'company_news_score': sentiment.get('companyNewsScore', 0),
                'sector_avg_bullish': sentiment.get('sectorAverageBullishPercent', 0),
                'sector_avg_news_score': sentiment.get('sectorAverageNewsScore', 0),
                'bearish_pct': sentiment.get('sentiment', {}).get('bearishPercent', 0),
                'bullish_pct': sentiment.get('sentiment', {}).get('bullishPercent', 0),
            }
            with open(f'data/{ticker}_sentiment.json', 'w') as f:
                json.dump(sentiment_data, f, indent=2)
            print(f"  Sentiment: bullish={sentiment_data['bullish_pct']:.0%}, "
                  f"news_score={sentiment_data['company_news_score']:.4f}")
            time.sleep(1)  # Rate limit
        except Exception as e:
            print(f"  Sentiment failed: {e}")

        # 2. INSIDER TRANSACTIONS (last 12 months)
        try:
            end = datetime.now().strftime('%Y-%m-%d')
            start = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            insiders = client.stock_insider_transactions(ticker, start, end)
            
            if insiders.get('data'):
                insider_df = pd.DataFrame(insiders['data'])
                insider_df.to_csv(f'data/{ticker}_insiders.csv', index=False)
                
                # Summary stats
                buys = len(insider_df[insider_df['change'] > 0])
                sells = len(insider_df[insider_df['change'] < 0])
                print(f"  Insider trades: {buys} buys, {sells} sells (12 months)")
            else:
                print(f"  No insider data available")
            time.sleep(1)
        except Exception as e:
            print(f"  Insider data failed: {e}")

        # 3. EARNINGS SURPRISES (last 20 quarters)
        try:
            earnings = client.company_earnings(ticker, limit=20)
            if earnings:
                earnings_df = pd.DataFrame(earnings)
                earnings_df.to_csv(f'data/{ticker}_earnings.csv', index=False)
                
                # Recent surprise
                latest = earnings[0]
                surprise_pct = latest.get('surprisePercent', 0)
                print(f"  Last earnings surprise: {surprise_pct:.2f}%")
            time.sleep(1)
        except Exception as e:
            print(f"  Earnings failed: {e}")

        # 4. ANALYST RECOMMENDATIONS
        try:
            recs = client.recommendation_trends(ticker)
            if recs:
                recs_df = pd.DataFrame(recs)
                recs_df.to_csv(f'data/{ticker}_recommendations.csv', index=False)
                
                latest = recs[0]
                print(f"  Recommendations: {latest.get('strongBuy',0)} strong buy, "
                      f"{latest.get('buy',0)} buy, {latest.get('hold',0)} hold, "
                      f"{latest.get('sell',0)} sell")
            time.sleep(1)
        except Exception as e:
            print(f"  Recommendations failed: {e}")

        # 5. INSIDER SENTIMENT (MSPR - Monthly Share Purchase Ratio)
        try:
            insider_sent = client.stock_insider_sentiment(ticker, '2020-01-01', end)
            if insider_sent.get('data'):
                isent_df = pd.DataFrame(insider_sent['data'])
                isent_df.to_csv(f'data/{ticker}_insider_sentiment.csv', index=False)
                print(f"  Insider sentiment: {len(isent_df)} monthly records")
            time.sleep(1)
        except Exception as e:
            print(f"  Insider sentiment failed: {e}")

        # 6. COMPANY NEWS (last 30 days for daily sentiment scoring)
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            all_news = []
            current_end = end_date
            
            # Fetch in 30-day chunks (Finnhub limit)
            for _ in range(12):  # 12 months
                current_start = (datetime.strptime(current_end, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')
                news = client.company_news(ticker, _from=current_start, to=current_end)
                if news:
                    all_news.extend(news)
                current_end = current_start
                time.sleep(1)  # Rate limit
            
            if all_news:
                news_df = pd.DataFrame(all_news)
                news_df['date'] = pd.to_datetime(news_df['datetime'], unit='s').dt.date
                
                # Count articles per day as a proxy for attention/buzz
                daily_buzz = news_df.groupby('date').size().reset_index(name='news_count')
                daily_buzz.to_csv(f'data/{ticker}_daily_news.csv', index=False)
                print(f"  News: {len(all_news)} articles, {len(daily_buzz)} unique days")
        except Exception as e:
            print(f"  News download failed: {e}")


# ==================== PART 2: MARKET CONTEXT (yfinance) ====================

def download_market_context():
    """Download VIX, sector ETFs, bonds, gold, dollar."""
    print("\n\n========== MARKET CONTEXT DATA ==========")

    # VIX
    print("\n--- VIX (Fear Index) ---")
    vix = yf.download('^VIX', start=START_DATE, end=END_DATE)
    vix.columns = vix.columns.get_level_values(0)
    vix = vix[['Close']].rename(columns={'Close': 'vix'})
    vix.to_csv('data/VIX.csv')
    print(f"VIX: {len(vix)} rows")

    # Market ETFs
    etfs = {
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq 100',
        'XLK': 'Tech sector',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'TLT': 'Bonds 20yr',
        'GLD': 'Gold',
        'UUP': 'US Dollar',
    }

    print("\n--- Market ETFs ---")
    for ticker, name in etfs.items():
        df = yf.download(ticker, start=START_DATE, end=END_DATE)
        df.columns = df.columns.get_level_values(0)
        df.to_csv(f'data/{ticker}.csv')
        print(f"{ticker} ({name}): {len(df)} rows")


# ==================== PART 3: ECONOMIC DATA (FRED) ====================

def download_fred_data():
    """Download treasury yields, unemployment, CPI from FRED."""
    print("\n\n========== FRED ECONOMIC DATA ==========\n")

    try:
        import pandas_datareader.data as pdr
    except ImportError:
        print("pandas_datareader not installed. Run: pip install pandas_datareader")
        return

    series = {
        'DGS10': '10yr Treasury yield',
        'DGS2': '2yr Treasury yield',
        'T10Y2Y': 'Yield curve (10yr-2yr)',
        'DFF': 'Fed funds rate',
        'UNRATE': 'Unemployment rate',
        'CPIAUCSL': 'CPI (inflation)',
        'ICSA': 'Jobless claims',
    }

    for code, name in series.items():
        try:
            df = pdr.DataReader(code, 'fred', START_DATE, END_DATE)
            df.to_csv(f'data/FRED_{code}.csv')
            print(f"{code} ({name}): {len(df)} rows")
        except Exception as e:
            print(f"{code} failed: {e}")


# ==================== RUN ALL ====================

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    print("========== FINNHUB ALTERNATIVE DATA ==========")
    download_finnhub_data()

    download_market_context()
    download_fred_data()

    print("\n\n=== ALL DATA DOWNLOADED ===")
    print("Next: python src/build_features.py")