import yfinance as yf
from bot_config import TICKERS as tickers, INTERVAL

# yfinance max history per interval:
#   1d  → full history (start from 2018)
#   1h  → 730 days
#   30m → 60 days  (limited — use 1h instead for training)
PERIOD_MAP = {
    '1d':  {'start': '2018-01-01'},
    '1h':  {'period': '730d'},
    '30m': {'period': '60d'},
}
kwargs = PERIOD_MAP.get(INTERVAL, {'period': '730d'})

for ticker in tickers:
    print(f"Downloading {ticker} ({INTERVAL} bars)...")
    data = yf.download(ticker, interval=INTERVAL, **kwargs)
    data.columns = data.columns.get_level_values(0)
    data.to_csv(f'data/{ticker}.csv')
    print(f"  {len(data)} rows saved to data/{ticker}.csv")

print(f"\nAll stock data downloaded ({INTERVAL} bars).")
