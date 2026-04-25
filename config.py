import os
from alpaca_trade_api import REST
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ['API_KEY']
SECRET_KEY = os.environ['SECRET_KEY']
BASE_URL = os.environ.get('BASE_URL', 'https://paper-api.alpaca.markets')

api = REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

# Quick test
if __name__ == '__main__':
    account = api.get_account()
    print(f"Account status: {account.status}")
    print(f"Cash: ${account.cash}")
    print(f"Portfolio value: ${account.portfolio_value}")

    positions = api.list_positions()
    if positions:
        for p in positions:
            print(f"Position: {p.symbol} | {p.qty} shares | P/L: ${p.unrealized_pl}")
    else:
        print("No open positions")
