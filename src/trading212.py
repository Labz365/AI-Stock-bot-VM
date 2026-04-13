"""
TRADING 212 API CLIENT
======================
Wraps the Trading 212 REST API.
Set demo=True (default) to use the paper-trading demo environment.

Docs: https://t212public-api-docs.redoc.ly/
"""

import requests
import time


DEMO_BASE = "https://demo.trading212.com/api/v0"
LIVE_BASE = "https://live.trading212.com/api/v0"


class Trading212:

    def __init__(self, api_key: str, demo: bool = True):
        self.base  = DEMO_BASE if demo else LIVE_BASE
        self._demo = demo
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization":  api_key,
            "Content-Type":   "application/json",
        })
        self._instrument_cache = None   # populated lazily

    # ── private helpers ──────────────────────────────────────────────────────

    def _get(self, path: str, **params):
        r = self._session.get(f"{self.base}{path}", params=params or None)
        self._raise(r)
        return r.json()

    def _post(self, path: str, body: dict = None):
        r = self._session.post(f"{self.base}{path}", json=body)
        self._raise(r)
        return r.json()

    def _put(self, path: str, body: dict = None):
        r = self._session.put(f"{self.base}{path}", json=body)
        self._raise(r)
        return r.json()

    def _delete(self, path: str):
        r = self._session.delete(f"{self.base}{path}")
        self._raise(r)
        return r.status_code

    @staticmethod
    def _raise(response: requests.Response):
        if not response.ok:
            raise requests.HTTPError(
                f"T212 API {response.status_code}: {response.text[:300]}",
                response=response,
            )

    # ── Account ──────────────────────────────────────────────────────────────

    def get_cash(self) -> dict:
        """Returns free, invested, result, total (all in account currency)."""
        return self._get("/equity/account/cash")

    def get_account_metadata(self) -> dict:
        return self._get("/equity/account/metadata")

    # ── Portfolio ─────────────────────────────────────────────────────────────

    def get_portfolio(self) -> list:
        """Open positions. Each has ticker, quantity, averagePrice, currentPrice, ppl."""
        return self._get("/equity/portfolio")

    # ── Orders ───────────────────────────────────────────────────────────────

    def get_orders(self) -> list:
        return self._get("/equity/orders")

    def place_market_buy(self, t212_ticker: str, quantity: float) -> dict:
        """Buy `quantity` shares (fractional supported)."""
        return self._post("/equity/orders/market", {
            "ticker":   t212_ticker,
            "quantity": round(quantity, 4),
        })

    def place_market_sell(self, t212_ticker: str, quantity: float) -> dict:
        """Sell `quantity` shares (use positive number — this method sets direction)."""
        return self._post("/equity/orders/market", {
            "ticker":   t212_ticker,
            "quantity": -abs(round(quantity, 4)),
        })

    def cancel_order(self, order_id) -> int:
        return self._delete(f"/equity/orders/{order_id}")

    # ── Instruments ──────────────────────────────────────────────────────────

    def get_instruments(self) -> list:
        """All tradeable instruments. Cached after first call."""
        if self._instrument_cache is None:
            self._instrument_cache = self._get("/equity/metadata/instruments")
        return self._instrument_cache

    def find_ticker(self, symbol: str) -> str | None:
        """
        Map a standard symbol (e.g. 'AAPL') → T212 ticker ID (e.g. 'AAPL_US_EQ').
        Tries exact shortName match first, then prefix search.
        Returns None if not found on this platform.
        """
        symbol = symbol.upper()
        instruments = self.get_instruments()

        # 1. Exact shortName match (US equity preferred)
        for inst in instruments:
            if (inst.get("shortName", "").upper() == symbol
                    and inst.get("type") == "STOCK"
                    and "_US_" in inst.get("ticker", "")):
                return inst["ticker"]

        # 2. Any shortName match
        for inst in instruments:
            if inst.get("shortName", "").upper() == symbol:
                return inst["ticker"]

        # 3. Ticker ID prefix (e.g. AAPL_US_EQ starts with AAPL_)
        for inst in instruments:
            if inst.get("ticker", "").startswith(symbol + "_"):
                return inst["ticker"]

        return None

    def map_tickers(self, symbols: list[str]) -> dict[str, str | None]:
        """Batch map {symbol -> t212_ticker}. Logs missing ones."""
        mapping = {}
        for sym in symbols:
            t = self.find_ticker(sym)
            if t is None:
                print(f"  WARNING: '{sym}' not found on T212 — will be skipped")
            mapping[sym] = t
        return mapping

    # ── Pies ─────────────────────────────────────────────────────────────────

    def get_pies(self) -> list:
        """List all pies (summary, no instrument detail)."""
        return self._get("/equity/pies")

    def get_pie(self, pie_id: int) -> dict:
        """Full pie detail including instruments and performance."""
        return self._get(f"/equity/pies/{pie_id}")

    def create_pie(
        self,
        name: str,
        instruments: list[dict],
        icon: str = "Pie",
        dividend_action: str = "REINVEST",
    ) -> dict:
        """
        instruments: [{"ticker": "AAPL_US_EQ", "target": 60.0}, ...]
        targets must sum to exactly 100.0
        """
        _validate_targets(instruments)
        return self._post("/equity/pies", {
            "name":                 name,
            "icon":                 icon,
            "dividendCashAction":   dividend_action,
            "endDate":              None,
            "goal":                 None,
            "instruments":          instruments,
        })

    def update_pie(
        self,
        pie_id: int,
        name: str,
        instruments: list[dict],
        icon: str = "Pie",
        dividend_action: str = "REINVEST",
    ) -> dict:
        _validate_targets(instruments)
        return self._put(f"/equity/pies/{pie_id}", {
            "name":                 name,
            "icon":                 icon,
            "dividendCashAction":   dividend_action,
            "endDate":              None,
            "goal":                 None,
            "instruments":          instruments,
        })

    def delete_pie(self, pie_id: int) -> int:
        return self._delete(f"/equity/pies/{pie_id}")

    def find_pie_by_name(self, name: str) -> dict | None:
        """Return the first pie matching `name`, or None."""
        for pie in self.get_pies():
            settings = pie.get("settings") or pie
            if settings.get("name") == name:
                return pie
        return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _validate_targets(instruments: list[dict]):
    total = round(sum(i["target"] for i in instruments), 2)
    if abs(total - 100.0) > 0.1:
        raise ValueError(f"Pie targets must sum to 100.0, got {total}")
