"""In-process fakes for Alpaca.

These reproduce the *shape* of Alpaca's responses so integration tests can run
offline. They do not reproduce real market data or real fills, and no test
derives a performance claim from them.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Iterable, Sequence


class FakeAlpacaClient:
    """Stand-in for :class:`stockbot.alpaca_client.AlpacaClient`."""

    def __init__(
        self,
        settings,
        bars: dict[str, list[dict]] | None = None,
        market_open: bool = True,
        equity: float = 100_000.0,
    ) -> None:
        from zoneinfo import ZoneInfo

        self.settings = settings
        self.tz = ZoneInfo(settings.timezone)
        self._bars = bars or {}
        self.market_open = market_open
        self.equity = equity
        self.positions: list[dict] = []
        self.open_orders: list[dict] = []
        self.submitted: list[dict] = []
        self._order_seq = 0
        self.reject_next = False

    # -- account ---------------------------------------------------------
    def get_account(self) -> dict:
        return {
            "account_number": "PAPER0001",
            "status": "ACTIVE",
            "equity": self.equity,
            "last_equity": self.equity,
            "cash": self.equity,
            "buying_power": self.equity * 2,
            "long_market_value": sum(p.get("market_value", 0.0) for p in self.positions),
            "daytrade_count": 0,
            "account_blocked": False,
            "trading_blocked": False,
            "pattern_day_trader": False,
        }

    def get_positions(self) -> list[dict]:
        return list(self.positions)

    def get_open_orders(self, symbols: Sequence[str] | None = None) -> list[dict]:
        if symbols:
            return [o for o in self.open_orders if o["symbol"] in symbols]
        return list(self.open_orders)

    def get_order_by_client_id(self, client_order_id: str) -> dict | None:
        for order in self.submitted:
            if order["client_order_id"] == client_order_id:
                return order
        return None

    def submit_market_order(self, symbol, qty, side, client_order_id, time_in_force="day"):
        from stockbot.alpaca_client import AlpacaError

        if self.reject_next:
            self.reject_next = False
            raise AlpacaError("simulated rejection")
        self._order_seq += 1
        order = {
            "id": f"fake-{self._order_seq}",
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side,
            "qty": float(qty),
            "notional": None,
            "filled_qty": 0.0,
            "filled_avg_price": None,
            "status": "accepted",
            "type": "market",
            "submitted_at": datetime.now(timezone.utc),
            "created_at": datetime.now(timezone.utc),
        }
        self.submitted.append(order)
        self.open_orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> None:
        self.open_orders = [o for o in self.open_orders if o["id"] != order_id]

    # -- clock / calendar -------------------------------------------------
    def get_clock(self) -> dict:
        now = datetime.now(timezone.utc)
        return {
            "timestamp": now,
            "is_open": self.market_open,
            "next_open": now + timedelta(hours=1),
            "next_close": now + timedelta(hours=6),
        }

    def get_calendar(self, start, end) -> list[dict]:
        days = []
        cursor = start
        while cursor <= end:
            if cursor.weekday() < 5:
                days.append({"date": cursor, "open": time(9, 30), "close": time(16, 0)})
            cursor += timedelta(days=1)
        return days

    def session_minutes_map(self, start, end) -> dict:
        return {day["date"]: 390.0 for day in self.get_calendar(start, end)}

    # -- data -------------------------------------------------------------
    def get_bars(self, symbols: Iterable[str], start, end=None, limit=None) -> dict[str, list[dict]]:
        return {symbol: list(self._bars.get(symbol, [])) for symbol in symbols}

    def get_latest_quotes(self, symbols: Iterable[str]) -> dict[str, dict]:
        out = {}
        for symbol in symbols:
            rows = self._bars.get(symbol) or []
            price = rows[-1]["close"] if rows else 100.0
            out[symbol] = {
                "bid": price * 0.9999,
                "ask": price * 1.0001,
                "bid_size": 500,
                "ask_size": 500,
                "mid": price,
                "spread_bps": 2.0,
                "timestamp": datetime.now(timezone.utc),
            }
        return out

    def get_snapshots(self, symbols: Iterable[str]) -> dict[str, dict]:
        return {
            symbol: {"last_price": 100.0, "last_trade_at": datetime.now(timezone.utc)}
            for symbol in symbols
        }

    def connection_test(self) -> dict:
        return {"ok": True, "paper": True, "checks": {"account": {"ok": True}}}

    # -- helpers for tests -------------------------------------------------
    def fill_all(self, price: float = 100.0) -> None:
        for order in self.open_orders:
            order["status"] = "filled"
            order["filled_qty"] = order["qty"]
            order["filled_avg_price"] = price
        self.open_orders = []

    def partially_fill_all(self, fraction: float = 0.4, price: float = 100.0) -> None:
        for order in self.open_orders:
            order["status"] = "partially_filled"
            order["filled_qty"] = order["qty"] * fraction
            order["filled_avg_price"] = price

    def reject_all(self) -> None:
        for order in self.open_orders:
            order["status"] = "rejected"
        self.open_orders = []

    def cancel_all(self) -> None:
        for order in self.open_orders:
            order["status"] = "canceled"
        self.open_orders = []
