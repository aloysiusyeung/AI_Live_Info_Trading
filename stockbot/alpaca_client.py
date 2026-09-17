"""Alpaca connectivity — paper trading only.

Guard rails enforced here:
  * ``TradingClient`` is always constructed with ``paper=True``.
  * ``url_override`` is never passed, so the live endpoint cannot be reached.
  * A runtime assertion re-checks the resolved base URL and raises
    :class:`LiveTradingRefused` if it does not point at the paper host.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import (
    NewsRequest,
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import (
    GetAssetsRequest,
    GetCalendarRequest,
    GetOrdersRequest,
    MarketOrderRequest,
)

from .config import LiveTradingRefused, Settings

logger = logging.getLogger(__name__)

PAPER_HOST = "paper-api.alpaca.markets"
LIVE_HOST = "api.alpaca.markets"


class AlpacaError(RuntimeError):
    """Wraps any failure talking to Alpaca."""


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class AlpacaClient:
    """Paper-only facade over alpaca-py's trading and market-data clients."""

    def __init__(self, settings: Settings) -> None:
        if not settings.paper:
            raise LiveTradingRefused("AlpacaClient may only be constructed in paper mode.")
        self.settings = settings
        self.tz = ZoneInfo(settings.timezone)

        api_key, secret_key = settings.credentials()
        # paper=True is hard-coded, not read from a variable a caller could flip.
        self.trading = TradingClient(api_key=api_key, secret_key=secret_key, paper=True)
        self.data = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.news = NewsClient(api_key=api_key, secret_key=secret_key)
        self._assert_paper_endpoint()

    # -- safety ----------------------------------------------------------
    def _assert_paper_endpoint(self) -> None:
        # alpaca-py stores this as a BaseURL enum, whose str() is the enum name
        # rather than the URL, so read .value when it is present.
        raw = getattr(self.trading, "_base_url", "")
        base = str(getattr(raw, "value", raw) or "")
        if LIVE_HOST in base and PAPER_HOST not in base:
            raise LiveTradingRefused(
                f"Trading client resolved to a live endpoint ({base}); refusing to continue."
            )
        if base and PAPER_HOST not in base:
            logger.warning("Unrecognised trading endpoint", extra={"endpoint": base})

    @property
    def timeframe(self) -> TimeFrame:
        return TimeFrame(self.settings.bar_minutes, TimeFrameUnit.Minute)

    # -- account ---------------------------------------------------------
    def get_account(self) -> dict:
        try:
            acct = self.trading.get_account()
        except Exception as exc:  # noqa: BLE001 - surface as AlpacaError
            raise AlpacaError(f"get_account failed: {exc}") from exc
        return {
            "account_number": getattr(acct, "account_number", None),
            "status": str(getattr(acct, "status", "")),
            "currency": getattr(acct, "currency", None),
            "equity": _as_float(getattr(acct, "equity", None)),
            "last_equity": _as_float(getattr(acct, "last_equity", None)),
            "cash": _as_float(getattr(acct, "cash", None)),
            "buying_power": _as_float(getattr(acct, "buying_power", None)),
            "long_market_value": _as_float(getattr(acct, "long_market_value", None)),
            "daytrade_count": getattr(acct, "daytrade_count", None),
            "account_blocked": bool(getattr(acct, "account_blocked", False)),
            "trading_blocked": bool(getattr(acct, "trading_blocked", False)),
            "pattern_day_trader": bool(getattr(acct, "pattern_day_trader", False)),
        }

    def get_positions(self) -> list[dict]:
        try:
            positions = self.trading.get_all_positions()
        except Exception as exc:  # noqa: BLE001
            raise AlpacaError(f"get_all_positions failed: {exc}") from exc
        return [
            {
                "symbol": p.symbol,
                "qty": _as_float(p.qty) or 0.0,
                "side": str(getattr(p, "side", "")),
                "avg_entry_price": _as_float(p.avg_entry_price),
                "market_value": _as_float(p.market_value),
                "cost_basis": _as_float(p.cost_basis),
                "unrealized_pl": _as_float(p.unrealized_pl),
                "unrealized_plpc": _as_float(p.unrealized_plpc),
                "current_price": _as_float(p.current_price),
            }
            for p in positions
        ]

    def get_open_orders(self, symbols: Sequence[str] | None = None) -> list[dict]:
        request = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=list(symbols) if symbols else None,
            limit=500,
        )
        try:
            orders = self.trading.get_orders(filter=request)
        except Exception as exc:  # noqa: BLE001
            raise AlpacaError(f"get_orders failed: {exc}") from exc
        return [self._order_to_dict(o) for o in orders]

    def get_order_by_client_id(self, client_order_id: str) -> dict | None:
        try:
            order = self.trading.get_order_by_client_id(client_order_id)
        except Exception:  # noqa: BLE001 - not found is a normal outcome
            return None
        return self._order_to_dict(order)

    @staticmethod
    def _order_to_dict(order: Any) -> dict:
        return {
            "id": str(getattr(order, "id", "")),
            "client_order_id": getattr(order, "client_order_id", None),
            "symbol": getattr(order, "symbol", None),
            "side": str(getattr(order, "side", "")).lower().replace("orderside.", ""),
            "qty": _as_float(getattr(order, "qty", None)),
            "notional": _as_float(getattr(order, "notional", None)),
            "filled_qty": _as_float(getattr(order, "filled_qty", None)) or 0.0,
            "filled_avg_price": _as_float(getattr(order, "filled_avg_price", None)),
            "status": str(getattr(order, "status", "")).lower().replace("orderstatus.", ""),
            "type": str(getattr(order, "order_type", getattr(order, "type", ""))).lower(),
            "submitted_at": getattr(order, "submitted_at", None),
            "created_at": getattr(order, "created_at", None),
        }

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        client_order_id: str,
        time_in_force: str = "day",
    ) -> dict:
        """Submit a market order to the *paper* account.

        Callers must have passed the risk engine first; this method performs no
        risk checks of its own beyond refusing non-paper mode.
        """
        if not self.settings.paper:
            raise LiveTradingRefused("submit_market_order is paper-only.")
        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC,
            client_order_id=client_order_id,
            extended_hours=False,
        )
        try:
            order = self.trading.submit_order(order_data=request)
        except Exception as exc:  # noqa: BLE001
            raise AlpacaError(f"submit_order failed for {symbol}: {exc}") from exc
        return self._order_to_dict(order)

    def cancel_order(self, order_id: str) -> None:
        try:
            self.trading.cancel_order_by_id(order_id)
        except Exception as exc:  # noqa: BLE001
            raise AlpacaError(f"cancel_order failed: {exc}") from exc

    # -- market clock / calendar -----------------------------------------
    def get_clock(self) -> dict:
        """Alpaca's authoritative market clock.

        ``next_open``/``next_close`` come back tz-aware from Alpaca, which is how
        DST, early closes and holidays are handled correctly without a local
        holiday table.
        """
        try:
            clock = self.trading.get_clock()
        except Exception as exc:  # noqa: BLE001
            raise AlpacaError(f"get_clock failed: {exc}") from exc
        return {
            "timestamp": _ensure_utc(clock.timestamp),
            "is_open": bool(clock.is_open),
            "next_open": _ensure_utc(clock.next_open),
            "next_close": _ensure_utc(clock.next_close),
        }

    def get_calendar(self, start: date, end: date) -> list[dict]:
        """Trading sessions between two dates, including early-close days."""
        try:
            days = self.trading.get_calendar(GetCalendarRequest(start=start, end=end))
        except Exception as exc:  # noqa: BLE001
            raise AlpacaError(f"get_calendar failed: {exc}") from exc
        out: list[dict] = []
        for day in days:
            out.append(
                {
                    "date": day.date,
                    "open": day.open,   # datetime.time in America/New_York
                    "close": day.close,
                }
            )
        return out

    def session_minutes_map(self, start: date, end: date) -> dict[date, float]:
        """``{trading date: scheduled session length in minutes}``.

        Early-close days come back shorter, straight from Alpaca's calendar, so
        the time-of-day features stay correct on half-days without a local
        holiday table. The schedule is published in advance, so using it is not
        look-ahead information.
        """
        out: dict[date, float] = {}
        for day in self.get_calendar(start, end):
            open_t, close_t = day["open"], day["close"]
            minutes = (
                (close_t.hour * 60 + close_t.minute) - (open_t.hour * 60 + open_t.minute)
            )
            out[day["date"]] = float(minutes)
        return out

    def session_bounds(self, on: date) -> tuple[datetime, datetime] | None:
        """Return (open, close) as tz-aware UTC datetimes for a trading date.

        Returns ``None`` for weekends and holidays. Early closes come straight
        from Alpaca's calendar, so a 13:00 ET close is reflected automatically.
        """
        days = self.get_calendar(on, on)
        for day in days:
            if day["date"] == on:
                open_dt = datetime.combine(on, day["open"], tzinfo=self.tz)
                close_dt = datetime.combine(on, day["close"], tzinfo=self.tz)
                return open_dt.astimezone(timezone.utc), close_dt.astimezone(timezone.utc)
        return None

    # -- market data -----------------------------------------------------
    def get_bars(
        self,
        symbols: Iterable[str],
        start: datetime,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> dict[str, list[dict]]:
        """Historical bars for the configured timeframe, keyed by symbol."""
        symbols = list(symbols)
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=self.timeframe,
            start=start,
            end=end,
            limit=limit,
            feed=self.settings.data_feed,
        )
        try:
            barset = self.data.get_stock_bars(request)
        except Exception as exc:  # noqa: BLE001
            raise AlpacaError(f"get_stock_bars failed: {exc}") from exc

        result: dict[str, list[dict]] = {symbol: [] for symbol in symbols}
        raw = getattr(barset, "data", barset) or {}
        for symbol, bars in raw.items():
            rows = []
            for bar in bars:
                rows.append(
                    {
                        "symbol": symbol,
                        "bar_start": _ensure_utc(bar.timestamp),
                        "bar_minutes": self.settings.bar_minutes,
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": float(bar.volume),
                        "trade_count": _as_float(getattr(bar, "trade_count", None)),
                        "vwap": _as_float(getattr(bar, "vwap", None)),
                        "feed": self.settings.data_feed,
                    }
                )
            result[symbol] = rows
        return result

    def get_latest_quotes(self, symbols: Iterable[str]) -> dict[str, dict]:
        """Latest NBBO-ish quote per symbol; used for the spread check."""
        symbols = list(symbols)
        request = StockLatestQuoteRequest(
            symbol_or_symbols=symbols, feed=self.settings.data_feed
        )
        try:
            quotes = self.data.get_stock_latest_quote(request)
        except Exception as exc:  # noqa: BLE001
            raise AlpacaError(f"get_stock_latest_quote failed: {exc}") from exc

        out: dict[str, dict] = {}
        for symbol, quote in (quotes or {}).items():
            bid = _as_float(getattr(quote, "bid_price", None)) or 0.0
            ask = _as_float(getattr(quote, "ask_price", None)) or 0.0
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else None
            spread_bps = ((ask - bid) / mid * 10_000) if mid and mid > 0 else None
            out[symbol] = {
                "bid": bid,
                "ask": ask,
                "bid_size": _as_float(getattr(quote, "bid_size", None)),
                "ask_size": _as_float(getattr(quote, "ask_size", None)),
                "mid": mid,
                "spread_bps": spread_bps,
                "timestamp": _ensure_utc(getattr(quote, "timestamp", None)),
            }
        return out

    def get_snapshots(self, symbols: Iterable[str]) -> dict[str, dict]:
        request = StockSnapshotRequest(
            symbol_or_symbols=list(symbols), feed=self.settings.data_feed
        )
        try:
            snaps = self.data.get_stock_snapshot(request)
        except Exception as exc:  # noqa: BLE001
            raise AlpacaError(f"get_stock_snapshot failed: {exc}") from exc
        out: dict[str, dict] = {}
        for symbol, snap in (snaps or {}).items():
            trade = getattr(snap, "latest_trade", None)
            out[symbol] = {
                "last_price": _as_float(getattr(trade, "price", None)),
                "last_trade_at": _ensure_utc(getattr(trade, "timestamp", None)),
            }
        return out

    # -- news (market-wide) -----------------------------------------------
    def get_news(
        self,
        start: datetime,
        end: datetime | None = None,
        symbols: Iterable[str] | None = None,
        max_articles: int = 2000,
        sort: str = "asc",
        include_content: bool = False,
        exclude_contentless: bool = False,
    ) -> tuple[list[dict], bool]:
        """Fetch news articles, **market-wide by default**.

        Leaving ``symbols`` as ``None`` omits the symbol filter from the
        request, so Alpaca returns every article for every US symbol it covers,
        each tagged with the symbols it mentions. Passing ``symbols`` narrows
        it, which is only used for targeted backfills.

        ``alpaca-py`` paginates internally and stops only when the feed is
        exhausted or ``limit`` is reached, and it does not hand back a usable
        page token. So ``max_articles`` is a hard cap — without one, a wide
        window would pull the entire feed in a single unbounded call.

        Returns ``(articles, truncated)``. ``truncated`` means the cap was hit
        and more articles exist in the window, so callers record the window as
        incomplete instead of assuming full coverage. With the default
        ascending sort, the caller's watermark advances and the next run
        resumes where this one stopped.
        """
        if max_articles < 1:
            return [], False

        request = NewsRequest(
            start=start,
            end=end,
            symbols=",".join(s.upper() for s in symbols) if symbols else None,
            limit=max_articles,
            sort=sort,
            include_content=include_content,
            exclude_contentless=exclude_contentless,
        )
        try:
            response = self.news.get_news(request)
        except Exception as exc:  # noqa: BLE001
            raise AlpacaError(f"get_news failed: {exc}") from exc

        raw = getattr(response, "data", response) or {}
        batch = raw.get("news", []) if isinstance(raw, dict) else []
        articles = [self._news_to_dict(item) for item in batch]
        return articles, len(articles) >= max_articles

    @staticmethod
    def _news_to_dict(item: Any) -> dict:
        symbols = [str(s).upper() for s in (getattr(item, "symbols", None) or [])]
        return {
            "id": int(getattr(item, "id", 0)),
            "headline": getattr(item, "headline", "") or "",
            "summary": getattr(item, "summary", None),
            "author": getattr(item, "author", None),
            "source": getattr(item, "source", None),
            "url": getattr(item, "url", None),
            "content": getattr(item, "content", None),
            "created_at": _ensure_utc(getattr(item, "created_at", None)),
            "updated_at": _ensure_utc(getattr(item, "updated_at", None)),
            "symbols": symbols,
        }

    # -- assets (the tradable US equity universe) --------------------------
    def get_us_equities(self, active_only: bool = True) -> list[dict]:
        """Every US equity Alpaca lists, used to screen news-derived symbols."""
        request = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE if active_only else None,
        )
        try:
            assets = self.trading.get_all_assets(request)
        except Exception as exc:  # noqa: BLE001
            raise AlpacaError(f"get_all_assets failed: {exc}") from exc
        return [
            {
                "symbol": str(a.symbol).upper(),
                "name": getattr(a, "name", None),
                "exchange": str(getattr(a, "exchange", "") or ""),
                "asset_class": str(getattr(a, "asset_class", "") or ""),
                "status": str(getattr(a, "status", "") or ""),
                "tradable": bool(getattr(a, "tradable", False)),
                "shortable": bool(getattr(a, "shortable", False)),
                "fractionable": bool(getattr(a, "fractionable", False)),
            }
            for a in assets
        ]

    # -- connection test --------------------------------------------------
    def credentials_look_like_placeholders(self) -> bool:
        """True when the configured keys are obviously not real Alpaca keys.

        Real Alpaca keys are alphanumeric with no underscores. A value such as
        ``YOUR_ALPACA_API_KEY_HERE`` is a template that was never filled in, and
        saying so is far more useful than reporting the resulting 401.
        """
        markers = ("YOUR", "PLACEHOLDER", "EXAMPLE", "CHANGEME", "HERE", "TODO", "XXXX")
        for value in self.settings.credentials():
            if not value:
                return True
            upper = value.upper()
            if "_" in value or any(marker in upper for marker in markers):
                return True
        return False

    def connection_test(self) -> dict:
        """Read-only probe. Never submits an order."""
        report: dict[str, Any] = {"paper": True, "checks": {}}
        if self.credentials_look_like_placeholders():
            report["credentials"] = "placeholder"
            report["ok"] = False
            report["error"] = (
                "ALPACA_API_KEY / ALPACA_SECRET_KEY are placeholder values, not real "
                "Alpaca paper keys. Every request will return 401 until they are "
                "replaced with keys from https://app.alpaca.markets/paper/dashboard/overview"
            )
            return report
        report["credentials"] = "present"
        try:
            account = self.get_account()
            report["checks"]["account"] = {
                "ok": True,
                "status": account["status"],
                "equity": account["equity"],
                "trading_blocked": account["trading_blocked"],
            }
        except AlpacaError as exc:
            report["checks"]["account"] = {"ok": False, "error": str(exc)}

        try:
            clock = self.get_clock()
            report["checks"]["clock"] = {
                "ok": True,
                "is_open": clock["is_open"],
                "next_open": clock["next_open"].isoformat() if clock["next_open"] else None,
                "next_close": clock["next_close"].isoformat() if clock["next_close"] else None,
            }
        except AlpacaError as exc:
            report["checks"]["clock"] = {"ok": False, "error": str(exc)}

        try:
            today = datetime.now(self.tz).date()
            calendar = self.get_calendar(today - timedelta(days=7), today + timedelta(days=7))
            report["checks"]["calendar"] = {"ok": True, "sessions": len(calendar)}
        except AlpacaError as exc:
            report["checks"]["calendar"] = {"ok": False, "error": str(exc)}

        try:
            start = datetime.now(timezone.utc) - timedelta(days=7)
            bars = self.get_bars(self.settings.watchlist[:2], start=start, limit=50)
            report["checks"]["market_data"] = {
                "ok": any(bars.values()),
                "bars": {sym: len(rows) for sym, rows in bars.items()},
                "feed": self.settings.data_feed,
            }
        except AlpacaError as exc:
            report["checks"]["market_data"] = {"ok": False, "error": str(exc)}

        try:
            start = datetime.now(timezone.utc) - timedelta(days=2)
            articles, truncated = self.get_news(start=start, max_articles=10)
            symbols = {s for a in articles for s in a["symbols"]}
            report["checks"]["news"] = {
                "ok": True,
                "articles": len(articles),
                "distinct_symbols": len(symbols),
                "scope": "market-wide (no symbol filter)",
                "more_pages_available": truncated,
            }
        except AlpacaError as exc:
            report["checks"]["news"] = {"ok": False, "error": str(exc)}

        try:
            equities = self.get_us_equities()
            report["checks"]["assets"] = {
                "ok": bool(equities),
                "us_equities": len(equities),
                "tradable": sum(1 for a in equities if a["tradable"]),
            }
        except AlpacaError as exc:
            report["checks"]["assets"] = {"ok": False, "error": str(exc)}

        report["ok"] = all(c.get("ok") for c in report["checks"].values())
        if not report["ok"] and any(
            "401" in str(c.get("error", "")) or "unauthorized" in str(c.get("error", "")).lower()
            for c in report["checks"].values()
        ):
            report["error"] = (
                "Alpaca rejected the credentials (401). Check that ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY are a valid *paper* key pair."
            )
        return report


def _ensure_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return value
