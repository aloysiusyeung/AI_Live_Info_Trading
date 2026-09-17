"""Bar collection: fetch from Alpaca, validate, persist, serve to the model."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from ..alpaca_client import AlpacaClient, AlpacaError
from ..config import Settings
from ..db import Database
from .validation import BarValidator, ValidationReport

logger = logging.getLogger(__name__)

BAR_COLUMNS = [
    "symbol", "bar_start", "bar_minutes", "open", "high", "low", "close",
    "volume", "trade_count", "vwap",
]


class BarCollector:
    """Incrementally maintains the local bar store for the watchlist."""

    def __init__(self, settings: Settings, client: AlpacaClient, db: Database) -> None:
        self.settings = settings
        self.client = client
        self.db = db
        self.validator = BarValidator(
            bar_minutes=settings.bar_minutes,
            max_age_seconds=settings.max_data_age_seconds,
            min_rows=max(60, settings.prediction_horizon_bars + 40),
            settle_seconds=settings.bar_settle_seconds,
        )

    # -- ingestion -------------------------------------------------------
    def backfill(self, symbols: Iterable[str] | None = None, days: int | None = None) -> dict[str, int]:
        """Fetch full history for the requested symbols and store it."""
        symbols = list(symbols or self.settings.all_symbols)
        days = days or self.settings.history_days
        start = datetime.now(timezone.utc) - timedelta(days=days)
        return self._fetch_and_store(symbols, start)

    def update(self, symbols: Iterable[str] | None = None) -> dict[str, int]:
        """Fetch only what is missing since the last stored bar per symbol."""
        symbols = list(symbols or self.settings.all_symbols)
        stored: dict[str, int] = {}
        for symbol in symbols:
            latest = self.db.latest_bar_start(symbol, self.settings.bar_minutes)
            if latest:
                start = pd.Timestamp(latest).to_pydatetime()
                start = start.astimezone(timezone.utc) + timedelta(minutes=1)
                # Guard against a very old store: never ask for more than the
                # configured history window in one go.
                floor = datetime.now(timezone.utc) - timedelta(days=self.settings.history_days)
                start = max(start, floor)
            else:
                start = datetime.now(timezone.utc) - timedelta(days=self.settings.history_days)
            stored.update(self._fetch_and_store([symbol], start))
        return stored

    def _fetch_and_store(self, symbols: list[str], start: datetime) -> dict[str, int]:
        counts: dict[str, int] = {}
        try:
            bars = self.client.get_bars(symbols, start=start)
        except AlpacaError as exc:
            logger.error("Bar fetch failed", extra={"symbols": symbols, "error": str(exc)})
            self.db.log_error("collector", str(exc), symbol=",".join(symbols))
            return {symbol: 0 for symbol in symbols}

        for symbol, rows in bars.items():
            if not rows:
                counts[symbol] = 0
                continue
            # Only persist bars whose window has fully closed.
            complete = [r for r in rows if self.validator.bar_is_complete(r["bar_start"])]
            counts[symbol] = self.db.upsert_bars(complete)
            logger.info(
                "Stored bars",
                extra={"symbol": symbol, "stored": counts[symbol], "fetched": len(rows)},
            )
        return counts

    # -- retrieval -------------------------------------------------------
    def load_frame(self, symbol: str, limit: int | None = None) -> pd.DataFrame:
        """Bars for one symbol as a tidy DataFrame, oldest first."""
        rows = self.db.get_bars(symbol, self.settings.bar_minutes, limit=limit)
        if not rows:
            return pd.DataFrame(columns=BAR_COLUMNS)
        df = pd.DataFrame(rows)
        df["bar_start"] = pd.to_datetime(df["bar_start"], utc=True)
        keep = [c for c in BAR_COLUMNS if c in df.columns]
        return df[keep].sort_values("bar_start").reset_index(drop=True)

    def load_validated(
        self, symbol: str, limit: int | None = None, now: datetime | None = None
    ) -> tuple[pd.DataFrame, ValidationReport]:
        frame = self.load_frame(symbol, limit=limit)
        return self.validator.validate(symbol, frame, now=now)

    def load_many_validated(
        self, symbols: Iterable[str] | None = None, now: datetime | None = None
    ) -> tuple[dict[str, pd.DataFrame], dict[str, ValidationReport]]:
        frames: dict[str, pd.DataFrame] = {}
        reports: dict[str, ValidationReport] = {}
        for symbol in list(symbols or self.settings.all_symbols):
            frame, report = self.load_validated(symbol, now=now)
            frames[symbol] = frame
            reports[symbol] = report
            if not report.ok:
                logger.warning(
                    "Data validation failed",
                    extra={"symbol": symbol, "reasons": report.reasons},
                )
                self.db.log_error(
                    "validation",
                    f"validation failed: {', '.join(report.reasons)}",
                    severity="WARNING",
                    symbol=symbol,
                    detail=report.as_dict(),
                )
        return frames, reports
