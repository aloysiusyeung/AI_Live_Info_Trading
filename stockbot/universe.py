"""The analysis universe.

News collection is market-wide and unconditional. *Analysis* cannot be: a
signal needs months of bar history plus a trained, validated model per symbol,
and there are roughly eleven thousand listed US equities. Training and
refreshing that many models on a ten-minute cadence is not feasible, and
pretending otherwise would mean shipping models that were never validated.

So the universe is built in two parts:

``core``
    The configured ``WATCHLIST``. Always analysed.

    The benchmark (``SPY`` by default) is deliberately *not* here. Its bars are
    always collected, because the relative-performance features need them, but
    being a benchmark is not a reason to emit a signal on it. Put it in
    ``WATCHLIST`` as well if you want it analysed.

``news``
    Symbols the market-wide news store has been talking about, ranked by recent
    article count, screened for tradability, and capped at
    ``MAX_DYNAMIC_SYMBOLS``.

Every admission and rejection is recorded in ``universe_snapshots`` with a
reason, so the dashboard can show exactly which symbols were covered and which
were dropped for want of capacity rather than leaving that invisible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .alpaca_client import AlpacaClient, AlpacaError
from .config import Settings
from .db import Database

logger = logging.getLogger(__name__)

#: Exchanges that count as the regular US equity market.
US_EQUITY_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "NYSEARCA", "ASCX"}


@dataclass
class UniverseEntry:
    symbol: str
    source: str                 # "core" | "news"
    news_count: int = 0
    rank: int | None = None
    admitted: bool = True
    reason: str = ""


@dataclass
class Universe:
    core: list[str] = field(default_factory=list)
    news_driven: list[str] = field(default_factory=list)
    rejected: list[UniverseEntry] = field(default_factory=list)
    entries: list[UniverseEntry] = field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        """Everything to analyse this cycle, core first, no duplicates."""
        seen: set[str] = set()
        ordered: list[str] = []
        for symbol in [*self.core, *self.news_driven]:
            if symbol not in seen:
                seen.add(symbol)
                ordered.append(symbol)
        return ordered

    def as_dict(self) -> dict[str, Any]:
        return {
            "core": list(self.core),
            "news_driven": list(self.news_driven),
            "total": len(self.symbols),
            "rejected": len(self.rejected),
            "rejected_examples": [
                {"symbol": e.symbol, "reason": e.reason, "news_count": e.news_count}
                for e in self.rejected[:10]
            ],
        }


class UniverseManager:
    """Resolves the tradable asset list and builds the analysis universe."""

    def __init__(self, settings: Settings, client: AlpacaClient, db: Database) -> None:
        self.settings = settings
        self.client = client
        self.db = db
        self._assets_refreshed_at: datetime | None = None

    # -- tradable asset list -----------------------------------------------
    def refresh_assets(self, force: bool = False) -> dict[str, Any]:
        """Pull Alpaca's US equity list into SQLite, at most twice a day."""
        stats = self.db.asset_count()
        if not force and stats["total"] and self._assets_fresh(stats["refreshed_at"]):
            return {"status": "current", **stats}

        try:
            assets = self.client.get_us_equities()
        except AlpacaError as exc:
            logger.warning("Asset refresh failed", extra={"error": str(exc)})
            self.db.log_error("universe", f"asset refresh failed: {exc}", severity="WARNING")
            return {"status": "ERROR", "error": str(exc), **stats}

        stored = self.db.upsert_assets(assets)
        self._assets_refreshed_at = datetime.now(timezone.utc)
        logger.info(
            "Assets refreshed",
            extra={"stored": stored, "tradable": sum(1 for a in assets if a["tradable"])},
        )
        return {"status": "OK", "stored": stored, **self.db.asset_count()}

    def _assets_fresh(self, refreshed_at: str | None) -> bool:
        if not refreshed_at:
            return False
        try:
            stamp = datetime.fromisoformat(refreshed_at)
        except ValueError:
            return False
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - stamp
        return age < timedelta(hours=self.settings.universe_refresh_hours)

    # -- universe construction ---------------------------------------------
    def build(self, now: datetime | None = None, persist: bool = True) -> Universe:
        now = now or datetime.now(timezone.utc)
        # WATCHLIST, not all_symbols: the benchmark is collected for features,
        # not analysed for its own sake.
        universe = Universe(core=list(self.settings.watchlist))
        for symbol in universe.core:
            universe.entries.append(
                UniverseEntry(symbol=symbol, source="core", reason="configured_watchlist")
            )

        if not self.settings.dynamic_universe_enabled or self.settings.max_dynamic_symbols < 1:
            if persist:
                self.db.save_universe_snapshot([e.__dict__ for e in universe.entries])
            return universe

        since = (now - timedelta(hours=self.settings.candidate_lookback_hours)).isoformat()
        counts = self.db.news_counts_by_symbol(since)
        if not counts:
            if persist:
                self.db.save_universe_snapshot([e.__dict__ for e in universe.entries])
            return universe

        tradable = self.db.tradable_symbols()
        assets_known = bool(tradable)
        core = set(universe.core)
        admitted = 0

        for rank, row in enumerate(counts, start=1):
            symbol = str(row["symbol"]).upper()
            count = int(row["news_count"] or 0)

            if symbol in core:
                continue        # already always-analysed
            if count < self.settings.min_news_for_candidate:
                continue        # below the noise floor; not worth recording

            entry = UniverseEntry(symbol=symbol, source="news", news_count=count, rank=rank)

            # Screen against the tradable asset list when we have one. Without
            # it, admit nothing new rather than guess that a news ticker is
            # tradable — a news mention is not evidence of that.
            if not assets_known:
                entry.admitted = False
                entry.reason = "asset_list_unavailable"
                universe.rejected.append(entry)
                universe.entries.append(entry)
                continue
            if symbol not in tradable:
                entry.admitted = False
                entry.reason = "not_tradable_on_alpaca"
                universe.rejected.append(entry)
                universe.entries.append(entry)
                continue
            if admitted >= self.settings.max_dynamic_symbols:
                entry.admitted = False
                entry.reason = "capacity_cap_reached"
                universe.rejected.append(entry)
                universe.entries.append(entry)
                continue

            entry.reason = "news_active_and_tradable"
            universe.news_driven.append(symbol)
            universe.entries.append(entry)
            admitted += 1

        if persist:
            self.db.save_universe_snapshot([e.__dict__ for e in universe.entries])

        logger.info(
            "Universe built",
            extra={
                "core": len(universe.core),
                "news_driven": len(universe.news_driven),
                "rejected": len(universe.rejected),
                "news_symbols_seen": len(counts),
            },
        )
        return universe

    # -- reporting ----------------------------------------------------------
    def coverage_report(self, now: datetime | None = None) -> dict[str, Any]:
        """How market-wide news coverage compares with analysis coverage."""
        now = now or datetime.now(timezone.utc)
        since = (now - timedelta(hours=self.settings.candidate_lookback_hours)).isoformat()
        counts = self.db.news_counts_by_symbol(since)
        tradable = self.db.tradable_symbols()
        news_symbols = {str(r["symbol"]).upper() for r in counts}
        universe = self.build(now=now, persist=False)

        return {
            "news_symbols_in_window": len(news_symbols),
            "news_symbols_tradable": len(news_symbols & tradable) if tradable else None,
            "analysed_symbols": len(universe.symbols),
            "core_symbols": len(universe.core),
            "news_driven_symbols": len(universe.news_driven),
            "not_analysed_for_capacity": sum(
                1 for e in universe.rejected if e.reason == "capacity_cap_reached"
            ),
            "not_analysed_not_tradable": sum(
                1 for e in universe.rejected if e.reason == "not_tradable_on_alpaca"
            ),
            "lookback_hours": self.settings.candidate_lookback_hours,
            "max_dynamic_symbols": self.settings.max_dynamic_symbols,
            "assets": self.db.asset_count(),
        }
