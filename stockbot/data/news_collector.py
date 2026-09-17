"""Market-wide news ingestion.

Coverage is deliberately **not** limited to the watchlist. Every article Alpaca
returns is stored, along with every symbol it mentions, so the news store
reflects the whole US market rather than a pre-chosen shortlist. The watchlist
only ever narrows what is *analysed*, never what is *collected*.

Ingestion is resumable: the watermark is the newest ``created_at`` already
stored, and each run resumes from slightly before it. Articles are keyed on
Alpaca's article id, so an overlapping window is idempotent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import pandas as pd

from ..alpaca_client import AlpacaClient, AlpacaError
from ..config import Settings
from ..db import Database
from ..news_features import headline_polarity

logger = logging.getLogger(__name__)

#: Re-fetch a little before the watermark; stories are sometimes published with
#: a slightly earlier timestamp than the moment they become queryable.
WATERMARK_OVERLAP = timedelta(minutes=10)


class NewsCollector:
    """Pulls the market-wide news feed into SQLite."""

    def __init__(self, settings: Settings, client: AlpacaClient, db: Database) -> None:
        self.settings = settings
        self.client = client
        self.db = db

    # -- validation --------------------------------------------------------
    @staticmethod
    def validate(articles: Iterable[dict], now: datetime | None = None) -> tuple[list[dict], dict]:
        """Drop unusable articles and report why.

        An article is rejected when it has no id, no headline, no timestamp, or
        a publication time in the future — the same conservative posture the bar
        validator takes.
        """
        now = now or datetime.now(timezone.utc)
        cutoff = now + timedelta(minutes=5)   # small allowance for clock skew
        kept: list[dict] = []
        report = {"seen": 0, "no_id": 0, "no_headline": 0, "no_timestamp": 0,
                  "future_dated": 0, "duplicate_id": 0, "kept": 0}
        seen_ids: set[int] = set()

        for article in articles:
            report["seen"] += 1
            if not article.get("id"):
                report["no_id"] += 1
                continue
            if not (article.get("headline") or "").strip():
                report["no_headline"] += 1
                continue
            created = article.get("created_at")
            if not isinstance(created, datetime):
                report["no_timestamp"] += 1
                continue
            if created > cutoff:
                report["future_dated"] += 1
                continue
            if article["id"] in seen_ids:
                report["duplicate_id"] += 1
                continue
            seen_ids.add(article["id"])
            kept.append(article)

        report["kept"] = len(kept)
        return kept, report

    # -- ingestion ---------------------------------------------------------
    def backfill(self, days: int | None = None) -> dict:
        """Market-wide backfill over a trailing window."""
        days = days or self.settings.news_backfill_days
        start = datetime.now(timezone.utc) - timedelta(days=days)
        return self._ingest(start, None, reason=f"backfill_{days}d")

    def update(self) -> dict:
        """Incremental market-wide ingestion from the stored watermark."""
        watermark = self.db.latest_news_created_at()
        if watermark:
            start = pd.Timestamp(watermark).to_pydatetime().astimezone(timezone.utc)
            start -= WATERMARK_OVERLAP
            # Never reach back further than the configured history window.
            floor = datetime.now(timezone.utc) - timedelta(days=self.settings.news_backfill_days)
            start = max(start, floor)
        else:
            start = datetime.now(timezone.utc) - timedelta(
                days=self.settings.news_backfill_days
            )
        return self._ingest(start, None, reason="incremental")

    def _ingest(self, start: datetime, end: datetime | None, reason: str) -> dict:
        if not self.settings.news_enabled:
            return {"status": "DISABLED", "articles_stored": 0, "reason": "NEWS_ENABLED=false"}

        run_id = self.db.start_news_run(
            start.isoformat(), end.isoformat() if end else None
        )
        max_articles = max(1, self.settings.news_page_limit * self.settings.news_max_pages)

        try:
            articles, truncated = self.client.get_news(
                start=start,
                end=end,
                symbols=None,            # market-wide: no symbol filter
                max_articles=max_articles,
                include_content=self.settings.news_include_content,
                exclude_contentless=self.settings.news_exclude_contentless,
            )
        except AlpacaError as exc:
            logger.error("News fetch failed", extra={"error": str(exc)})
            self.db.log_error("news", f"fetch failed: {exc}")
            self.db.finish_news_run(run_id, "ERROR", detail={"error": str(exc)})
            return {"status": "ERROR", "articles_stored": 0, "error": str(exc)}

        clean, validation = self.validate(articles)
        for article in clean:
            # A crude lexicon score, computed once at ingest. It is explicitly
            # not a sentiment model; see news_features.headline_polarity.
            article["polarity"] = headline_polarity(
                f"{article['headline']} {article.get('summary') or ''}"
            )

        stored = self.db.upsert_news(clean)
        symbols = {s for a in clean for s in a["symbols"]}

        self.db.finish_news_run(
            run_id,
            "OK",
            pages_fetched=1,
            articles_seen=validation["seen"],
            articles_stored=stored,
            symbols_seen=len(symbols),
            truncated=truncated,
            detail={"reason": reason, "validation": validation,
                    "window_start": start.isoformat()},
        )
        if truncated:
            # Honest about incomplete coverage rather than silently dropping news.
            logger.warning(
                "News window truncated at the article cap; increase NEWS_MAX_PAGES "
                "or ingest more often",
                extra={"cap": max_articles, "window_start": start.isoformat()},
            )
            self.db.log_error(
                "news",
                f"news window truncated at {max_articles} articles; coverage for this "
                "window is incomplete until the next run catches up",
                severity="WARNING",
                detail={"window_start": start.isoformat()},
            )

        logger.info(
            "News ingested",
            extra={
                "articles_stored": stored,
                "distinct_symbols": len(symbols),
                "truncated": truncated,
                "scope": "market-wide",
            },
        )
        return {
            "status": "OK",
            "articles_seen": validation["seen"],
            "articles_stored": stored,
            "distinct_symbols": len(symbols),
            "truncated": truncated,
            "validation": validation,
        }

    # -- reporting ---------------------------------------------------------
    def coverage(self) -> dict[str, Any]:
        """Coverage stats, including whether any window was truncated."""
        stats = self.db.news_coverage()
        runs = self.db.recent_news_runs(limit=20)
        stats["recent_runs"] = len(runs)
        stats["truncated_runs"] = sum(1 for r in runs if r.get("truncated"))
        stats["last_run_status"] = runs[0]["status"] if runs else None
        stats["scope"] = "market-wide (all US symbols Alpaca covers)"
        return stats
