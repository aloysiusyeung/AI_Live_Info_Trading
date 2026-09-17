"""News-derived features, computed strictly as-of each bar.

**Leakage is the whole problem here.** A news feature attached to the bar
starting at *t* may only use articles published at or before that bar's
**close** (``t + bar_minutes``), which is the earliest moment the bar could be
acted on. Two rules enforce that:

1. Every window is measured backwards from ``bar_end``, never from "now" and
   never from the end of the series.
2. Only ``created_at`` is used. ``updated_at`` is deliberately ignored — a story
   revised hours later would otherwise inject future information into a past
   bar.

The counts are exact; the polarity score is not a sentiment model. See
:func:`headline_polarity`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: News feature columns handed to the models.
NEWS_FEATURE_COLUMNS: list[str] = [
    "news_count_1h",
    "news_count_24h",
    "news_count_7d",
    "news_burst",
    "news_recency_minutes",
    "news_polarity_1h",
    "news_polarity_24h",
    "has_news_24h",
    "market_news_count_1h",
]

NEWS_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "news_count_1h": "articles in the last hour",
    "news_count_24h": "articles in the last 24 hours",
    "news_count_7d": "articles in the last 7 days",
    "news_burst": "news rate vs its 7-day baseline",
    "news_recency_minutes": "minutes since the last article",
    "news_polarity_1h": "headline tone in the last hour",
    "news_polarity_24h": "headline tone in the last 24 hours",
    "has_news_24h": "any news in the last 24 hours",
    "market_news_count_1h": "market-wide news intensity in the last hour",
}

#: Cap on "minutes since last article" when there is no prior article at all.
NO_NEWS_RECENCY_MINUTES = 10_080.0   # 7 days

# ---------------------------------------------------------------------------
# Crude keyword polarity
# ---------------------------------------------------------------------------
# This is a hand-written finance word list, NOT a sentiment model. It has no
# handling of negation, sarcasm, or context ("beats lowered expectations"), and
# it was not validated against labelled data. It is included because a coarse,
# transparent signal is preferable to an opaque one, and because the models can
# learn to ignore it. Do not read a polarity number as sentiment.
_POSITIVE = {
    "beat", "beats", "surge", "surges", "surged", "soar", "soars", "soared",
    "jump", "jumps", "jumped", "rally", "rallies", "rallied", "gain", "gains",
    "upgrade", "upgrades", "upgraded", "outperform", "raises", "raised",
    "record", "profit", "profits", "growth", "strong", "stronger", "approval",
    "approved", "wins", "win", "awarded", "expands", "expansion", "beat-and-raise",
    "buyback", "dividend", "beats-estimates", "top", "tops", "topped",
    "bullish", "breakthrough", "acquisition", "partnership", "launch", "launches",
}
_NEGATIVE = {
    "miss", "misses", "missed", "plunge", "plunges", "plunged", "slump", "slumps",
    "fall", "falls", "fell", "drop", "drops", "dropped", "sink", "sinks", "sank",
    "downgrade", "downgrades", "downgraded", "underperform", "cuts", "cut",
    "lowers", "lowered", "loss", "losses", "weak", "weaker", "warning", "warns",
    "recall", "recalls", "lawsuit", "sues", "sued", "investigation", "probe",
    "fraud", "bankruptcy", "bankrupt", "delisting", "halt", "halted", "layoffs",
    "layoff", "bearish", "decline", "declines", "declined", "shortfall", "delay",
    "delays", "delayed", "resign", "resigns", "resigned", "subpoena", "fine",
    "fined", "breach", "default",
}

_WORD_RE = re.compile(r"[a-z][a-z'-]+")


def headline_polarity(text: str | None) -> float:
    """Signed keyword score in [-1, 1] for a headline or summary.

    **This is not sentiment analysis.** It counts words from two hand-written
    lists and normalises the difference. There is no negation handling and no
    validation against labelled data. It exists as a transparent, cheap feature
    the models may or may not find useful.

    Returns 0.0 for empty text or text containing none of the listed words,
    which is why :data:`NEWS_FEATURE_COLUMNS` also carries ``has_news_24h`` —
    without it, "neutral" and "no news" would be indistinguishable.
    """
    if not text:
        return 0.0
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    positive = sum(1 for w in words if w in _POSITIVE)
    negative = sum(1 for w in words if w in _NEGATIVE)
    total = positive + negative
    if total == 0:
        return 0.0
    return float((positive - negative) / total)


# ---------------------------------------------------------------------------
# As-of index
# ---------------------------------------------------------------------------
@dataclass
class _SymbolSeries:
    times: np.ndarray            # int64 nanoseconds, ascending
    polarity_cumsum: np.ndarray  # cumulative polarity, prefixed with 0


@dataclass
class NewsIndex:
    """Per-symbol article timestamps, arranged for fast as-of lookups.

    Built once per cycle from the whole news store, then queried per symbol.
    ``available`` is False for an empty store, which makes the feature builder
    emit NaN so the news columns get dropped rather than silently reading as
    "no news anywhere".
    """

    by_symbol: dict[str, _SymbolSeries] = field(default_factory=dict)
    market_times: np.ndarray = field(default_factory=lambda: np.array([], dtype="int64"))
    available: bool = False

    @classmethod
    def from_rows(cls, rows: Iterable[dict]) -> "NewsIndex":
        """Build from ``(symbol, created_at, polarity)`` rows."""
        frame = pd.DataFrame(list(rows))
        if frame.empty:
            return cls()

        frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, format="mixed")
        if "polarity" not in frame.columns:
            frame["polarity"] = 0.0
        frame["polarity"] = frame["polarity"].astype(float).fillna(0.0)
        frame = frame.sort_values("created_at")

        by_symbol: dict[str, _SymbolSeries] = {}
        for symbol, group in frame.groupby("symbol", sort=False):
            times = group["created_at"].to_numpy(dtype="datetime64[ns]").astype("int64")
            polarity = group["polarity"].to_numpy(dtype=float)
            by_symbol[str(symbol).upper()] = _SymbolSeries(
                times=times,
                polarity_cumsum=np.concatenate([[0.0], np.cumsum(polarity)]),
            )

        # Market-wide intensity uses distinct articles, not symbol links, so a
        # story tagging 40 tickers counts once.
        if "article_id" in frame.columns:
            distinct = frame.drop_duplicates(subset=["article_id"])
        else:
            distinct = frame
        market_times = (
            distinct.sort_values("created_at")["created_at"]
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64")
        )
        return cls(by_symbol=by_symbol, market_times=market_times, available=True)

    @classmethod
    def from_db(cls, db, since: datetime | None = None) -> "NewsIndex":
        rows = db.query(
            """SELECT s.symbol, s.created_at, s.article_id, a.polarity
               FROM news_article_symbols s
               JOIN news_articles a ON a.id = s.article_id
               """ + ("WHERE s.created_at >= ? " if since else "") +
            "ORDER BY s.created_at ASC",
            (since.isoformat(),) if since else (),
        )
        return cls.from_rows(rows)

    def symbols(self) -> set[str]:
        return set(self.by_symbol)


def _window_counts(times: np.ndarray, cutoffs: np.ndarray, window_ns: int) -> np.ndarray:
    """Number of entries in ``(cutoff - window, cutoff]`` for each cutoff."""
    upper = np.searchsorted(times, cutoffs, side="right")
    lower = np.searchsorted(times, cutoffs - window_ns, side="right")
    return (upper - lower).astype(float)


def build_news_features(
    bar_starts: pd.Series,
    symbol: str,
    index: NewsIndex | None,
    bar_minutes: int,
) -> pd.DataFrame:
    """News features for one symbol, aligned to ``bar_starts``.

    Each row's windows end at that bar's **close**, so no row can see an article
    published after the bar it is attached to.
    """
    n = len(bar_starts)
    empty = pd.DataFrame(
        {col: np.full(n, np.nan) for col in NEWS_FEATURE_COLUMNS},
        index=bar_starts.index,
    )
    if index is None or not index.available or n == 0:
        return empty

    starts = pd.to_datetime(bar_starts, utc=True)
    # The decision point is the bar's close, not its open.
    cutoffs = (
        (starts + pd.Timedelta(minutes=bar_minutes))
        .to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )

    hour_ns = 3_600_000_000_000
    day_ns = 24 * hour_ns
    week_ns = 7 * day_ns

    series = index.by_symbol.get(symbol.upper())
    out = pd.DataFrame(index=bar_starts.index)

    if series is None or len(series.times) == 0:
        # The store has news, this symbol simply has none. Zero is informative
        # and must not be NaN, or these rows would be dropped from training.
        out["news_count_1h"] = 0.0
        out["news_count_24h"] = 0.0
        out["news_count_7d"] = 0.0
        out["news_burst"] = 0.0
        out["news_recency_minutes"] = NO_NEWS_RECENCY_MINUTES
        out["news_polarity_1h"] = 0.0
        out["news_polarity_24h"] = 0.0
        out["has_news_24h"] = 0.0
    else:
        times = series.times
        count_1h = _window_counts(times, cutoffs, hour_ns)
        count_24h = _window_counts(times, cutoffs, day_ns)
        count_7d = _window_counts(times, cutoffs, week_ns)

        out["news_count_1h"] = count_1h
        out["news_count_24h"] = count_24h
        out["news_count_7d"] = count_7d

        # Burst: this hour's rate against the trailing 7-day hourly baseline.
        # Divide only where the baseline is positive; a symbol with no weekly
        # history has no baseline to be surprising against, so the burst is 0.
        baseline = count_7d / 168.0
        out["news_burst"] = np.divide(
            count_1h, baseline, out=np.zeros_like(count_1h, dtype=float),
            where=baseline > 0,
        )

        upper = np.searchsorted(times, cutoffs, side="right")
        last_time = np.where(upper > 0, times[np.clip(upper - 1, 0, None)], np.nan)
        recency = (cutoffs - last_time) / 6e10   # ns -> minutes
        out["news_recency_minutes"] = np.where(
            upper > 0, np.minimum(recency, NO_NEWS_RECENCY_MINUTES), NO_NEWS_RECENCY_MINUTES
        )

        cumsum = series.polarity_cumsum
        for label, window_ns in (("1h", hour_ns), ("24h", day_ns)):
            lower = np.searchsorted(times, cutoffs - window_ns, side="right")
            total = cumsum[upper] - cumsum[lower]
            count = (upper - lower).astype(float)
            out[f"news_polarity_{label}"] = np.where(count > 0, total / np.maximum(count, 1), 0.0)

        out["has_news_24h"] = (count_24h > 0).astype(float)

    # Market-wide news intensity: a regime feature shared by every symbol.
    if len(index.market_times):
        out["market_news_count_1h"] = _window_counts(index.market_times, cutoffs, hour_ns)
    else:
        out["market_news_count_1h"] = 0.0

    return out[NEWS_FEATURE_COLUMNS]


def news_context(db, symbol: str, limit: int = 5, now: datetime | None = None) -> dict:
    """Recent headlines for a symbol, for the dashboard and explanations."""
    articles = db.news_for_symbol(symbol, limit=limit)
    now = now or datetime.now(timezone.utc)
    recent_24h = 0
    for article in articles:
        try:
            created = pd.Timestamp(article["created_at"]).to_pydatetime()
        except (ValueError, TypeError):
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if now - created <= timedelta(hours=24):
            recent_24h += 1
    return {
        "symbol": symbol,
        "articles": articles,
        "count_24h": recent_24h,
        "latest_headline": articles[0]["headline"] if articles else None,
        "latest_at": articles[0]["created_at"] if articles else None,
    }
