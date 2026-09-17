"""SQLite persistence layer.

One file, WAL mode, plain SQL. The dashboard opens the same file read-only.
Tables: bars, features, predictions, signals, model_versions, backtest_runs,
paper_orders, positions, errors, scheduler_runs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

logger = logging.getLogger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS bars (
    symbol         TEXT NOT NULL,
    bar_start      TEXT NOT NULL,          -- ISO8601 UTC, bar open time
    bar_minutes    INTEGER NOT NULL,
    open           REAL NOT NULL,
    high           REAL NOT NULL,
    low            REAL NOT NULL,
    close          REAL NOT NULL,
    volume         REAL NOT NULL,
    trade_count    REAL,
    vwap           REAL,
    feed           TEXT,
    ingested_at    TEXT NOT NULL,
    PRIMARY KEY (symbol, bar_start, bar_minutes)
);
CREATE INDEX IF NOT EXISTS idx_bars_symbol_time ON bars(symbol, bar_start);

CREATE TABLE IF NOT EXISTS features (
    symbol         TEXT NOT NULL,
    bar_start      TEXT NOT NULL,
    computed_at    TEXT NOT NULL,
    payload        TEXT NOT NULL,          -- JSON dict of feature -> value
    PRIMARY KEY (symbol, bar_start)
);

CREATE TABLE IF NOT EXISTS model_versions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    model_name          TEXT NOT NULL,
    horizon_bars        INTEGER NOT NULL,
    feature_names       TEXT NOT NULL,     -- JSON list
    train_rows          INTEGER NOT NULL,
    oos_metrics         TEXT NOT NULL,     -- JSON dict
    baseline_metrics    TEXT NOT NULL,     -- JSON dict
    selected            INTEGER NOT NULL DEFAULT 0,
    artifact_path       TEXT,
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_model_symbol ON model_versions(symbol, created_at);

CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    bar_start           TEXT NOT NULL,     -- bar the prediction was computed on
    model_version_id    INTEGER,
    model_name          TEXT,
    horizon_bars        INTEGER NOT NULL,
    prob_up             REAL,
    expected_return     REAL,
    uncertainty         REAL,
    top_features        TEXT,              -- JSON list of [name, contribution]
    UNIQUE (symbol, bar_start, model_version_id)
);
CREATE INDEX IF NOT EXISTS idx_pred_symbol_time ON predictions(symbol, created_at);

CREATE TABLE IF NOT EXISTS signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    bar_start           TEXT NOT NULL,
    prediction_id       INTEGER,
    signal              TEXT NOT NULL,     -- BUY | HOLD | AVOID | INSUFFICIENT_EVIDENCE
    confidence          REAL,
    expected_return     REAL,
    uncertainty         REAL,
    last_price          REAL,
    explanation         TEXT,
    risk_decision       TEXT,              -- ALLOW | BLOCK | NO_ACTION
    risk_reasons        TEXT,              -- JSON list
    target_qty          REAL,
    target_notional     REAL,
    UNIQUE (symbol, bar_start)
);
CREATE INDEX IF NOT EXISTS idx_signal_symbol_time ON signals(symbol, created_at);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    model_name          TEXT NOT NULL,
    horizon_bars        INTEGER NOT NULL,
    start_date          TEXT,
    end_date            TEXT,
    n_folds             INTEGER,
    n_test_rows         INTEGER,
    metrics             TEXT NOT NULL,     -- JSON dict (net of costs)
    baselines           TEXT NOT NULL,     -- JSON dict
    cost_bps            REAL,
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS paper_orders (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT,
    client_order_id     TEXT NOT NULL UNIQUE,
    broker_order_id     TEXT,
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL,
    qty                 REAL,
    notional            REAL,
    order_type          TEXT,
    time_in_force       TEXT,
    status              TEXT NOT NULL,
    filled_qty          REAL DEFAULT 0,
    filled_avg_price    REAL,
    signal_id           INTEGER,
    prediction_id       INTEGER,
    risk_snapshot       TEXT,              -- JSON dict of the risk decision
    prediction_snapshot TEXT,              -- JSON dict of the prediction
    error               TEXT,
    paper               INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON paper_orders(symbol, created_at);

CREATE TABLE IF NOT EXISTS positions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at         TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    qty                 REAL NOT NULL,
    avg_entry_price     REAL,
    market_value        REAL,
    cost_basis          REAL,
    unrealized_pl       REAL,
    unrealized_plpc     REAL,
    current_price       REAL
);
CREATE INDEX IF NOT EXISTS idx_positions_time ON positions(snapshot_at);

CREATE TABLE IF NOT EXISTS account_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at         TEXT NOT NULL,
    equity              REAL,
    last_equity         REAL,
    cash                REAL,
    buying_power        REAL,
    long_market_value   REAL,
    daytrade_count      INTEGER,
    account_blocked     INTEGER,
    trading_blocked     INTEGER
);

CREATE TABLE IF NOT EXISTS errors (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at          TEXT NOT NULL,
    component           TEXT NOT NULL,
    symbol              TEXT,
    severity            TEXT NOT NULL,
    message             TEXT NOT NULL,
    detail              TEXT
);
CREATE INDEX IF NOT EXISTS idx_errors_time ON errors(created_at);

CREATE TABLE IF NOT EXISTS scheduler_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL,     -- RUNNING | OK | SKIPPED | ERROR
    reason              TEXT,
    symbols_processed   INTEGER DEFAULT 0,
    signals_generated   INTEGER DEFAULT 0,
    orders_submitted    INTEGER DEFAULT 0,
    next_run_at         TEXT,
    detail              TEXT
);
CREATE INDEX IF NOT EXISTS idx_sched_time ON scheduler_runs(started_at);

-- News is stored market-wide: every article Alpaca returns is kept, whatever
-- symbols it mentions. There is deliberately no watchlist filter here.
CREATE TABLE IF NOT EXISTS news_articles (
    id                  INTEGER PRIMARY KEY,        -- Alpaca/Benzinga article id
    created_at          TEXT NOT NULL,              -- when the story was published
    updated_at          TEXT,
    headline            TEXT NOT NULL,
    summary             TEXT,
    author              TEXT,
    source              TEXT,
    url                 TEXT,
    content             TEXT,
    symbol_count        INTEGER NOT NULL DEFAULT 0,
    polarity            REAL,                       -- crude lexicon score, see news_features
    ingested_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_news_created ON news_articles(created_at);

-- Many-to-many: one article can tag many symbols, and most do.
CREATE TABLE IF NOT EXISTS news_article_symbols (
    article_id          INTEGER NOT NULL,
    symbol              TEXT NOT NULL,
    created_at          TEXT NOT NULL,              -- denormalised for fast as-of queries
    PRIMARY KEY (article_id, symbol),
    FOREIGN KEY (article_id) REFERENCES news_articles(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_news_symbol_time ON news_article_symbols(symbol, created_at);

CREATE TABLE IF NOT EXISTS news_ingest_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT NOT NULL,
    window_start        TEXT,
    window_end          TEXT,
    pages_fetched       INTEGER DEFAULT 0,
    articles_seen       INTEGER DEFAULT 0,
    articles_stored     INTEGER DEFAULT 0,
    symbols_seen        INTEGER DEFAULT 0,
    truncated           INTEGER DEFAULT 0,          -- 1 when the page cap was hit
    detail              TEXT
);

-- The tradable US equity universe, refreshed from Alpaca's assets endpoint.
CREATE TABLE IF NOT EXISTS assets (
    symbol              TEXT PRIMARY KEY,
    name                TEXT,
    exchange            TEXT,
    asset_class         TEXT,
    status              TEXT,
    tradable            INTEGER NOT NULL DEFAULT 0,
    shortable           INTEGER NOT NULL DEFAULT 0,
    fractionable        INTEGER NOT NULL DEFAULT 0,
    refreshed_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assets_tradable ON assets(tradable, exchange);

-- Snapshot of which symbols the analysis universe covered, and why.
CREATE TABLE IF NOT EXISTS universe_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_at         TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    source              TEXT NOT NULL,              -- core | news
    news_count          INTEGER DEFAULT 0,
    rank                INTEGER,
    admitted            INTEGER NOT NULL DEFAULT 1,
    reason              TEXT
);
CREATE INDEX IF NOT EXISTS idx_universe_time ON universe_snapshots(snapshot_at);

CREATE TABLE IF NOT EXISTS app_state (
    key                 TEXT PRIMARY KEY,
    value               TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


class Database:
    """Thin SQLite wrapper. Thread-safe via a lock around writes."""

    def __init__(self, path: str, read_only: bool = False) -> None:
        self.path = str(path)
        self.read_only = read_only
        self._lock = threading.RLock()
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        if not read_only:
            with self.connect() as conn:
                conn.executescript(SCHEMA)

    # -- connections -----------------------------------------------------
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)
        else:
            conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            if not self.read_only:
                conn.commit()
        finally:
            conn.close()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self._lock, self.connect() as conn:
            conn.execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> int:
        rows = list(rows)
        if not rows:
            return 0
        with self._lock, self.connect() as conn:
            conn.executemany(sql, rows)
        return len(rows)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(sql, params).fetchall()]

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def insert(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self._lock, self.connect() as conn:
            cursor = conn.execute(sql, params)
            return int(cursor.lastrowid or 0)

    # -- bars ------------------------------------------------------------
    def upsert_bars(self, rows: Iterable[dict]) -> int:
        """Insert bars idempotently. Duplicate (symbol, bar_start) is replaced."""
        payload = [
            (
                r["symbol"], _to_iso(r["bar_start"]), int(r["bar_minutes"]),
                float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]),
                float(r["volume"]),
                None if r.get("trade_count") is None else float(r["trade_count"]),
                None if r.get("vwap") is None else float(r["vwap"]),
                r.get("feed"), utc_now_iso(),
            )
            for r in rows
        ]
        return self.executemany(
            """INSERT INTO bars
               (symbol, bar_start, bar_minutes, open, high, low, close, volume,
                trade_count, vwap, feed, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol, bar_start, bar_minutes) DO UPDATE SET
                 open=excluded.open, high=excluded.high, low=excluded.low,
                 close=excluded.close, volume=excluded.volume,
                 trade_count=excluded.trade_count, vwap=excluded.vwap,
                 feed=excluded.feed, ingested_at=excluded.ingested_at""",
            payload,
        )

    def get_bars(self, symbol: str, bar_minutes: int, limit: int | None = None) -> list[dict]:
        sql = ("SELECT * FROM bars WHERE symbol=? AND bar_minutes=? "
               "ORDER BY bar_start ASC")
        rows = self.query(sql, (symbol, bar_minutes))
        return rows[-limit:] if limit else rows

    def latest_bar_start(self, symbol: str, bar_minutes: int) -> str | None:
        row = self.query_one(
            "SELECT MAX(bar_start) AS m FROM bars WHERE symbol=? AND bar_minutes=?",
            (symbol, bar_minutes),
        )
        return row["m"] if row and row["m"] else None

    # -- features --------------------------------------------------------
    def save_features(self, symbol: str, bar_start: Any, payload: dict) -> None:
        self.execute(
            """INSERT INTO features (symbol, bar_start, computed_at, payload)
               VALUES (?,?,?,?)
               ON CONFLICT(symbol, bar_start) DO UPDATE SET
                 computed_at=excluded.computed_at, payload=excluded.payload""",
            (symbol, _to_iso(bar_start), utc_now_iso(), json.dumps(payload, default=float)),
        )

    # -- predictions / signals -------------------------------------------
    def save_prediction(self, **kw: Any) -> int:
        return self.insert(
            """INSERT INTO predictions
               (created_at, symbol, bar_start, model_version_id, model_name,
                horizon_bars, prob_up, expected_return, uncertainty, top_features)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol, bar_start, model_version_id) DO UPDATE SET
                 created_at=excluded.created_at, prob_up=excluded.prob_up,
                 expected_return=excluded.expected_return,
                 uncertainty=excluded.uncertainty,
                 top_features=excluded.top_features""",
            (
                utc_now_iso(), kw["symbol"], _to_iso(kw["bar_start"]),
                kw.get("model_version_id"), kw.get("model_name"),
                int(kw["horizon_bars"]), kw.get("prob_up"), kw.get("expected_return"),
                kw.get("uncertainty"), json.dumps(kw.get("top_features") or [], default=float),
            ),
        )

    def save_signal(self, **kw: Any) -> int:
        return self.insert(
            """INSERT INTO signals
               (created_at, symbol, bar_start, prediction_id, signal, confidence,
                expected_return, uncertainty, last_price, explanation,
                risk_decision, risk_reasons, target_qty, target_notional)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol, bar_start) DO UPDATE SET
                 created_at=excluded.created_at, signal=excluded.signal,
                 confidence=excluded.confidence,
                 expected_return=excluded.expected_return,
                 uncertainty=excluded.uncertainty, last_price=excluded.last_price,
                 explanation=excluded.explanation,
                 risk_decision=excluded.risk_decision,
                 risk_reasons=excluded.risk_reasons,
                 target_qty=excluded.target_qty,
                 target_notional=excluded.target_notional""",
            (
                utc_now_iso(), kw["symbol"], _to_iso(kw["bar_start"]),
                kw.get("prediction_id"), kw["signal"], kw.get("confidence"),
                kw.get("expected_return"), kw.get("uncertainty"), kw.get("last_price"),
                kw.get("explanation"), kw.get("risk_decision"),
                json.dumps(kw.get("risk_reasons") or [], default=str),
                kw.get("target_qty"), kw.get("target_notional"),
            ),
        )

    def latest_signals(self, limit: int = 50) -> list[dict]:
        return self.query(
            """SELECT s.* FROM signals s
               JOIN (SELECT symbol, MAX(created_at) AS mx FROM signals GROUP BY symbol) t
                 ON s.symbol = t.symbol AND s.created_at = t.mx
               ORDER BY s.symbol LIMIT ?""",
            (limit,),
        )

    # -- models ----------------------------------------------------------
    def save_model_version(self, **kw: Any) -> int:
        return self.insert(
            """INSERT INTO model_versions
               (created_at, symbol, model_name, horizon_bars, feature_names,
                train_rows, oos_metrics, baseline_metrics, selected, artifact_path, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                utc_now_iso(), kw["symbol"], kw["model_name"], int(kw["horizon_bars"]),
                json.dumps(kw["feature_names"]), int(kw["train_rows"]),
                json.dumps(kw["oos_metrics"], default=float),
                json.dumps(kw.get("baseline_metrics") or {}, default=float),
                1 if kw.get("selected") else 0, kw.get("artifact_path"), kw.get("notes"),
            ),
        )

    def mark_selected_model(self, symbol: str, model_version_id: int) -> None:
        with self._lock, self.connect() as conn:
            conn.execute("UPDATE model_versions SET selected=0 WHERE symbol=?", (symbol,))
            conn.execute("UPDATE model_versions SET selected=1 WHERE id=?", (model_version_id,))

    def selected_model(self, symbol: str) -> dict | None:
        return self.query_one(
            "SELECT * FROM model_versions WHERE symbol=? AND selected=1 "
            "ORDER BY created_at DESC LIMIT 1",
            (symbol,),
        )

    def save_backtest_run(self, **kw: Any) -> int:
        return self.insert(
            """INSERT INTO backtest_runs
               (created_at, symbol, model_name, horizon_bars, start_date, end_date,
                n_folds, n_test_rows, metrics, baselines, cost_bps, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                utc_now_iso(), kw["symbol"], kw["model_name"], int(kw["horizon_bars"]),
                kw.get("start_date"), kw.get("end_date"), kw.get("n_folds"),
                kw.get("n_test_rows"), json.dumps(kw["metrics"], default=float),
                json.dumps(kw.get("baselines") or {}, default=float),
                kw.get("cost_bps"), kw.get("notes"),
            ),
        )

    # -- orders / positions ----------------------------------------------
    def record_order(self, **kw: Any) -> int:
        return self.insert(
            """INSERT INTO paper_orders
               (created_at, updated_at, client_order_id, broker_order_id, symbol,
                side, qty, notional, order_type, time_in_force, status, filled_qty,
                filled_avg_price, signal_id, prediction_id, risk_snapshot,
                prediction_snapshot, error, paper)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
            (
                utc_now_iso(), utc_now_iso(), kw["client_order_id"],
                kw.get("broker_order_id"), kw["symbol"], kw["side"], kw.get("qty"),
                kw.get("notional"), kw.get("order_type", "market"),
                kw.get("time_in_force", "day"), kw["status"], kw.get("filled_qty", 0.0),
                kw.get("filled_avg_price"), kw.get("signal_id"), kw.get("prediction_id"),
                json.dumps(kw.get("risk_snapshot") or {}, default=str),
                json.dumps(kw.get("prediction_snapshot") or {}, default=str),
                kw.get("error"),
            ),
        )

    def update_order_status(self, client_order_id: str, **kw: Any) -> None:
        self.execute(
            """UPDATE paper_orders SET updated_at=?, status=COALESCE(?, status),
                   broker_order_id=COALESCE(?, broker_order_id),
                   filled_qty=COALESCE(?, filled_qty),
                   filled_avg_price=COALESCE(?, filled_avg_price),
                   error=COALESCE(?, error)
               WHERE client_order_id=?""",
            (
                utc_now_iso(), kw.get("status"), kw.get("broker_order_id"),
                kw.get("filled_qty"), kw.get("filled_avg_price"), kw.get("error"),
                client_order_id,
            ),
        )

    def recent_orders(self, symbol: str | None = None, limit: int = 100) -> list[dict]:
        if symbol:
            return self.query(
                "SELECT * FROM paper_orders WHERE symbol=? ORDER BY created_at DESC LIMIT ?",
                (symbol, limit),
            )
        return self.query(
            "SELECT * FROM paper_orders ORDER BY created_at DESC LIMIT ?", (limit,)
        )

    def orders_since(self, symbol: str, since_iso: str) -> list[dict]:
        return self.query(
            "SELECT * FROM paper_orders WHERE symbol=? AND created_at >= ? "
            "ORDER BY created_at DESC",
            (symbol, since_iso),
        )

    def snapshot_positions(self, positions: Iterable[dict]) -> int:
        stamp = utc_now_iso()
        return self.executemany(
            """INSERT INTO positions
               (snapshot_at, symbol, qty, avg_entry_price, market_value, cost_basis,
                unrealized_pl, unrealized_plpc, current_price)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            [
                (
                    stamp, p["symbol"], p["qty"], p.get("avg_entry_price"),
                    p.get("market_value"), p.get("cost_basis"), p.get("unrealized_pl"),
                    p.get("unrealized_plpc"), p.get("current_price"),
                )
                for p in positions
            ],
        )

    def snapshot_account(self, account: dict) -> int:
        return self.insert(
            """INSERT INTO account_snapshots
               (snapshot_at, equity, last_equity, cash, buying_power,
                long_market_value, daytrade_count, account_blocked, trading_blocked)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                utc_now_iso(), account.get("equity"), account.get("last_equity"),
                account.get("cash"), account.get("buying_power"),
                account.get("long_market_value"), account.get("daytrade_count"),
                1 if account.get("account_blocked") else 0,
                1 if account.get("trading_blocked") else 0,
            ),
        )

    # -- errors / scheduler ----------------------------------------------
    def log_error(self, component: str, message: str, severity: str = "ERROR",
                  symbol: str | None = None, detail: Any = None) -> int:
        return self.insert(
            """INSERT INTO errors (created_at, component, symbol, severity, message, detail)
               VALUES (?,?,?,?,?,?)""",
            (utc_now_iso(), component, symbol, severity, message,
             json.dumps(detail, default=str) if detail is not None else None),
        )

    def recent_errors(self, limit: int = 50) -> list[dict]:
        return self.query("SELECT * FROM errors ORDER BY created_at DESC LIMIT ?", (limit,))

    def start_scheduler_run(self, reason: str | None = None) -> int:
        return self.insert(
            "INSERT INTO scheduler_runs (started_at, status, reason) VALUES (?,?,?)",
            (utc_now_iso(), "RUNNING", reason),
        )

    def finish_scheduler_run(self, run_id: int, status: str, **kw: Any) -> None:
        self.execute(
            """UPDATE scheduler_runs SET finished_at=?, status=?, reason=COALESCE(?, reason),
                   symbols_processed=?, signals_generated=?, orders_submitted=?,
                   next_run_at=?, detail=?
               WHERE id=?""",
            (
                utc_now_iso(), status, kw.get("reason"), kw.get("symbols_processed", 0),
                kw.get("signals_generated", 0), kw.get("orders_submitted", 0),
                kw.get("next_run_at"),
                json.dumps(kw.get("detail"), default=str) if kw.get("detail") else None,
                run_id,
            ),
        )

    def recent_scheduler_runs(self, limit: int = 20) -> list[dict]:
        return self.query(
            "SELECT * FROM scheduler_runs ORDER BY started_at DESC LIMIT ?", (limit,)
        )

    # -- news (market-wide) ----------------------------------------------
    def upsert_news(self, articles: Iterable[dict]) -> int:
        """Store articles idempotently, keyed by Alpaca's article id.

        Re-ingesting an overlapping window is therefore free, and a revised
        story replaces its earlier copy rather than duplicating it.
        """
        articles = list(articles)
        if not articles:
            return 0
        stamp = utc_now_iso()
        rows = [
            (
                int(a["id"]), _to_iso(a["created_at"]),
                _to_iso(a["updated_at"]) if a.get("updated_at") else None,
                a["headline"], a.get("summary"), a.get("author"), a.get("source"),
                a.get("url"), a.get("content"), len(a.get("symbols") or []),
                a.get("polarity"), stamp,
            )
            for a in articles
        ]
        self.executemany(
            """INSERT INTO news_articles
               (id, created_at, updated_at, headline, summary, author, source, url,
                content, symbol_count, polarity, ingested_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 updated_at=excluded.updated_at, headline=excluded.headline,
                 summary=excluded.summary, source=excluded.source,
                 content=excluded.content, symbol_count=excluded.symbol_count,
                 polarity=excluded.polarity, ingested_at=excluded.ingested_at""",
            rows,
        )
        links = [
            (int(a["id"]), symbol.upper(), _to_iso(a["created_at"]))
            for a in articles
            for symbol in (a.get("symbols") or [])
        ]
        self.executemany(
            """INSERT INTO news_article_symbols (article_id, symbol, created_at)
               VALUES (?,?,?)
               ON CONFLICT(article_id, symbol) DO UPDATE SET
                 created_at=excluded.created_at""",
            links,
        )
        return len(rows)

    def news_for_symbol(self, symbol: str, limit: int = 50) -> list[dict]:
        return self.query(
            """SELECT a.* FROM news_articles a
               JOIN news_article_symbols s ON s.article_id = a.id
               WHERE s.symbol = ?
               ORDER BY a.created_at DESC LIMIT ?""",
            (symbol.upper(), limit),
        )

    def latest_news(self, limit: int = 100) -> list[dict]:
        """Most recent articles across the whole market."""
        return self.query(
            "SELECT * FROM news_articles ORDER BY created_at DESC LIMIT ?", (limit,)
        )

    def news_symbol_rows(self, since_iso: str | None = None) -> list[dict]:
        """Raw (symbol, created_at) pairs, used to build as-of news features."""
        if since_iso:
            return self.query(
                "SELECT symbol, created_at, article_id FROM news_article_symbols "
                "WHERE created_at >= ? ORDER BY created_at ASC",
                (since_iso,),
            )
        return self.query(
            "SELECT symbol, created_at, article_id FROM news_article_symbols "
            "ORDER BY created_at ASC"
        )

    def news_counts_by_symbol(self, since_iso: str) -> list[dict]:
        """Article counts per symbol since a timestamp, most active first."""
        return self.query(
            """SELECT s.symbol, COUNT(DISTINCT s.article_id) AS news_count,
                      MAX(s.created_at) AS latest_at
               FROM news_article_symbols s
               WHERE s.created_at >= ?
               GROUP BY s.symbol
               ORDER BY news_count DESC, latest_at DESC""",
            (since_iso,),
        )

    def news_coverage(self) -> dict:
        """Coverage stats that answer 'is this really market-wide?'."""
        totals = self.query_one(
            "SELECT COUNT(*) AS articles, MIN(created_at) AS first_at, "
            "MAX(created_at) AS last_at FROM news_articles"
        ) or {}
        symbols = self.query_one(
            "SELECT COUNT(DISTINCT symbol) AS symbols FROM news_article_symbols"
        ) or {}
        links = self.query_one(
            "SELECT COUNT(*) AS links FROM news_article_symbols"
        ) or {}
        return {
            "articles": totals.get("articles", 0),
            "distinct_symbols": symbols.get("symbols", 0),
            "article_symbol_links": links.get("links", 0),
            "first_article_at": totals.get("first_at"),
            "last_article_at": totals.get("last_at"),
        }

    def latest_news_created_at(self) -> str | None:
        row = self.query_one("SELECT MAX(created_at) AS m FROM news_articles")
        return row["m"] if row and row["m"] else None

    def start_news_run(self, window_start: str | None, window_end: str | None) -> int:
        return self.insert(
            """INSERT INTO news_ingest_runs
               (started_at, status, window_start, window_end) VALUES (?,?,?,?)""",
            (utc_now_iso(), "RUNNING", window_start, window_end),
        )

    def finish_news_run(self, run_id: int, status: str, **kw: Any) -> None:
        self.execute(
            """UPDATE news_ingest_runs SET finished_at=?, status=?, pages_fetched=?,
                   articles_seen=?, articles_stored=?, symbols_seen=?, truncated=?,
                   detail=?
               WHERE id=?""",
            (
                utc_now_iso(), status, kw.get("pages_fetched", 0),
                kw.get("articles_seen", 0), kw.get("articles_stored", 0),
                kw.get("symbols_seen", 0), 1 if kw.get("truncated") else 0,
                json.dumps(kw.get("detail"), default=str) if kw.get("detail") else None,
                run_id,
            ),
        )

    def recent_news_runs(self, limit: int = 20) -> list[dict]:
        return self.query(
            "SELECT * FROM news_ingest_runs ORDER BY started_at DESC LIMIT ?", (limit,)
        )

    # -- assets / universe -------------------------------------------------
    def upsert_assets(self, assets: Iterable[dict]) -> int:
        stamp = utc_now_iso()
        return self.executemany(
            """INSERT INTO assets
               (symbol, name, exchange, asset_class, status, tradable, shortable,
                fractionable, refreshed_at)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(symbol) DO UPDATE SET
                 name=excluded.name, exchange=excluded.exchange,
                 asset_class=excluded.asset_class, status=excluded.status,
                 tradable=excluded.tradable, shortable=excluded.shortable,
                 fractionable=excluded.fractionable,
                 refreshed_at=excluded.refreshed_at""",
            [
                (
                    a["symbol"].upper(), a.get("name"), a.get("exchange"),
                    a.get("asset_class"), a.get("status"),
                    1 if a.get("tradable") else 0, 1 if a.get("shortable") else 0,
                    1 if a.get("fractionable") else 0, stamp,
                )
                for a in assets
            ],
        )

    def tradable_symbols(self) -> set[str]:
        return {
            row["symbol"]
            for row in self.query("SELECT symbol FROM assets WHERE tradable=1")
        }

    def asset_count(self) -> dict:
        row = self.query_one(
            "SELECT COUNT(*) AS total, SUM(tradable) AS tradable, "
            "MAX(refreshed_at) AS refreshed_at FROM assets"
        ) or {}
        return {
            "total": row.get("total", 0) or 0,
            "tradable": row.get("tradable", 0) or 0,
            "refreshed_at": row.get("refreshed_at"),
        }

    def save_universe_snapshot(self, entries: Iterable[dict]) -> int:
        stamp = utc_now_iso()
        return self.executemany(
            """INSERT INTO universe_snapshots
               (snapshot_at, symbol, source, news_count, rank, admitted, reason)
               VALUES (?,?,?,?,?,?,?)""",
            [
                (
                    stamp, e["symbol"], e["source"], e.get("news_count", 0),
                    e.get("rank"), 1 if e.get("admitted", True) else 0, e.get("reason"),
                )
                for e in entries
            ],
        )

    def latest_universe(self) -> list[dict]:
        row = self.query_one("SELECT MAX(snapshot_at) AS mx FROM universe_snapshots")
        if not row or not row["mx"]:
            return []
        return self.query(
            "SELECT * FROM universe_snapshots WHERE snapshot_at=? "
            "ORDER BY source, rank, symbol",
            (row["mx"],),
        )

    # -- app state (kill switch etc.) ------------------------------------
    def set_state(self, key: str, value: Any) -> None:
        self.execute(
            """INSERT INTO app_state (key, value, updated_at) VALUES (?,?,?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                                              updated_at=excluded.updated_at""",
            (key, json.dumps(value, default=str), utc_now_iso()),
        )

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self.query_one("SELECT value FROM app_state WHERE key=?", (key,))
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default
