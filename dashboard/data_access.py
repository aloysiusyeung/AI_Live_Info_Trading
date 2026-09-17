"""Read-side helpers for the dashboard.

The dashboard never writes market data or submits orders. The only write it
performs is the emergency kill switch, which is intentionally one-way safe: it
can always stop trading, and releasing it is an explicit, confirmed action.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from stockbot.db import Database


def parse_json(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def latest_signals(db: Database) -> pd.DataFrame:
    rows = db.latest_signals()
    for row in rows:
        row["risk_reasons"] = parse_json(row.get("risk_reasons"), [])
    return to_frame(rows)


def signal_history(db: Database, limit: int = 300) -> pd.DataFrame:
    return to_frame(
        db.query("SELECT * FROM signals ORDER BY created_at DESC LIMIT ?", (limit,))
    )


def prediction_history(db: Database, limit: int = 300) -> pd.DataFrame:
    rows = db.query("SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,))
    for row in rows:
        row["top_features"] = parse_json(row.get("top_features"), [])
    return to_frame(rows)


def order_history(db: Database, limit: int = 200) -> pd.DataFrame:
    return to_frame(db.recent_orders(limit=limit))


def pending_orders(db: Database) -> pd.DataFrame:
    """Locally recorded orders that are not in a terminal state.

    The status list comes from :data:`stockbot.orders.PENDING_STATUSES` so the
    dashboard and the reconciler cannot drift apart.
    """
    from stockbot.orders import PENDING_STATUSES

    statuses = sorted(PENDING_STATUSES | {"submitting"})
    placeholders = ",".join("?" for _ in statuses)
    return to_frame(
        db.query(
            f"SELECT * FROM paper_orders WHERE LOWER(status) IN ({placeholders}) "
            "ORDER BY created_at DESC",
            statuses,
        )
    )


def latest_positions(db: Database) -> pd.DataFrame:
    row = db.query_one("SELECT MAX(snapshot_at) AS mx FROM positions")
    if not row or not row["mx"]:
        return pd.DataFrame()
    return to_frame(
        db.query("SELECT * FROM positions WHERE snapshot_at = ?", (row["mx"],))
    )


def account_history(db: Database, limit: int = 500) -> pd.DataFrame:
    rows = db.query(
        "SELECT * FROM account_snapshots ORDER BY snapshot_at DESC LIMIT ?", (limit,)
    )
    df = to_frame(rows)
    if not df.empty:
        df["snapshot_at"] = pd.to_datetime(df["snapshot_at"], utc=True, format="mixed")
        df = df.sort_values("snapshot_at")
    return df


def backtest_runs(db: Database, limit: int = 100) -> pd.DataFrame:
    rows = db.query("SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT ?", (limit,))
    for row in rows:
        row["metrics"] = parse_json(row.get("metrics"), {})
        row["baselines"] = parse_json(row.get("baselines"), {})
    return to_frame(rows)


def model_versions(db: Database, limit: int = 50) -> pd.DataFrame:
    rows = db.query("SELECT * FROM model_versions ORDER BY created_at DESC LIMIT ?", (limit,))
    for row in rows:
        row["oos_metrics"] = parse_json(row.get("oos_metrics"), {})
        row["feature_names"] = parse_json(row.get("feature_names"), [])
    return to_frame(rows)


def bars_frame(db: Database, symbol: str, bar_minutes: int, limit: int = 400) -> pd.DataFrame:
    rows = db.get_bars(symbol, bar_minutes, limit=limit)
    df = to_frame(rows)
    if not df.empty:
        df["bar_start"] = pd.to_datetime(df["bar_start"], utc=True, format="mixed")
    return df


def recent_errors(db: Database, limit: int = 50) -> pd.DataFrame:
    return to_frame(db.recent_errors(limit=limit))


def scheduler_runs(db: Database, limit: int = 25) -> pd.DataFrame:
    return to_frame(db.recent_scheduler_runs(limit=limit))


def paper_performance(db: Database) -> dict[str, Any]:
    """Equity-based performance of the paper account, from stored snapshots."""
    history = account_history(db)
    if history.empty or "equity" not in history.columns:
        return {"available": False, "reason": "no account snapshots recorded yet"}

    equity = history["equity"].dropna()
    if len(equity) < 2:
        return {"available": False, "reason": "need at least two account snapshots"}

    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    peak = equity.cummax()
    drawdown = (equity / peak - 1.0).min()

    orders = order_history(db, limit=1000)
    filled = 0
    if not orders.empty and "status" in orders.columns:
        filled = int((orders["status"].str.lower() == "filled").sum())

    return {
        "available": True,
        "start_equity": start,
        "current_equity": end,
        "total_return": (end / start - 1.0) if start else None,
        "max_drawdown": float(drawdown) if pd.notna(drawdown) else None,
        "snapshots": int(len(equity)),
        "first_snapshot": history["snapshot_at"].iloc[0],
        "last_snapshot": history["snapshot_at"].iloc[-1],
        "orders_filled": filled,
        "history": history,
    }


def age_text(iso_timestamp: Any) -> str:
    """'3m ago' style relative time for a stored ISO timestamp."""
    if not iso_timestamp:
        return "never"
    try:
        stamp = pd.to_datetime(iso_timestamp, utc=True, format="mixed").to_pydatetime()
    except (ValueError, TypeError):
        return str(iso_timestamp)
    delta = (datetime.now(timezone.utc) - stamp).total_seconds()
    if delta < 0:
        return f"in {int(abs(delta) // 60)}m"
    if delta < 90:
        return f"{int(delta)}s ago"
    if delta < 5400:
        return f"{int(delta // 60)}m ago"
    if delta < 172800:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // 86400)}d ago"


# -- news (market-wide) -----------------------------------------------------
def news_feed(db: Database, limit: int = 100) -> pd.DataFrame:
    """Most recent articles across the whole market, with their symbol tags."""
    rows = db.latest_news(limit=limit)
    if not rows:
        return pd.DataFrame()
    ids = [r["id"] for r in rows]
    placeholders = ",".join("?" for _ in ids)
    links = db.query(
        f"SELECT article_id, symbol FROM news_article_symbols "
        f"WHERE article_id IN ({placeholders})",
        ids,
    )
    by_article: dict[int, list[str]] = {}
    for link in links:
        by_article.setdefault(link["article_id"], []).append(link["symbol"])
    for row in rows:
        row["symbols"] = ", ".join(sorted(by_article.get(row["id"], [])))
    return to_frame(rows)


def news_for_symbol(db: Database, symbol: str, limit: int = 20) -> pd.DataFrame:
    return to_frame(db.news_for_symbol(symbol, limit=limit))


def news_coverage(db: Database) -> dict[str, Any]:
    stats = db.news_coverage()
    runs = db.recent_news_runs(limit=20)
    stats["truncated_runs"] = sum(1 for r in runs if r.get("truncated"))
    stats["last_run_status"] = runs[0]["status"] if runs else None
    stats["last_run_at"] = runs[0]["started_at"] if runs else None
    return stats


def news_runs(db: Database, limit: int = 20) -> pd.DataFrame:
    return to_frame(db.recent_news_runs(limit=limit))


def top_news_symbols(db: Database, hours: int = 48, limit: int = 25) -> pd.DataFrame:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    return to_frame(db.news_counts_by_symbol(since)[:limit])


def analysis_universe(db: Database) -> pd.DataFrame:
    return to_frame(db.latest_universe())


def asset_stats(db: Database) -> dict[str, Any]:
    return db.asset_count()
