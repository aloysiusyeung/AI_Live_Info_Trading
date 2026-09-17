"""Shared fixtures.

No test touches the network. Synthetic bars are generated from a seeded random
walk and are clearly labelled as synthetic: they exist to exercise code paths,
never to stand in for real market data or to produce performance claims.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

NY = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from a known, paper-only environment."""
    for key in (
        "ALPACA_PAPER", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "WATCHLIST",
        "ENABLE_PAPER_ORDERS", "KILL_SWITCH", "ALPACA_DATA_FEED", "DATABASE_PATH",
        "MODEL_DIR", "LOG_DIR", "MIN_TRAINING_ROWS", "PREDICTION_HORIZON_BARS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ALPACA_PAPER", "true")
    monkeypatch.setenv("ALPACA_API_KEY", "test-key-not-real")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret-not-real")
    monkeypatch.setenv("WATCHLIST", "AAPL,MSFT")
    monkeypatch.setenv("ENABLE_PAPER_ORDERS", "false")


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    from stockbot.config import load_settings

    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    return load_settings()


@pytest.fixture
def db(tmp_settings):
    from stockbot.db import Database

    return Database(tmp_settings.database_path)


def session_timestamps(n_sessions: int, bars_per_session: int = 39,
                       end_date: datetime | None = None) -> list[datetime]:
    """Regular-session 10-minute bar starts, weekends skipped.

    Uses a 09:30 ET open so the timestamps behave like real session data under
    both EST and EDT.
    """
    end = (end_date or datetime(2025, 6, 30, tzinfo=NY)).date()
    days: list = []
    cursor = end
    while len(days) < n_sessions:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    days.reverse()

    stamps: list[datetime] = []
    for day in days:
        open_dt = datetime.combine(day, time(9, 30), tzinfo=NY)
        for i in range(bars_per_session):
            stamps.append((open_dt + timedelta(minutes=10 * i)).astimezone(timezone.utc))
    return stamps


def synthetic_bars(
    symbol: str = "TEST",
    n_sessions: int = 40,
    bars_per_session: int = 39,
    seed: int = 11,
    start_price: float = 100.0,
    drift: float = 0.0,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """SYNTHETIC random-walk bars for testing code paths only.

    This is not market data and must never be presented as such.
    """
    rng = np.random.default_rng(seed)
    stamps = session_timestamps(n_sessions, bars_per_session, end_date=end_date)
    n = len(stamps)

    steps = rng.normal(loc=drift, scale=0.0015, size=n)
    close = start_price * np.exp(np.cumsum(steps))
    spread = np.abs(rng.normal(0.0, 0.0008, size=n)) * close
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(60_000, 400_000, size=n).astype(float)

    return pd.DataFrame(
        {
            "symbol": symbol,
            "bar_start": stamps,
            "bar_minutes": 10,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "trade_count": volume / 90.0,
            "vwap": (high + low + close) / 3.0,
        }
    )


@pytest.fixture
def bars():
    return synthetic_bars()


@pytest.fixture
def benchmark_bars():
    return synthetic_bars(symbol="SPY", seed=29, start_price=450.0)
