"""Data-quality gate.

Nothing reaches the feature layer until it passes here. The checks are
deliberately conservative: when a check cannot be evaluated the bar is rejected,
because a wrong trade is more expensive than a missed one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Outcome of validating one symbol's bar history."""

    symbol: str
    ok: bool = True
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    invalid_ohlc_removed: int = 0
    nonpositive_removed: int = 0
    gaps_detected: int = 0
    stale: bool = False
    age_seconds: float | None = None
    latest_bar_start: datetime | None = None
    reasons: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> "ValidationReport":
        self.ok = False
        self.reasons.append(reason)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ok": self.ok,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "duplicates_removed": self.duplicates_removed,
            "invalid_ohlc_removed": self.invalid_ohlc_removed,
            "nonpositive_removed": self.nonpositive_removed,
            "gaps_detected": self.gaps_detected,
            "stale": self.stale,
            "age_seconds": self.age_seconds,
            "latest_bar_start": self.latest_bar_start.isoformat() if self.latest_bar_start else None,
            "reasons": list(self.reasons),
        }


class BarValidator:
    """Rejects stale, incomplete, duplicated or missing bar data."""

    def __init__(
        self,
        bar_minutes: int,
        max_age_seconds: int,
        min_rows: int = 60,
        settle_seconds: int = 45,
    ) -> None:
        self.bar_minutes = bar_minutes
        self.max_age_seconds = max_age_seconds
        self.min_rows = min_rows
        self.settle_seconds = settle_seconds

    # -- helpers ---------------------------------------------------------
    def bar_is_complete(self, bar_start: datetime, now: datetime | None = None) -> bool:
        """True once the bar's window has closed *and* the settle delay elapsed.

        Alpaca stamps a bar with its **start**, so a 10-minute bar stamped
        14:30 covers 14:30:00-14:39:59.999 and must not be analysed before
        14:40 plus a small settle allowance for late-arriving trades.
        """
        now = now or datetime.now(timezone.utc)
        bar_end = bar_start + timedelta(minutes=self.bar_minutes)
        return now >= bar_end + timedelta(seconds=self.settle_seconds)

    # -- main entry point -------------------------------------------------
    def validate(
        self,
        symbol: str,
        frame: pd.DataFrame,
        now: datetime | None = None,
        session_open: datetime | None = None,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Clean ``frame`` and report on it.

        ``frame`` must have a ``bar_start`` column (tz-aware UTC) plus OHLCV.
        Returns the cleaned frame (may be empty) and the report.
        """
        now = now or datetime.now(timezone.utc)
        report = ValidationReport(symbol=symbol, rows_in=len(frame))

        if frame is None or frame.empty:
            return _empty_like(frame), report.fail("no_data")

        df = frame.copy()
        df["bar_start"] = pd.to_datetime(df["bar_start"], utc=True)

        # 1. Duplicates: keep the last observation for a timestamp.
        before = len(df)
        df = df.sort_values("bar_start").drop_duplicates(subset=["bar_start"], keep="last")
        report.duplicates_removed = before - len(df)

        # 2. Structural OHLC sanity.
        ohlc_ok = (
            (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
        )
        report.invalid_ohlc_removed = int((~ohlc_ok).sum())
        df = df[ohlc_ok]

        # 3. Non-positive prices / negative volume are impossible.
        price_ok = (
            (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)
            & (df["volume"] >= 0)
        )
        report.nonpositive_removed = int((~price_ok).sum())
        df = df[price_ok]

        # 4. Drop any not-yet-complete trailing bar.
        complete = df["bar_start"].apply(lambda ts: self.bar_is_complete(ts, now))
        incomplete_dropped = int((~complete).sum())
        df = df[complete]
        if incomplete_dropped:
            report.reasons.append(f"dropped_{incomplete_dropped}_incomplete_bars")

        report.rows_out = len(df)
        if df.empty:
            return _empty_like(frame), report.fail("empty_after_cleaning")

        # 5. Intraday gap detection (informational; excludes overnight breaks).
        report.gaps_detected = self._count_intraday_gaps(df["bar_start"])

        # 6. Staleness of the newest bar.
        latest = df["bar_start"].iloc[-1].to_pydatetime()
        report.latest_bar_start = latest
        latest_end = latest + timedelta(minutes=self.bar_minutes)
        age = (now - latest_end).total_seconds()
        report.age_seconds = age
        if age > self.max_age_seconds:
            report.stale = True
            report.fail(f"stale_data_age_{int(age)}s")

        # 7. Enough history to compute features at all.
        if len(df) < self.min_rows:
            report.fail(f"insufficient_history_{len(df)}_rows")

        return df.reset_index(drop=True), report

    def _count_intraday_gaps(self, timestamps: pd.Series) -> int:
        """Count missing bars inside a contiguous session run.

        Overnight and weekend breaks are much larger than a few bar widths, so
        anything above ``max_intraday_gap`` is treated as a session boundary
        rather than a data gap.
        """
        if len(timestamps) < 2:
            return 0
        deltas = timestamps.diff().dropna().dt.total_seconds()
        step = self.bar_minutes * 60
        max_intraday_gap = step * 12  # ~2h at 10-minute bars
        intraday = deltas[(deltas > step) & (deltas <= max_intraday_gap)]
        return int((intraday / step - 1).round().sum())


def _empty_like(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = list(frame.columns) if frame is not None else [
        "symbol", "bar_start", "open", "high", "low", "close", "volume"
    ]
    return pd.DataFrame(columns=columns)
