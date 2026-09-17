"""Data-quality gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from stockbot.data.validation import BarValidator
from tests.conftest import synthetic_bars


@pytest.fixture
def validator():
    return BarValidator(bar_minutes=10, max_age_seconds=1800, min_rows=10, settle_seconds=45)


def _now_after(frame: pd.DataFrame, seconds: int = 60) -> datetime:
    last = frame["bar_start"].iloc[-1]
    return last.to_pydatetime() + timedelta(minutes=10, seconds=seconds)


def test_bar_is_not_complete_before_window_closes(validator):
    start = datetime(2025, 6, 2, 14, 30, tzinfo=timezone.utc)
    assert not validator.bar_is_complete(start, now=start + timedelta(minutes=9))
    assert not validator.bar_is_complete(start, now=start + timedelta(minutes=10))
    # Window closed at +10min, settle adds 45s.
    assert not validator.bar_is_complete(start, now=start + timedelta(minutes=10, seconds=30))
    assert validator.bar_is_complete(start, now=start + timedelta(minutes=10, seconds=46))


def test_clean_data_passes(validator, bars):
    cleaned, report = validator.validate("TEST", bars, now=_now_after(bars))
    assert report.ok
    assert report.rows_out == len(bars)
    assert report.duplicates_removed == 0


def test_duplicate_timestamps_removed(validator, bars):
    duped = pd.concat([bars, bars.tail(3)], ignore_index=True)
    cleaned, report = validator.validate("TEST", duped, now=_now_after(bars))
    assert report.duplicates_removed == 3
    assert cleaned["bar_start"].is_unique


def test_impossible_ohlc_removed(validator, bars):
    broken = bars.copy()
    broken.loc[5, "high"] = broken.loc[5, "low"] - 1.0
    cleaned, report = validator.validate("TEST", broken, now=_now_after(bars))
    assert report.invalid_ohlc_removed == 1
    assert len(cleaned) == len(bars) - 1


def test_nonpositive_prices_removed(validator, bars):
    broken = bars.copy()
    for col in ("open", "high", "low", "close"):
        broken.loc[7, col] = 0.0
    cleaned, report = validator.validate("TEST", broken, now=_now_after(bars))
    assert report.nonpositive_removed + report.invalid_ohlc_removed >= 1
    assert (cleaned["close"] > 0).all()


def test_stale_data_flagged(validator, bars):
    late = bars["bar_start"].iloc[-1].to_pydatetime() + timedelta(hours=3)
    cleaned, report = validator.validate("TEST", bars, now=late)
    assert report.stale
    assert not report.ok
    assert any("stale" in reason for reason in report.reasons)


def test_incomplete_trailing_bar_dropped(validator, bars):
    # "Now" is only five minutes past the final bar's start.
    now = bars["bar_start"].iloc[-1].to_pydatetime() + timedelta(minutes=5)
    cleaned, report = validator.validate("TEST", bars, now=now)
    assert len(cleaned) == len(bars) - 1


def test_empty_frame_rejected(validator):
    cleaned, report = validator.validate("TEST", pd.DataFrame())
    assert not report.ok
    assert "no_data" in report.reasons
    assert cleaned.empty


def test_insufficient_history_rejected(bars):
    validator = BarValidator(bar_minutes=10, max_age_seconds=1800, min_rows=100_000)
    cleaned, report = validator.validate("TEST", bars, now=_now_after(bars))
    assert not report.ok
    assert any("insufficient_history" in r for r in report.reasons)


def test_intraday_gap_counted(validator, bars):
    with_gap = bars.drop(index=[20, 21]).reset_index(drop=True)
    cleaned, report = validator.validate("TEST", with_gap, now=_now_after(bars))
    assert report.gaps_detected >= 2


def test_overnight_break_is_not_a_gap(validator):
    two_days = synthetic_bars(n_sessions=2)
    cleaned, report = validator.validate("TEST", two_days, now=_now_after(two_days))
    assert report.gaps_detected == 0
