"""Feature engineering, including leakage checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockbot.features import (
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    atr,
    build_features,
    latest_feature_row,
    macd,
    prepare_training_frame,
    rsi,
)
from tests.conftest import synthetic_bars


def test_all_features_present(bars, benchmark_bars):
    featured = build_features(bars, benchmark_bars)
    missing = [c for c in FEATURE_COLUMNS if c not in featured.columns]
    assert missing == []


def test_every_feature_has_a_description():
    assert set(FEATURE_COLUMNS) <= set(FEATURE_DESCRIPTIONS)


def test_no_lookahead_features_change_when_future_is_truncated(bars, benchmark_bars):
    """The defining leakage test.

    Features for bar *t* must be identical whether or not bars after *t* exist.
    """
    cutoff = len(bars) - 30
    full = build_features(bars, benchmark_bars)
    truncated = build_features(
        bars.iloc[:cutoff].reset_index(drop=True),
        benchmark_bars.iloc[:cutoff].reset_index(drop=True),
    )

    compare = [c for c in FEATURE_COLUMNS if c in full.columns]
    tail_full = full.iloc[cutoff - 1][compare].astype(float)
    tail_truncated = truncated.iloc[-1][compare].astype(float)

    for name in compare:
        a, b = tail_full[name], tail_truncated[name]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), f"{name} changed with future data"


def test_rsi_bounds_and_direction():
    rising = pd.Series(np.linspace(100, 130, 80))
    values = rsi(rising, 14).dropna()
    assert (values >= 0).all() and (values <= 100).all()
    assert values.iloc[-1] > 70


def test_rsi_falling_series_is_low():
    falling = pd.Series(np.linspace(130, 100, 80))
    values = rsi(falling, 14).dropna()
    assert values.iloc[-1] < 30


def test_macd_histogram_is_line_minus_signal():
    series = pd.Series(np.cumsum(np.random.default_rng(3).normal(0, 1, 300)) + 100)
    line, sig, hist = macd(series)
    valid = hist.dropna().index
    assert np.allclose((line - sig).loc[valid], hist.loc[valid])


def test_atr_is_non_negative(bars):
    values = atr(bars["high"], bars["low"], bars["close"], 14).dropna()
    assert (values >= 0).all()


def test_time_of_day_features_respect_session(bars, benchmark_bars):
    featured = build_features(bars, benchmark_bars)
    assert featured["minutes_since_open"].min() == 0
    assert featured["session_progress"].between(0, 1).all()
    assert set(featured["is_first_hour"].unique()) <= {0.0, 1.0}
    assert featured["day_of_week"].between(0, 4).all()


def test_first_bar_of_session_flagged_first_hour(bars, benchmark_bars):
    featured = build_features(bars, benchmark_bars)
    first_bars = featured[featured["minutes_since_open"] == 0]
    assert (first_bars["is_first_hour"] == 1.0).all()


def test_benchmark_relative_features_nan_without_benchmark(bars):
    featured = build_features(bars, benchmark=None)
    assert featured["spy_relative_12"].isna().all()
    assert featured["beta_78"].isna().all()


def test_prepare_training_frame_drops_all_nan_features(bars):
    featured = build_features(bars, benchmark=None)
    _, usable = prepare_training_frame(featured, FEATURE_COLUMNS)
    assert "spy_relative_12" not in usable
    assert "rsi_14" in usable


def test_no_infinities_survive(bars, benchmark_bars):
    featured = build_features(bars, benchmark_bars)
    numeric = featured[FEATURE_COLUMNS].to_numpy(dtype=float)
    assert not np.isinf(numeric).any()


def test_latest_feature_row_requires_complete_vector(bars, benchmark_bars):
    featured = build_features(bars, benchmark_bars)
    _, usable = prepare_training_frame(featured, FEATURE_COLUMNS)
    row = latest_feature_row(featured, usable)
    assert row is not None
    assert not row[usable].isna().any()


def test_empty_input_returns_empty_frame():
    out = build_features(pd.DataFrame())
    assert out.empty


def test_features_handle_short_history():
    short = synthetic_bars(n_sessions=1)
    featured = build_features(short, benchmark=None)
    assert len(featured) == len(short)
    # Long-window features cannot be computed from one session.
    assert featured["realised_vol_78"].isna().all()
