"""Target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockbot.features import FEATURE_COLUMNS, build_features, prepare_training_frame
from stockbot.labeling import add_labels, build_supervised_frame, drop_unresolved


def test_forward_return_matches_manual_calculation():
    frame = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]})
    labelled = add_labels(frame, horizon_bars=2, round_trip_cost=0.0)
    assert labelled["forward_return"].iloc[0] == pytest.approx(102 / 100 - 1)
    assert labelled["forward_return"].iloc[2] == pytest.approx(104 / 102 - 1)
    assert np.isnan(labelled["forward_return"].iloc[3])


def test_costs_are_deducted_before_labelling():
    frame = pd.DataFrame({"close": [100.0, 100.05, 100.0]})
    # +5 bps gross move against a 10 bps round trip is a negative example.
    labelled = add_labels(frame, horizon_bars=1, round_trip_cost=0.001)
    assert labelled["forward_return"].iloc[0] > 0
    assert labelled["net_forward_return"].iloc[0] < 0
    assert labelled["target"].iloc[0] == 0.0


def test_positive_label_requires_clearing_costs():
    frame = pd.DataFrame({"close": [100.0, 101.0, 100.0]})
    labelled = add_labels(frame, horizon_bars=1, round_trip_cost=0.001)
    assert labelled["target"].iloc[0] == 1.0


def test_threshold_raises_the_bar():
    frame = pd.DataFrame({"close": [100.0, 100.2, 100.0]})
    loose = add_labels(frame, horizon_bars=1, round_trip_cost=0.001, threshold=0.0)
    strict = add_labels(frame, horizon_bars=1, round_trip_cost=0.001, threshold=0.005)
    assert loose["target"].iloc[0] == 1.0
    assert strict["target"].iloc[0] == 0.0


def test_unresolved_tail_dropped():
    frame = pd.DataFrame({"close": np.linspace(100, 110, 50)})
    labelled = add_labels(frame, horizon_bars=10, round_trip_cost=0.0)
    resolved = drop_unresolved(labelled, 10)
    assert len(resolved) == 40
    assert resolved["target"].notna().all()


def test_supervised_frame_has_no_nan_targets(bars, benchmark_bars):
    featured = build_features(bars, benchmark_bars)
    _, usable = prepare_training_frame(featured, FEATURE_COLUMNS)
    supervised = build_supervised_frame(featured, usable, 39, 0.001)
    assert not supervised.empty
    assert supervised["target"].notna().all()
    assert supervised[usable].notna().all().all()


def test_supervised_frame_is_chronological(bars, benchmark_bars):
    featured = build_features(bars, benchmark_bars)
    _, usable = prepare_training_frame(featured, FEATURE_COLUMNS)
    supervised = build_supervised_frame(featured, usable, 39, 0.001)
    assert supervised["bar_start"].is_monotonic_increasing


def test_supervised_frame_excludes_last_horizon_bars(bars, benchmark_bars):
    """The final rows have no observable future, so they must not be trainable."""
    featured = build_features(bars, benchmark_bars)
    _, usable = prepare_training_frame(featured, FEATURE_COLUMNS)
    supervised = build_supervised_frame(featured, usable, 39, 0.001)
    assert supervised["bar_start"].max() < featured["bar_start"].max()
