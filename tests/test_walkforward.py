"""Walk-forward validation: ordering, embargo and leakage-free preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockbot.models.registry import build_candidates
from stockbot.models.walkforward import expanding_splits, walk_forward_evaluate


def test_splits_are_chronological_and_non_overlapping():
    for train, test in expanding_splits(1000, 4, embargo=39, min_train=500):
        assert train.max() < test.min()
        assert len(np.intersect1d(train, test)) == 0


def test_embargo_gap_is_respected():
    embargo = 39
    for train, test in expanding_splits(1000, 4, embargo=embargo, min_train=500):
        assert test.min() - train.max() > embargo - 1


def test_training_windows_expand():
    sizes = [len(train) for train, _ in expanding_splits(1200, 4, 39, 500)]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) > 1


def test_no_splits_when_data_too_short():
    assert list(expanding_splits(100, 5, embargo=39, min_train=500)) == []


def test_last_fold_reaches_end_of_data():
    splits = list(expanding_splits(1000, 4, 39, 500))
    assert splits[-1][1].max() == 999


def _labelled_frame(n: int = 1400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, 4))
    forward = features[:, 0] * 0.002 + rng.normal(0, 0.004, n)
    return pd.DataFrame(
        {
            "f0": features[:, 0],
            "f1": features[:, 1],
            "f2": features[:, 2],
            "f3": features[:, 3],
            "target": (forward > 0).astype(float),
            "forward_return": forward,
            "bar_start": pd.date_range("2025-01-01", periods=n, freq="10min", tz="UTC"),
        }
    )


def test_walk_forward_produces_out_of_sample_predictions():
    frame = _labelled_frame()
    pipeline = build_candidates()["logistic_regression"]
    result = walk_forward_evaluate(
        pipeline, frame, ["f0", "f1", "f2", "f3"], "logistic_regression",
        n_splits=4, embargo=39, min_train=600,
    )
    assert result.metrics["n_folds"] >= 3
    assert not result.predictions.empty
    assert result.predictions["prob_up"].between(0, 1).all()


def test_predictions_come_only_from_test_periods():
    frame = _labelled_frame()
    pipeline = build_candidates()["logistic_regression"]
    result = walk_forward_evaluate(
        pipeline, frame, ["f0", "f1", "f2", "f3"], "logistic_regression",
        n_splits=4, embargo=39, min_train=600,
    )
    # No prediction may precede the first test block.
    first_test_start = min(f.test_start for f in result.folds)
    assert result.predictions["bar_start"].min() >= first_test_start


def test_scaler_is_fitted_per_fold_not_globally():
    """A shifted test block must not affect training-fold statistics.

    If the scaler were fitted on the whole series, the huge shift in the tail
    would change the transform applied to early training rows and the first
    fold's predictions would move. They must not.
    """
    frame = _labelled_frame()
    shifted = frame.copy()
    tail = shifted.index[-200:]
    shifted.loc[tail, "f0"] = shifted.loc[tail, "f0"] + 1000.0

    features = ["f0", "f1", "f2", "f3"]
    kwargs = dict(n_splits=4, embargo=39, min_train=600)
    base = walk_forward_evaluate(
        build_candidates()["logistic_regression"], frame, features, "lr", **kwargs
    )
    moved = walk_forward_evaluate(
        build_candidates()["logistic_regression"], shifted, features, "lr", **kwargs
    )

    fold0_base = base.predictions[base.predictions["fold"] == 0]["prob_up"].to_numpy()
    fold0_moved = moved.predictions[moved.predictions["fold"] == 0]["prob_up"].to_numpy()
    assert np.allclose(fold0_base, fold0_moved)


def test_metrics_report_costs_and_trade_counts():
    frame = _labelled_frame()
    result = walk_forward_evaluate(
        build_candidates()["logistic_regression"], frame, ["f0", "f1", "f2", "f3"], "lr",
        n_splits=4, embargo=39, min_train=600, round_trip_cost=0.001,
    )
    assert result.metrics["cost_bps"] == pytest.approx(10.0)
    assert result.metrics["n_trades"] >= 0
    assert 0 <= result.metrics["trade_rate"] <= 1


def test_insufficient_data_reported_not_faked():
    short = _labelled_frame(n=50)
    result = walk_forward_evaluate(
        build_candidates()["logistic_regression"], short, ["f0", "f1", "f2", "f3"], "lr",
        n_splits=5, embargo=39, min_train=600,
    )
    assert result.metrics["insufficient_data"] is True
    assert result.predictions.empty


def test_all_candidates_run():
    frame = _labelled_frame(n=1200)
    for name, pipeline in build_candidates().items():
        result = walk_forward_evaluate(
            pipeline, frame, ["f0", "f1", "f2", "f3"], name,
            n_splits=3, embargo=39, min_train=600,
        )
        assert not result.predictions.empty, name
