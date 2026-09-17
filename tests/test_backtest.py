"""Backtesting and baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockbot.backtest import (
    backtest_predictions,
    baseline_buy_and_hold,
    baseline_momentum,
    compare_to_baselines,
)


def _predictions(n: int = 300, prob: float = 0.9, ret: float = 0.01) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bar_start": pd.date_range("2025-01-01", periods=n, freq="10min", tz="UTC"),
            "prob_up": np.full(n, prob),
            "forward_return": np.full(n, ret),
            "target": np.ones(n),
        }
    )


def test_costs_reduce_returns():
    preds = _predictions()
    free = backtest_predictions(preds, 0.55, round_trip_cost=0.0, horizon_bars=39)
    costly = backtest_predictions(preds, 0.55, round_trip_cost=0.002, horizon_bars=39)
    assert costly.metrics["total_return"] < free.metrics["total_return"]


def test_a_strategy_that_cannot_beat_costs_loses_money():
    # A +5 bps move cannot pay a 20 bps round trip.
    preds = _predictions(ret=0.0005)
    result = backtest_predictions(preds, 0.55, round_trip_cost=0.002, horizon_bars=39)
    assert result.metrics["total_return"] < 0


def test_holding_periods_do_not_overlap():
    preds = _predictions(n=390)
    result = backtest_predictions(preds, 0.55, round_trip_cost=0.0, horizon_bars=39)
    # 390 bars / 39-bar holds => at most 10 entries.
    assert result.metrics["n_trades"] <= 10


def test_no_trades_when_probability_below_threshold():
    preds = _predictions(prob=0.3)
    result = backtest_predictions(preds, 0.55, round_trip_cost=0.001, horizon_bars=39)
    assert result.metrics["n_trades"] == 0
    assert result.metrics["total_return"] == pytest.approx(0.0)


def test_empty_predictions_report_insufficient_data():
    result = backtest_predictions(pd.DataFrame(), 0.55, 0.001, 39)
    assert result.metrics["insufficient_data"] is True


def test_buy_and_hold_tracks_the_price(bars):
    result = baseline_buy_and_hold(bars, round_trip_cost=0.0, horizon_bars=39)
    price_return = bars["close"].iloc[-1] / bars["close"].iloc[0] - 1
    assert result.metrics["total_return"] == pytest.approx(price_return, rel=1e-6)


def test_momentum_baseline_uses_lagged_signal(bars, benchmark_bars):
    from stockbot.features import build_features

    featured = build_features(bars, benchmark_bars)
    result = baseline_momentum(featured, round_trip_cost=0.001)
    assert not result.metrics.get("insufficient_data")
    assert result.metrics["n_periods"] > 0


def test_momentum_reports_insufficient_data_without_the_column(bars):
    result = baseline_momentum(bars, round_trip_cost=0.001)
    assert result.metrics["insufficient_data"] is True


def test_drawdown_is_never_positive():
    rng = np.random.default_rng(1)
    preds = pd.DataFrame(
        {
            "bar_start": pd.date_range("2025-01-01", periods=400, freq="10min", tz="UTC"),
            "prob_up": rng.uniform(0.4, 0.8, 400),
            "forward_return": rng.normal(0, 0.01, 400),
            "target": rng.integers(0, 2, 400).astype(float),
        }
    )
    result = backtest_predictions(preds, 0.55, 0.001, 39)
    assert result.metrics["max_drawdown"] <= 0


def test_comparison_refuses_to_claim_a_win_without_data():
    empty = backtest_predictions(pd.DataFrame(), 0.55, 0.001, 39)
    comparison = compare_to_baselines(empty, {})
    assert comparison["beats_baselines"] is False
    assert "INSUFFICIENT_EVIDENCE" in comparison["verdict"]


def test_comparison_reports_a_loss_honestly(bars):
    losing = backtest_predictions(_predictions(ret=-0.02), 0.55, 0.001, 39)
    bh = baseline_buy_and_hold(bars, 0.001, 39)
    comparison = compare_to_baselines(losing, {"buy_and_hold": bh})
    assert comparison["beats_baselines"] is False
    assert "did NOT beat" in comparison["verdict"]
