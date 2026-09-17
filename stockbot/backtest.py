"""Cost-aware backtesting and baseline comparison.

The backtest consumes the *out-of-sample* predictions produced by walk-forward
validation — it never scores in-sample fits. Two baselines are always computed
alongside the model so a positive model return can be judged against doing
something trivial instead:

``buy_and_hold``
    Always in the market, paying the round-trip cost once at the start.
``momentum``
    Long whenever the trailing momentum feature is positive, paying costs on
    every position change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)


def _summarise(
    returns: np.ndarray, bars_per_year: float, n_trades: int, label: str
) -> dict[str, Any]:
    """Summary statistics for a per-bar net return series."""
    if len(returns) == 0:
        return {"name": label, "n_periods": 0, "insufficient_data": True}

    total = float(np.prod(1.0 + returns) - 1.0)
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    periods = len(returns)
    years = periods / bars_per_year if bars_per_year else 0.0

    cagr = None
    if years > 0 and (1.0 + total) > 0:
        cagr = float((1.0 + total) ** (1.0 / years) - 1.0)

    sharpe = None
    if std > 0:
        sharpe = float(mean / std * np.sqrt(bars_per_year))

    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0

    downside = returns[returns < 0]
    sortino = None
    if len(downside) > 1 and downside.std(ddof=1) > 0:
        sortino = float(mean / downside.std(ddof=1) * np.sqrt(bars_per_year))

    return {
        "name": label,
        "n_periods": periods,
        "total_return": total,
        "mean_return_per_period": mean,
        "volatility_per_period": std,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "win_rate": float((returns > 0).mean()),
        "n_trades": n_trades,
        "insufficient_data": False,
    }


def backtest_predictions(
    predictions: pd.DataFrame,
    decision_threshold: float,
    round_trip_cost: float,
    horizon_bars: int,
    bars_per_session: int = 39,
    sessions_per_year: int = 252,
) -> BacktestResult:
    """Backtest a long-flat rule driven by out-of-sample probabilities.

    Non-overlapping holding periods are used: after entering, the next entry is
    only considered ``horizon_bars`` later. Overlapping the horizon would count
    the same price move many times and inflate every statistic.
    """
    if predictions is None or predictions.empty:
        return BacktestResult(name="model", metrics={"insufficient_data": True})

    preds = predictions.sort_values("bar_start").reset_index(drop=True)
    proba = preds["prob_up"].to_numpy(dtype=float)
    forward = preds["forward_return"].to_numpy(dtype=float)

    returns: list[float] = []
    stamps: list[Any] = []
    n_trades = 0
    i = 0
    while i < len(preds):
        if proba[i] >= decision_threshold:
            returns.append(float(forward[i] - round_trip_cost))
            n_trades += 1
            stamps.append(preds["bar_start"].iloc[i])
            i += horizon_bars
        else:
            returns.append(0.0)
            stamps.append(preds["bar_start"].iloc[i])
            i += 1

    returns_arr = np.asarray(returns, dtype=float)
    # Each element spans a different number of bars; annualise on the number of
    # decision points per year implied by non-overlapping horizon holds.
    bars_per_year = bars_per_session * sessions_per_year / max(horizon_bars, 1)
    metrics = _summarise(returns_arr, bars_per_year, n_trades, "model")
    metrics["decision_threshold"] = decision_threshold
    metrics["cost_bps"] = round_trip_cost * 10_000
    metrics["horizon_bars"] = horizon_bars

    equity = pd.DataFrame(
        {
            "bar_start": stamps,
            "period_return": returns_arr,
            "equity": np.cumprod(1.0 + returns_arr),
        }
    )
    return BacktestResult(name="model", metrics=metrics, equity_curve=equity)


def baseline_buy_and_hold(
    frame: pd.DataFrame,
    round_trip_cost: float,
    horizon_bars: int,
    bars_per_session: int = 39,
    sessions_per_year: int = 252,
) -> BacktestResult:
    """Always long over the same window, one round trip of cost in total."""
    if frame is None or frame.empty or "close" not in frame.columns:
        return BacktestResult(name="buy_and_hold", metrics={"insufficient_data": True})

    close = frame["close"].astype(float).to_numpy()
    returns = np.diff(close) / close[:-1]
    if len(returns) == 0:
        return BacktestResult(name="buy_and_hold", metrics={"insufficient_data": True})
    returns = returns.copy()
    returns[0] -= round_trip_cost  # one entry + exit charged up front

    metrics = _summarise(returns, bars_per_session * sessions_per_year, 1, "buy_and_hold")
    equity = pd.DataFrame(
        {
            "bar_start": frame["bar_start"].to_numpy()[1:],
            "period_return": returns,
            "equity": np.cumprod(1.0 + returns),
        }
    )
    return BacktestResult(name="buy_and_hold", metrics=metrics, equity_curve=equity)


def baseline_momentum(
    frame: pd.DataFrame,
    round_trip_cost: float,
    momentum_column: str = "momentum_12",
    bars_per_session: int = 39,
    sessions_per_year: int = 252,
) -> BacktestResult:
    """Long while trailing momentum is positive, flat otherwise.

    The signal is shifted by one bar before it is applied, so the position for
    bar *t* is decided using information available at *t-1*.
    """
    if frame is None or frame.empty or momentum_column not in frame.columns:
        return BacktestResult(name="momentum", metrics={"insufficient_data": True})

    df = frame.sort_values("bar_start").reset_index(drop=True)
    close = df["close"].astype(float)
    bar_return = close.pct_change().fillna(0.0)
    position = (df[momentum_column].astype(float) > 0).astype(float).shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strategy = position * bar_return - turnover * (round_trip_cost / 2.0)

    returns = strategy.to_numpy(dtype=float)[1:]
    n_trades = int((turnover > 0).sum())
    metrics = _summarise(returns, bars_per_session * sessions_per_year, n_trades, "momentum")
    equity = pd.DataFrame(
        {
            "bar_start": df["bar_start"].to_numpy()[1:],
            "period_return": returns,
            "equity": np.cumprod(1.0 + returns),
        }
    )
    return BacktestResult(name="momentum", metrics=metrics, equity_curve=equity)


def compare_to_baselines(
    model_result: BacktestResult,
    baselines: dict[str, BacktestResult],
) -> dict[str, Any]:
    """Flat comparison dict plus an honest verdict string."""
    model_metrics = model_result.metrics
    out: dict[str, Any] = {
        "model": model_metrics,
        "baselines": {name: res.metrics for name, res in baselines.items()},
    }

    if model_metrics.get("insufficient_data"):
        out["verdict"] = "INSUFFICIENT_EVIDENCE: no out-of-sample predictions available."
        out["beats_baselines"] = False
        return out

    model_total = model_metrics.get("total_return")
    beats = []
    for name, res in baselines.items():
        base_total = res.metrics.get("total_return")
        if base_total is None or model_total is None:
            continue
        beats.append(model_total > base_total)

    out["beats_baselines"] = bool(beats) and all(beats)
    if not beats:
        out["verdict"] = "INSUFFICIENT_EVIDENCE: baselines could not be computed."
    elif out["beats_baselines"] and model_metrics.get("n_trades", 0) >= 20:
        out["verdict"] = (
            "Model beat both baselines after costs on out-of-sample folds. "
            "Sample is small; this is not evidence of profitability."
        )
    else:
        out["verdict"] = (
            "Model did NOT beat both baselines after costs on out-of-sample folds."
        )
    return out
