"""Chronological walk-forward validation.

Financial time series are never shuffled here. Splits are expanding-window and
strictly ordered in time, and an **embargo** of ``horizon_bars`` rows sits
between each training block and its test block so that a training row's forward
window cannot overlap the test period. Without the embargo, the label of the
last training rows would be computed from prices that also appear in the test
set — a subtle but real form of leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_start: Any
    train_end: Any
    test_start: Any
    test_end: Any
    n_train: int
    n_test: int
    accuracy: float | None
    roc_auc: float | None
    brier: float | None
    net_return: float
    trades: int
    hit_rate: float | None


@dataclass
class WalkForwardResult:
    model_name: str
    folds: list[FoldResult] = field(default_factory=list)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "n_folds": len(self.folds),
            "metrics": self.metrics,
            "folds": [f.__dict__ for f in self.folds],
        }


def expanding_splits(
    n_rows: int, n_splits: int, embargo: int, min_train: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) pairs ordered in time.

    Each test block immediately follows its training block, separated by an
    ``embargo`` gap. Training blocks expand; test blocks are equal-sized.
    """
    if n_rows <= 0 or n_splits <= 0:
        return
    usable = n_rows - min_train - embargo
    if usable <= 0:
        return
    test_size = usable // n_splits
    if test_size <= 0:
        return

    for fold in range(n_splits):
        train_end = min_train + fold * test_size
        test_start = train_end + embargo
        test_end = test_start + test_size
        if fold == n_splits - 1:
            test_end = n_rows
        if test_start >= n_rows or test_end <= test_start:
            break
        yield np.arange(0, train_end), np.arange(test_start, min(test_end, n_rows))


def _strategy_net_return(
    probabilities: np.ndarray,
    forward_returns: np.ndarray,
    threshold: float,
    round_trip_cost: float,
) -> tuple[float, int, float | None]:
    """Mean per-decision net return for a long-only threshold rule.

    Returns ``(mean_net_return_per_bar, n_trades, hit_rate)``. A bar with no
    trade contributes zero, which is what a flat position actually earns.
    """
    take = probabilities >= threshold
    n_trades = int(take.sum())
    if n_trades == 0:
        return 0.0, 0, None
    realised = forward_returns[take] - round_trip_cost
    hit_rate = float((realised > 0).mean())
    total = float(realised.sum())
    return total / len(probabilities), n_trades, hit_rate


def walk_forward_evaluate(
    pipeline,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    model_name: str,
    n_splits: int = 5,
    embargo: int = 39,
    min_train: int = 750,
    decision_threshold: float = 0.55,
    round_trip_cost: float = 0.001,
) -> WalkForwardResult:
    """Run expanding-window validation and collect out-of-sample predictions.

    The pipeline (imputer + scaler + estimator) is cloned and refitted from
    scratch on every training block, so no statistic from a test block ever
    informs the transforms applied to it.
    """
    result = WalkForwardResult(model_name=model_name)
    features = frame[list(feature_columns)].to_numpy(dtype=float)
    target = frame["target"].to_numpy(dtype=float)
    forward = frame["forward_return"].to_numpy(dtype=float)
    timestamps = frame["bar_start"].to_numpy()

    rows: list[dict] = []
    for fold_idx, (train_idx, test_idx) in enumerate(
        expanding_splits(len(frame), n_splits, embargo, min_train)
    ):
        y_train = target[train_idx]
        if len(np.unique(y_train)) < 2:
            logger.warning(
                "Skipping fold with a single class",
                extra={"model": model_name, "fold": fold_idx},
            )
            continue

        estimator = clone(pipeline)
        estimator.fit(features[train_idx], y_train)
        proba = estimator.predict_proba(features[test_idx])[:, 1]
        y_test = target[test_idx]

        accuracy = float(accuracy_score(y_test, (proba >= 0.5).astype(float)))
        try:
            auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None
        except ValueError:
            auc = None
        brier = float(brier_score_loss(y_test, proba))
        net, trades, hit_rate = _strategy_net_return(
            proba, forward[test_idx], decision_threshold, round_trip_cost
        )

        result.folds.append(
            FoldResult(
                fold=fold_idx,
                train_start=timestamps[train_idx[0]],
                train_end=timestamps[train_idx[-1]],
                test_start=timestamps[test_idx[0]],
                test_end=timestamps[test_idx[-1]],
                n_train=len(train_idx),
                n_test=len(test_idx),
                accuracy=accuracy,
                roc_auc=auc,
                brier=brier,
                net_return=net,
                trades=trades,
                hit_rate=hit_rate,
            )
        )
        for i, idx in enumerate(test_idx):
            rows.append(
                {
                    "fold": fold_idx,
                    "bar_start": timestamps[idx],
                    "prob_up": float(proba[i]),
                    "target": float(target[idx]),
                    "forward_return": float(forward[idx]),
                }
            )

    result.predictions = pd.DataFrame(rows)
    result.metrics = _aggregate(result, decision_threshold, round_trip_cost)
    return result


def _aggregate(
    result: WalkForwardResult, decision_threshold: float, round_trip_cost: float
) -> dict[str, Any]:
    if result.predictions.empty:
        return {
            "n_folds": 0,
            "n_test_rows": 0,
            "insufficient_data": True,
        }

    preds = result.predictions
    proba = preds["prob_up"].to_numpy(dtype=float)
    y = preds["target"].to_numpy(dtype=float)
    forward = preds["forward_return"].to_numpy(dtype=float)

    net, trades, hit_rate = _strategy_net_return(
        proba, forward, decision_threshold, round_trip_cost
    )
    per_trade = None
    if trades:
        per_trade = float((forward[proba >= decision_threshold] - round_trip_cost).mean())

    try:
        auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else None
    except ValueError:
        auc = None

    fold_nets = [f.net_return for f in result.folds]
    return {
        "n_folds": len(result.folds),
        "n_test_rows": int(len(preds)),
        "accuracy": float(accuracy_score(y, (proba >= 0.5).astype(float))),
        "roc_auc": auc,
        "brier": float(brier_score_loss(y, proba)),
        "base_rate": float(y.mean()),
        "decision_threshold": decision_threshold,
        "cost_bps": round_trip_cost * 10_000,
        "net_return_per_bar": net,
        "net_return_per_trade": per_trade,
        "n_trades": trades,
        "trade_rate": float(trades / len(preds)),
        "hit_rate": hit_rate,
        "fold_net_returns": fold_nets,
        "fold_net_return_std": float(np.std(fold_nets)) if fold_nets else None,
        "positive_folds": int(sum(1 for n in fold_nets if n > 0)),
        "insufficient_data": False,
    }
