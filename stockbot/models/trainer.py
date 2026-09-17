"""Training orchestration and model selection.

Selection rule: the candidate with the best **out-of-sample net return per
trade** after costs, subject to guard conditions (enough trades, enough folds,
AUC above chance). If no candidate clears the guards, ``selected`` is ``None``
and the signal layer emits INSUFFICIENT_EVIDENCE rather than guessing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from ..backtest import (
    backtest_predictions,
    baseline_buy_and_hold,
    baseline_momentum,
    compare_to_baselines,
)
from ..config import Settings
from ..db import Database
from ..features import FEATURE_COLUMNS, build_features, prepare_training_frame
from ..labeling import build_supervised_frame
from .registry import build_candidates
from .walkforward import WalkForwardResult, walk_forward_evaluate

logger = logging.getLogger(__name__)

#: Minimum out-of-sample trades before a candidate can be selected at all.
MIN_OOS_TRADES = 15
#: Minimum ROC AUC. 0.5 is coin-flipping; require a small margin above it.
MIN_OOS_AUC = 0.52
#: Bootstrap refits used to estimate uncertainty for single-estimator models.
N_BOOTSTRAP = 15


@dataclass
class TrainingOutcome:
    symbol: str
    trained_at: datetime
    feature_names: list[str] = field(default_factory=list)
    n_rows: int = 0
    candidates: dict[str, dict] = field(default_factory=dict)
    baselines: dict[str, dict] = field(default_factory=dict)
    selected_model: str | None = None
    selected_version_id: int | None = None
    artifact_path: str | None = None
    reason: str = ""
    comparison: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trained_at": self.trained_at.isoformat(),
            "n_rows": self.n_rows,
            "feature_names": self.feature_names,
            "candidates": self.candidates,
            "baselines": self.baselines,
            "selected_model": self.selected_model,
            "selected_version_id": self.selected_version_id,
            "reason": self.reason,
            "comparison": self.comparison,
        }


class ModelTrainer:
    """Builds, validates and persists per-symbol models."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.model_dir = Path(settings.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # -- data preparation --------------------------------------------------
    def build_supervised(
        self,
        bars: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        session_minutes: dict | None = None,
        news_index=None,
        symbol: str | None = None,
    ) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
        """Return (supervised frame, usable feature names, featured frame)."""
        featured = build_features(
            bars, benchmark, tz=self.settings.timezone, session_minutes=session_minutes,
            news_index=news_index, bar_minutes=self.settings.bar_minutes, symbol=symbol,
        )
        featured, usable = prepare_training_frame(featured, FEATURE_COLUMNS)
        supervised = build_supervised_frame(
            featured,
            usable,
            horizon_bars=self.settings.prediction_horizon_bars,
            round_trip_cost=self.settings.round_trip_cost,
            threshold=self.settings.label_threshold_bps / 10_000.0,
        )
        return supervised, usable, featured

    # -- training ----------------------------------------------------------
    def train_symbol(
        self,
        symbol: str,
        bars: pd.DataFrame,
        benchmark: pd.DataFrame | None = None,
        persist: bool = True,
        session_minutes: dict | None = None,
        news_index=None,
    ) -> TrainingOutcome:
        outcome = TrainingOutcome(symbol=symbol, trained_at=datetime.now(timezone.utc))
        supervised, usable, featured = self.build_supervised(
            bars, benchmark, session_minutes, news_index=news_index, symbol=symbol
        )
        outcome.feature_names = usable
        outcome.n_rows = len(supervised)

        min_rows = self.settings.min_training_rows
        required = min_rows + 2 * self.settings.prediction_horizon_bars
        if len(supervised) < required:
            outcome.reason = (
                f"insufficient_history: {len(supervised)} labelled rows, "
                f"need at least {required}"
            )
            logger.warning("Training skipped", extra={"symbol": symbol, "reason": outcome.reason})
            return outcome

        decision_threshold = self.settings.min_confidence
        results: dict[str, WalkForwardResult] = {}
        for name, pipeline in build_candidates().items():
            try:
                results[name] = walk_forward_evaluate(
                    pipeline,
                    supervised,
                    usable,
                    model_name=name,
                    n_splits=self.settings.walkforward_splits,
                    embargo=self.settings.prediction_horizon_bars,
                    min_train=min_rows,
                    decision_threshold=decision_threshold,
                    round_trip_cost=self.settings.round_trip_cost,
                )
            except Exception as exc:  # noqa: BLE001 - one bad model must not kill the rest
                logger.exception("Walk-forward failed", extra={"symbol": symbol, "model": name})
                self.db.log_error("trainer", f"{name}: {exc}", symbol=symbol)

        if not results:
            outcome.reason = "all_candidates_failed"
            return outcome

        # --- baselines on the same out-of-sample window ---------------------
        oos_start = min(
            (r.predictions["bar_start"].min() for r in results.values() if not r.predictions.empty),
            default=None,
        )
        oos_frame = supervised
        if oos_start is not None:
            oos_frame = supervised[supervised["bar_start"] >= oos_start].reset_index(drop=True)

        bh = baseline_buy_and_hold(
            oos_frame, self.settings.round_trip_cost, self.settings.prediction_horizon_bars
        )
        mom = baseline_momentum(oos_frame, self.settings.round_trip_cost)
        baselines = {"buy_and_hold": bh, "momentum": mom}
        outcome.baselines = {name: res.metrics for name, res in baselines.items()}

        # --- score every candidate -------------------------------------------
        scored: list[tuple[str, WalkForwardResult, dict]] = []
        for name, result in results.items():
            bt = backtest_predictions(
                result.predictions,
                decision_threshold=decision_threshold,
                round_trip_cost=self.settings.round_trip_cost,
                horizon_bars=self.settings.prediction_horizon_bars,
            )
            comparison = compare_to_baselines(bt, baselines)
            entry = {
                "walkforward": result.metrics,
                "backtest": bt.metrics,
                "beats_baselines": comparison["beats_baselines"],
                "verdict": comparison["verdict"],
            }
            outcome.candidates[name] = entry
            scored.append((name, result, entry))

            if persist:
                self.db.save_backtest_run(
                    symbol=symbol,
                    model_name=name,
                    horizon_bars=self.settings.prediction_horizon_bars,
                    start_date=str(oos_frame["bar_start"].min()) if not oos_frame.empty else None,
                    end_date=str(oos_frame["bar_start"].max()) if not oos_frame.empty else None,
                    n_folds=result.metrics.get("n_folds"),
                    n_test_rows=result.metrics.get("n_test_rows"),
                    metrics=bt.metrics,
                    baselines=outcome.baselines,
                    cost_bps=self.settings.round_trip_cost * 10_000,
                    notes=comparison["verdict"],
                )

        best = self._select(scored)
        if best is None:
            outcome.reason = (
                "no_candidate_passed_guards: none produced enough out-of-sample "
                f"trades (>= {MIN_OOS_TRADES}), AUC >= {MIN_OOS_AUC} and a "
                "positive net return after costs"
            )
            logger.warning("No model selected", extra={"symbol": symbol})
            return outcome

        name, result, entry = best
        outcome.selected_model = name
        outcome.comparison = {
            "verdict": entry["verdict"],
            "beats_baselines": entry["beats_baselines"],
        }
        outcome.reason = "selected_on_oos_net_return_after_costs"

        # --- refit on all labelled data for live inference ---------------------
        X = supervised[usable].to_numpy(dtype=float)
        y = supervised["target"].to_numpy(dtype=float)
        final = clone(build_candidates()[name])
        final.fit(X, y)
        bootstraps = _fit_bootstrap_ensemble(name, X, y)

        if persist:
            artifact = self.model_dir / f"{symbol}_{name}.joblib"
            joblib.dump(
                {
                    "pipeline": final,
                    "bootstrap_pipelines": bootstraps,
                    "feature_names": usable,
                    "symbol": symbol,
                    "model_name": name,
                    "horizon_bars": self.settings.prediction_horizon_bars,
                    "trained_at": outcome.trained_at.isoformat(),
                },
                artifact,
            )
            outcome.artifact_path = str(artifact)
            version_id = self.db.save_model_version(
                symbol=symbol,
                model_name=name,
                horizon_bars=self.settings.prediction_horizon_bars,
                feature_names=usable,
                train_rows=len(supervised),
                oos_metrics=entry,
                baseline_metrics=outcome.baselines,
                selected=True,
                artifact_path=str(artifact),
                notes=entry["verdict"],
            )
            self.db.mark_selected_model(symbol, version_id)
            outcome.selected_version_id = version_id

        logger.info(
            "Model selected",
            extra={
                "symbol": symbol,
                "model": name,
                "net_return_per_trade": entry["walkforward"].get("net_return_per_trade"),
                "roc_auc": entry["walkforward"].get("roc_auc"),
                "beats_baselines": entry["beats_baselines"],
            },
        )
        return outcome

    @staticmethod
    def _select(scored: Sequence[tuple[str, WalkForwardResult, dict]]):
        """Pick the best candidate that clears the guard conditions."""
        eligible = []
        for name, result, entry in scored:
            wf = entry["walkforward"]
            if wf.get("insufficient_data"):
                continue
            if (wf.get("n_trades") or 0) < MIN_OOS_TRADES:
                continue
            auc = wf.get("roc_auc")
            if auc is None or auc < MIN_OOS_AUC:
                continue
            per_trade = wf.get("net_return_per_trade")
            if per_trade is None or per_trade <= 0:
                continue
            eligible.append((per_trade, name, result, entry))
        if not eligible:
            return None
        eligible.sort(key=lambda t: t[0], reverse=True)
        _, name, result, entry = eligible[0]
        return name, result, entry

    # -- inference artifacts ------------------------------------------------
    def load_model(self, symbol: str) -> dict | None:
        """Load the selected model artifact for a symbol, or ``None``."""
        record = self.db.selected_model(symbol)
        if not record or not record.get("artifact_path"):
            return None
        path = Path(record["artifact_path"])
        if not path.exists():
            logger.warning("Model artifact missing", extra={"symbol": symbol, "path": str(path)})
            return None
        try:
            bundle = joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load model", extra={"symbol": symbol})
            self.db.log_error("trainer", f"model load failed: {exc}", symbol=symbol)
            return None
        bundle["version_id"] = record["id"]
        bundle["oos_metrics"] = json.loads(record["oos_metrics"])
        bundle["baseline_metrics"] = json.loads(record["baseline_metrics"] or "{}")
        return bundle


def _fit_bootstrap_ensemble(model_name: str, X: np.ndarray, y: np.ndarray) -> list:
    """Bootstrap refits used as an uncertainty estimate.

    Tree ensembles already expose member estimators, so their native spread is
    used instead and this returns an empty list. A single-estimator model such
    as logistic regression has no members, so a handful of refits on resampled
    training data gives a dispersion measure on the *same scale* as an
    ensemble's — which is what makes one uncertainty threshold meaningful
    across model families.
    """
    probe = build_candidates()[model_name].named_steps["model"]
    if hasattr(probe, "estimators_") or hasattr(probe, "n_estimators"):
        return []

    rng = np.random.default_rng(17)
    n = len(X)
    ensemble = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        member = clone(build_candidates()[model_name])
        try:
            member.fit(X[idx], y[idx])
        except Exception:  # noqa: BLE001 - a failed member just shrinks the ensemble
            continue
        ensemble.append(member)
    return ensemble


def feature_contributions(
    pipeline, feature_names: Sequence[str], values: np.ndarray
) -> list[tuple[str, float]]:
    """Per-feature contribution for one observation, best-effort.

    Linear models expose signed contributions directly. Tree ensembles only
    expose global importances, so those are reported as unsigned importance
    weights and labelled as such by the caller.
    """
    try:
        model = pipeline.named_steps["model"]
    except (AttributeError, KeyError):
        return []

    transformed = values.reshape(1, -1)
    for name, step in pipeline.steps[:-1]:
        transformed = step.transform(transformed)
    transformed = np.asarray(transformed, dtype=float).ravel()

    if hasattr(model, "coef_"):
        coefs = np.asarray(model.coef_, dtype=float).ravel()
        contributions = coefs * transformed
        pairs = list(zip(feature_names, contributions.tolist()))
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float).ravel()
        pairs = list(zip(feature_names, importances.tolist()))
    else:
        return []

    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return pairs
