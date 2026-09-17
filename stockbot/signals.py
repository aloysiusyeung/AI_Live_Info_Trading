"""Signal generation.

Turns a model's probability into one of four labels plus a plain-English
explanation. This layer is advisory only — it never sizes or places an order;
that is the risk engine's job.

  BUY                   high-confidence positive edge after estimated costs
  HOLD                  a position exists but there is no fresh edge
  AVOID                 the model expects a negative edge after costs
  INSUFFICIENT_EVIDENCE no usable model, bad data, or confidence below floor
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import Settings
from .features import FEATURE_DESCRIPTIONS
from .models.trainer import feature_contributions

logger = logging.getLogger(__name__)

BUY = "BUY"
HOLD = "HOLD"
AVOID = "AVOID"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass
class Signal:
    symbol: str
    bar_start: datetime | None
    signal: str
    confidence: float | None = None
    prob_up: float | None = None
    expected_return: float | None = None
    uncertainty: float | None = None
    last_price: float | None = None
    model_name: str | None = None
    model_version_id: int | None = None
    horizon_bars: int = 0
    horizon_label: str = ""
    top_features: list[tuple[str, float]] = field(default_factory=list)
    contribution_kind: str = "none"
    explanation: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bar_start": self.bar_start.isoformat() if self.bar_start else None,
            "signal": self.signal,
            "confidence": self.confidence,
            "prob_up": self.prob_up,
            "expected_return": self.expected_return,
            "uncertainty": self.uncertainty,
            "last_price": self.last_price,
            "model_name": self.model_name,
            "model_version_id": self.model_version_id,
            "horizon_bars": self.horizon_bars,
            "horizon_label": self.horizon_label,
            "top_features": self.top_features,
            "contribution_kind": self.contribution_kind,
            "explanation": self.explanation,
            "diagnostics": self.diagnostics,
        }


class SignalGenerator:
    """Converts model output into a labelled, explained signal."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def insufficient(self, symbol: str, reason: str, bar_start: datetime | None = None,
                     last_price: float | None = None) -> Signal:
        return Signal(
            symbol=symbol,
            bar_start=bar_start,
            signal=INSUFFICIENT_EVIDENCE,
            last_price=last_price,
            horizon_bars=self.settings.prediction_horizon_bars,
            horizon_label=self.settings.horizon_label,
            explanation=f"No signal: {reason}.",
            diagnostics={"reason": reason},
        )

    def generate(
        self,
        symbol: str,
        bundle: dict,
        feature_row: pd.Series,
        has_position: bool = False,
        recent_volatility: float | None = None,
    ) -> Signal:
        """Score one feature row with the selected model."""
        pipeline = bundle["pipeline"]
        feature_names: list[str] = list(bundle["feature_names"])
        values = feature_row[feature_names].to_numpy(dtype=float)

        if np.isnan(values).any():
            missing = [n for n, v in zip(feature_names, values) if np.isnan(v)]
            return self.insufficient(
                symbol,
                f"missing features ({', '.join(missing[:3])})",
                bar_start=feature_row.get("bar_start"),
                last_price=float(feature_row.get("close", np.nan)),
            )

        prob_up = float(pipeline.predict_proba(values.reshape(1, -1))[0, 1])
        uncertainty = self._uncertainty(bundle, values)
        expected_return = self._expected_return(prob_up, bundle, recent_volatility)

        oos = (bundle.get("oos_metrics") or {}).get("walkforward", {})
        contributions = feature_contributions(pipeline, feature_names, values)
        kind = "signed" if hasattr(pipeline.named_steps["model"], "coef_") else "importance"

        signal = Signal(
            symbol=symbol,
            bar_start=feature_row.get("bar_start"),
            signal=INSUFFICIENT_EVIDENCE,
            confidence=max(prob_up, 1.0 - prob_up),
            prob_up=prob_up,
            expected_return=expected_return,
            uncertainty=uncertainty,
            last_price=float(feature_row.get("close", np.nan)),
            model_name=bundle.get("model_name"),
            model_version_id=bundle.get("version_id"),
            horizon_bars=self.settings.prediction_horizon_bars,
            horizon_label=self.settings.horizon_label,
            top_features=contributions[:5],
            contribution_kind=kind,
            diagnostics={
                "oos_roc_auc": oos.get("roc_auc"),
                "oos_n_trades": oos.get("n_trades"),
                "oos_net_return_per_trade": oos.get("net_return_per_trade"),
                "beats_baselines": (bundle.get("oos_metrics") or {}).get("beats_baselines"),
                "round_trip_cost_bps": self.settings.round_trip_cost_bps,
            },
        )
        signal.signal = self._classify(signal, has_position)
        signal.explanation = self._explain(signal, has_position)
        return signal

    # -- internals ---------------------------------------------------------
    def _classify(self, signal: Signal, has_position: bool) -> str:
        # No uncertainty estimate means no evidence about reliability.
        if signal.uncertainty is None or signal.uncertainty > self.settings.uncertainty_max:
            return INSUFFICIENT_EVIDENCE
        prob = signal.prob_up or 0.0
        if prob >= self.settings.min_confidence:
            # The edge must survive the estimated round trip.
            if (signal.expected_return or 0.0) <= 0:
                return HOLD if has_position else INSUFFICIENT_EVIDENCE
            return BUY
        if prob <= 1.0 - self.settings.min_confidence:
            return AVOID
        return HOLD if has_position else INSUFFICIENT_EVIDENCE

    def _uncertainty(self, bundle: dict, values: np.ndarray) -> float | None:
        """Spread of P(up) across the model's members — epistemic uncertainty.

        This is deliberately the *same* quantity for every model family, so a
        single threshold is meaningful:

        * tree ensembles use their native member estimators;
        * a single-estimator model uses the bootstrap refits stored with it by
          the trainer.

        Returns ``None`` when neither is available, which the classifier treats
        as insufficient evidence rather than as low uncertainty.
        """
        pipeline = bundle["pipeline"]
        row = values.reshape(1, -1)

        bootstraps = bundle.get("bootstrap_pipelines") or []
        if len(bootstraps) > 1:
            probs = []
            for member in bootstraps:
                try:
                    probs.append(float(member.predict_proba(row)[0, 1]))
                except Exception:  # noqa: BLE001
                    continue
            if len(probs) > 1:
                return float(np.std(probs))

        model = pipeline.named_steps.get("model")
        estimators = getattr(model, "estimators_", None)
        if estimators is not None and len(estimators) > 1:
            transformed = row
            for _, step in pipeline.steps[:-1]:
                transformed = step.transform(transformed)
            member_probs: list[float] = []
            for est in np.asarray(estimators).ravel():
                try:
                    member_probs.append(float(est.predict_proba(transformed)[0, 1]))
                except (AttributeError, IndexError, ValueError):
                    continue
            if len(member_probs) > 1:
                return float(np.std(member_probs))
        return None

    def _expected_return(
        self, prob_up: float, bundle: dict, recent_volatility: float | None
    ) -> float:
        """Expected net return over the horizon.

        Uses the realised average gain/loss magnitude from out-of-sample folds
        when available, so the number is grounded in measured moves rather than
        an assumed payoff. Falls back to recent realised volatility scaled to
        the horizon.
        """
        oos = (bundle.get("oos_metrics") or {}).get("walkforward", {})
        move = oos.get("net_return_per_trade")
        if move is None or not np.isfinite(move) or move == 0:
            if recent_volatility and np.isfinite(recent_volatility):
                move = float(recent_volatility) * np.sqrt(self.settings.prediction_horizon_bars)
            else:
                move = 0.0
        magnitude = abs(float(move))
        edge = (prob_up * magnitude) - ((1.0 - prob_up) * magnitude)
        return float(edge - self.settings.round_trip_cost)

    def _explain(self, signal: Signal, has_position: bool) -> str:
        horizon = signal.horizon_label or f"{signal.horizon_bars} bars"
        prob_pct = (signal.prob_up or 0.0) * 100
        exp_bps = (signal.expected_return or 0.0) * 10_000

        drivers = ", ".join(
            FEATURE_DESCRIPTIONS.get(name, name) for name, _ in signal.top_features[:3]
        )
        driver_clause = f" Main contributors: {drivers}." if drivers else ""
        cost_clause = (
            f" Estimated round-trip cost of {self.settings.round_trip_cost_bps:.0f} bps "
            "is already deducted."
        )

        if signal.signal == BUY:
            body = (
                f"{signal.symbol}: model puts a {prob_pct:.0f}% probability on a positive "
                f"return over the {horizon} horizon, an expected {exp_bps:+.0f} bps after costs."
            )
        elif signal.signal == AVOID:
            body = (
                f"{signal.symbol}: model puts only a {prob_pct:.0f}% probability on a positive "
                f"return over the {horizon} horizon, so the expected edge is negative."
            )
        elif signal.signal == HOLD:
            state = "an existing position" if has_position else "no position"
            body = (
                f"{signal.symbol}: at {prob_pct:.0f}% probability the edge is not clear enough to "
                f"act on over the {horizon} horizon; with {state} the recommendation is to hold."
            )
        else:
            body = (
                f"{signal.symbol}: evidence is insufficient — probability {prob_pct:.0f}% is below "
                f"the {self.settings.min_confidence * 100:.0f}% confidence floor or the model's "
                "spread of opinion is too wide."
            )
        return body + driver_clause + cost_clause


def top_feature_summary(
    contributions: Sequence[tuple[str, float]], limit: int = 5
) -> list[dict[str, Any]]:
    """Feature contributions annotated with human-readable descriptions."""
    return [
        {
            "feature": name,
            "description": FEATURE_DESCRIPTIONS.get(name, name),
            "contribution": value,
            "direction": "positive" if value > 0 else "negative",
        }
        for name, value in list(contributions)[:limit]
    ]
