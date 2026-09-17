"""Signal classification and explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from stockbot.models.registry import build_pipeline
from stockbot.signals import (
    AVOID,
    BUY,
    HOLD,
    INSUFFICIENT_EVIDENCE,
    SignalGenerator,
    top_feature_summary,
)

FEATURES = ["f0", "f1", "f2"]


def _bundle(bias: float = 0.0, net_return_per_trade: float = 0.01, bootstraps: int = 10):
    """A fitted pipeline whose output is driven by f0, plus fake OOS metadata.

    ``bootstraps`` mirrors the bootstrap refits the trainer stores alongside a
    single-estimator model; they are what the uncertainty estimate reads.
    """
    rng = np.random.default_rng(4)
    X = rng.normal(size=(600, 3))
    y = (X[:, 0] + bias > 0).astype(float)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)

    members = []
    for _ in range(bootstraps):
        idx = rng.integers(0, len(X), size=len(X))
        member = build_pipeline(LogisticRegression(max_iter=1000))
        member.fit(X[idx], y[idx])
        members.append(member)

    return {
        "pipeline": pipeline,
        "bootstrap_pipelines": members,
        "feature_names": FEATURES,
        "model_name": "logistic_regression",
        "version_id": 1,
        "oos_metrics": {
            "walkforward": {
                "roc_auc": 0.62,
                "n_trades": 40,
                "net_return_per_trade": net_return_per_trade,
            },
            "beats_baselines": True,
        },
    }


def _row(f0: float, close: float = 100.0) -> pd.Series:
    return pd.Series(
        {
            "f0": f0, "f1": 0.0, "f2": 0.0,
            "close": close,
            "bar_start": pd.Timestamp("2025-06-30 14:30", tz="UTC"),
        }
    )


def test_strong_positive_evidence_gives_buy(tmp_settings):
    signal = SignalGenerator(tmp_settings).generate("AAPL", _bundle(), _row(3.0))
    assert signal.signal == BUY
    assert signal.prob_up > tmp_settings.min_confidence
    assert signal.expected_return > 0


def test_strong_negative_evidence_gives_avoid(tmp_settings):
    signal = SignalGenerator(tmp_settings).generate("AAPL", _bundle(), _row(-3.0))
    assert signal.signal == AVOID


def test_ambiguous_evidence_without_a_position_is_insufficient(tmp_settings):
    signal = SignalGenerator(tmp_settings).generate("AAPL", _bundle(), _row(0.0))
    assert signal.signal == INSUFFICIENT_EVIDENCE


def test_ambiguous_evidence_with_a_position_is_hold(tmp_settings):
    signal = SignalGenerator(tmp_settings).generate(
        "AAPL", _bundle(), _row(0.0), has_position=True
    )
    assert signal.signal == HOLD


def test_positive_probability_with_no_edge_after_costs_is_not_a_buy(tmp_settings):
    """Confidence alone must not produce a BUY when costs eat the move."""
    bundle = _bundle(net_return_per_trade=0.0)
    generator = SignalGenerator(tmp_settings)
    signal = generator.generate("AAPL", bundle, _row(3.0))
    assert signal.expected_return <= 0
    assert signal.signal != BUY


def test_missing_features_produce_insufficient_evidence(tmp_settings):
    row = _row(1.0)
    row["f1"] = np.nan
    signal = SignalGenerator(tmp_settings).generate("AAPL", _bundle(), row)
    assert signal.signal == INSUFFICIENT_EVIDENCE
    assert "missing features" in signal.explanation


def test_no_uncertainty_estimate_means_insufficient_evidence(tmp_settings):
    """Without a dispersion estimate the model's reliability is unknown."""
    bundle = _bundle(bootstraps=0)
    signal = SignalGenerator(tmp_settings).generate("AAPL", bundle, _row(3.0))
    assert signal.uncertainty is None
    assert signal.signal == INSUFFICIENT_EVIDENCE


def test_tree_ensemble_uncertainty_uses_native_members(tmp_settings):
    """A forest needs no bootstrap refits; its own trees supply the spread."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(8)
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] > 0).astype(float)
    pipeline = build_pipeline(RandomForestClassifier(n_estimators=40, random_state=1), scale=False)
    pipeline.fit(X, y)
    bundle = {
        "pipeline": pipeline,
        "bootstrap_pipelines": [],
        "feature_names": FEATURES,
        "model_name": "random_forest",
        "version_id": 2,
        "oos_metrics": {"walkforward": {"net_return_per_trade": 0.01}},
    }
    signal = SignalGenerator(tmp_settings).generate("AAPL", bundle, _row(3.0))
    assert signal.uncertainty is not None
    assert signal.contribution_kind == "importance"


def test_high_uncertainty_downgrades_to_insufficient(tmp_settings, monkeypatch):
    monkeypatch.setattr(tmp_settings, "uncertainty_max", 0.0)
    signal = SignalGenerator(tmp_settings).generate("AAPL", _bundle(), _row(3.0))
    assert signal.signal == INSUFFICIENT_EVIDENCE


def test_explanation_is_plain_english_and_mentions_costs(tmp_settings):
    signal = SignalGenerator(tmp_settings).generate("AAPL", _bundle(), _row(3.0))
    assert "AAPL" in signal.explanation
    assert "%" in signal.explanation
    assert "cost" in signal.explanation.lower()
    assert "next trading day" in signal.explanation


def test_top_features_are_returned(tmp_settings):
    signal = SignalGenerator(tmp_settings).generate("AAPL", _bundle(), _row(3.0))
    assert signal.top_features
    assert signal.contribution_kind == "signed"
    names = [name for name, _ in signal.top_features]
    assert "f0" in names


def test_insufficient_helper_never_claims_a_number(tmp_settings):
    signal = SignalGenerator(tmp_settings).insufficient("AAPL", "no model")
    assert signal.signal == INSUFFICIENT_EVIDENCE
    assert signal.prob_up is None
    assert signal.expected_return is None


def test_signal_is_serialisable(tmp_settings):
    payload = SignalGenerator(tmp_settings).generate("AAPL", _bundle(), _row(3.0)).as_dict()
    assert payload["signal"] == BUY
    assert payload["horizon_label"] == "next trading day"


def test_confidence_is_distance_from_a_coin_flip(tmp_settings):
    signal = SignalGenerator(tmp_settings).generate("AAPL", _bundle(), _row(-3.0))
    assert signal.confidence == pytest.approx(max(signal.prob_up, 1 - signal.prob_up))


def test_top_feature_summary_annotates_descriptions():
    summary = top_feature_summary([("rsi_14", -0.4), ("macd", 0.2)])
    assert summary[0]["description"] == "RSI(14)"
    assert summary[0]["direction"] == "negative"
