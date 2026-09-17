"""End-to-end integration over the fake broker.

These exercise the full path: bars -> validation -> features -> training ->
signal -> risk -> order -> persistence. All data is synthetic and generated
in-process; nothing here is a claim about real market behaviour.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from stockbot.engine import TradingEngine
from stockbot.models.trainer import ModelTrainer
from stockbot.signals import INSUFFICIENT_EVIDENCE
from tests.conftest import synthetic_bars
from tests.fakes import FakeAlpacaClient


def _bar_rows(symbol: str, n_sessions: int, seed: int, drift: float = 0.0) -> list[dict]:
    """Synthetic bars ending a few minutes ago so they count as fresh."""
    frame = synthetic_bars(
        symbol=symbol, n_sessions=n_sessions, seed=seed, drift=drift,
        end_date=datetime.now(timezone.utc),
    )
    shift = datetime.now(timezone.utc) - timedelta(minutes=15) - frame["bar_start"].iloc[-1]
    frame["bar_start"] = frame["bar_start"] + shift
    return frame.to_dict("records")


@pytest.fixture
def wired(tmp_settings, db, monkeypatch):
    monkeypatch.setattr(tmp_settings, "watchlist", ["AAPL"])
    monkeypatch.setattr(tmp_settings, "min_training_rows", 400)
    bars = {
        "AAPL": _bar_rows("AAPL", 40, seed=11),
        "SPY": _bar_rows("SPY", 40, seed=29),
    }
    client = FakeAlpacaClient(tmp_settings, bars=bars, market_open=True)
    return tmp_settings, db, client, TradingEngine(tmp_settings, client, db)


def test_backfill_stores_bars(wired):
    _settings, db, _client, engine = wired
    counts = engine.collector.backfill()
    assert counts["AAPL"] > 1000
    assert db.get_bars("AAPL", 10)


def test_full_cycle_runs_and_persists(wired):
    _settings, db, _client, engine = wired
    engine.collector.backfill()
    engine.ensure_models()
    result = engine.run_cycle()

    assert result.status == "OK"
    assert result.market_open is True
    assert len(result.outcomes) == 1
    assert db.latest_signals()
    assert db.recent_scheduler_runs(1)[0]["status"] == "OK"


def test_cycle_skips_when_market_closed(tmp_settings, db, monkeypatch):
    monkeypatch.setattr(tmp_settings, "watchlist", ["AAPL"])
    bars = {"AAPL": _bar_rows("AAPL", 10, seed=3), "SPY": _bar_rows("SPY", 10, seed=4)}
    client = FakeAlpacaClient(tmp_settings, bars=bars, market_open=False)
    result = TradingEngine(tmp_settings, client, db).run_cycle()

    assert result.status == "SKIPPED"
    assert result.reason == "market_closed"
    assert client.submitted == []


def test_no_orders_while_disabled_even_on_a_buy(wired):
    settings, db, client, engine = wired
    engine.collector.backfill()
    engine.ensure_models()
    engine.run_cycle()
    assert settings.enable_paper_orders is False
    assert client.submitted == []


def test_no_model_yields_insufficient_evidence(wired, monkeypatch):
    """A symbol with no validated model must not produce a directional signal."""
    settings, db, _client, engine = wired
    # The cycle self-provisions models, so deny it any training budget to
    # reproduce the "no model yet" state a freshly admitted symbol is in.
    monkeypatch.setattr(settings, "max_trainings_per_cycle", 0)
    engine.collector.backfill()
    result = engine.run_cycle()
    assert result.outcomes[0].signal.signal == INSUFFICIENT_EVIDENCE
    assert result.training["AAPL"]["status"] == "deferred"


def test_training_budget_limits_work_per_cycle(wired, monkeypatch):
    """A freshly widened universe must not overrun the cycle training models."""
    settings, db, _client, engine = wired
    monkeypatch.setattr(settings, "watchlist", ["AAPL", "TSLA"])
    monkeypatch.setattr(settings, "max_trainings_per_cycle", 1)
    engine.collector.backfill()
    report = engine.ensure_models(symbols=["AAPL", "TSLA"], budget=1)
    statuses = [r["status"] for r in report.values()]
    assert statuses.count("deferred") == 1


def test_stale_data_yields_insufficient_evidence(tmp_settings, db, monkeypatch):
    monkeypatch.setattr(tmp_settings, "watchlist", ["AAPL"])
    old = synthetic_bars("AAPL", n_sessions=20,
                         end_date=datetime.now(timezone.utc) - timedelta(days=30))
    client = FakeAlpacaClient(
        tmp_settings, bars={"AAPL": old.to_dict("records"), "SPY": []}, market_open=True
    )
    engine = TradingEngine(tmp_settings, client, db)
    engine.collector.backfill()
    result = engine.run_cycle()
    assert result.outcomes[0].signal.signal == INSUFFICIENT_EVIDENCE


def test_kill_switch_stops_orders_end_to_end(wired, monkeypatch):
    settings, db, client, engine = wired
    monkeypatch.setattr(settings, "enable_paper_orders", True)
    engine.risk.engage_kill_switch(actor="test")
    engine.collector.backfill()
    engine.ensure_models()
    engine.run_cycle()
    assert client.submitted == []


def test_training_records_backtests_and_baselines(wired):
    _settings, db, _client, engine = wired
    engine.collector.backfill()
    engine.ensure_models()

    runs = db.query("SELECT * FROM backtest_runs")
    assert len(runs) >= 3      # one per candidate model
    assert all(run["baselines"] for run in runs)


def test_trainer_reports_insufficient_history_rather_than_guessing(tmp_settings, db):
    trainer = ModelTrainer(tmp_settings, db)
    short = synthetic_bars(n_sessions=3)
    outcome = trainer.train_symbol("AAPL", short, None, persist=False)
    assert outcome.selected_model is None
    assert "insufficient_history" in outcome.reason


def test_trainer_may_select_no_model_on_noise(tmp_settings, db, monkeypatch):
    """Pure noise should usually fail the selection guards.

    Whatever the outcome, the guards must hold: a selected model has to have
    cleared the minimum trade count and AUC floor.
    """
    monkeypatch.setattr(tmp_settings, "min_training_rows", 400)
    trainer = ModelTrainer(tmp_settings, db)
    noise = synthetic_bars(n_sessions=40, seed=99)
    outcome = trainer.train_symbol("AAPL", noise, None, persist=False)

    if outcome.selected_model is None:
        assert outcome.reason
    else:
        wf = outcome.candidates[outcome.selected_model]["walkforward"]
        assert wf["n_trades"] >= 15
        assert wf["roc_auc"] >= 0.52


def test_cycle_survives_a_broken_symbol(wired, monkeypatch):
    _settings, db, _client, engine = wired
    engine.collector.backfill()
    engine.ensure_models()

    def explode(*_args, **_kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(engine.collector, "load_validated", explode)
    result = engine.run_cycle()
    assert result.outcomes[0].error is not None
    assert db.recent_errors()


def test_order_path_when_enabled_and_approved(wired, monkeypatch):
    """Force an approved decision and confirm the order actually reaches the broker."""
    settings, db, client, engine = wired
    monkeypatch.setattr(settings, "enable_paper_orders", True)
    engine.collector.backfill()
    engine.ensure_models()

    from stockbot.risk import ALLOW, RiskDecision

    def always_allow(signal, context):
        decision = RiskDecision(symbol=signal.symbol)
        decision.decision = ALLOW
        decision.target_qty = 5
        decision.target_notional = 5 * (signal.last_price or 100.0)
        decision.reasons.append("forced_for_test")
        return decision

    monkeypatch.setattr(engine.risk, "evaluate", always_allow)
    result = engine.run_cycle()

    assert result.orders_submitted == 1
    assert len(client.submitted) == 1
    stored = db.recent_orders("AAPL")[0]
    assert stored["qty"] == 5
    assert stored["paper"] == 1


def test_repeat_cycle_does_not_duplicate_signal_rows(wired):
    _settings, db, _client, engine = wired
    engine.collector.backfill()
    engine.ensure_models()
    engine.run_cycle()
    engine.run_cycle()
    rows = db.query("SELECT * FROM signals WHERE symbol='AAPL'")
    assert len(rows) == 1
