"""Dashboard smoke tests.

The Streamlit script is executed headlessly with ``AppTest``, so a broken
widget, a bad column reference or an unhandled exception fails the test suite
rather than only appearing in a browser. Alpaca is stubbed, so no network call
is made.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from dashboard import data_access as da

# AppTest resolves a relative path against the *calling* file, so be explicit.
APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Point the dashboard at a temporary database and stub the broker."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "dash.sqlite"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "l"))
    monkeypatch.setenv("WATCHLIST", "AAPL")

    import stockbot.alpaca_client as ac

    class _Unreachable:
        def __init__(self, *_a, **_k):
            raise ac.AlpacaError("stubbed: no network in tests")

    monkeypatch.setattr(ac, "TradingClient", _Unreachable)
    monkeypatch.setattr(ac, "StockHistoricalDataClient", _Unreachable)

    # AppTest runs in-process, so Streamlit's resource cache would otherwise
    # carry the previous test's Settings (and therefore its database path)
    # into this one.
    import streamlit as st

    st.cache_resource.clear()
    st.cache_data.clear()

    from stockbot.config import load_settings
    from stockbot.db import Database

    settings = load_settings()
    return settings, Database(settings.database_path)


def _run(timeout: float = 120):
    from streamlit.testing.v1 import AppTest

    harness = AppTest.from_file(APP, default_timeout=timeout)
    harness.run()
    return harness


def test_dashboard_renders_on_an_empty_database(app_env):
    harness = _run()
    assert not harness.exception


def test_paper_trading_label_is_shown(app_env):
    harness = _run()
    text = " ".join(str(m.value) for m in harness.markdown)
    assert "PAPER TRADING" in text


def test_disconnected_broker_is_reported_not_faked(app_env):
    harness = _run()
    errors = " ".join(str(e.value) for e in harness.error)
    assert "Alpaca: disconnected" in errors


def test_dashboard_renders_with_data(app_env, bars):
    settings, db = app_env
    db.upsert_bars(bars.assign(symbol="AAPL").to_dict("records"))
    db.upsert_bars(bars.assign(symbol="SPY").to_dict("records"))
    db.save_signal(
        symbol="AAPL",
        bar_start=datetime.now(timezone.utc),
        signal="BUY",
        confidence=0.71,
        expected_return=0.004,
        uncertainty=0.05,
        last_price=101.0,
        explanation="AAPL: synthetic test signal.",
        risk_decision="NO_ACTION",
        risk_reasons=["paper_orders_disabled"],
    )
    db.save_prediction(
        symbol="AAPL",
        bar_start=datetime.now(timezone.utc),
        model_name="logistic_regression",
        horizon_bars=39,
        prob_up=0.71,
        expected_return=0.004,
        uncertainty=0.05,
        top_features=[["rsi_14", -0.3], ["macd", 0.2]],
    )
    db.record_order(
        client_order_id="dash-1", symbol="AAPL", side="buy", qty=3, status="filled",
        filled_qty=3, filled_avg_price=101.0,
        risk_snapshot={"decision": "ALLOW"}, prediction_snapshot={"prob_up": 0.71},
    )
    for equity in (100_000.0, 100_250.0):
        db.snapshot_account({
            "equity": equity, "last_equity": 100_000.0, "cash": equity,
            "buying_power": equity * 2, "long_market_value": 0.0, "daytrade_count": 0,
        })
    db.snapshot_positions([{
        "symbol": "AAPL", "qty": 3.0, "avg_entry_price": 100.0, "market_value": 303.0,
        "cost_basis": 300.0, "unrealized_pl": 3.0, "unrealized_plpc": 0.01,
        "current_price": 101.0,
    }])
    db.save_backtest_run(
        symbol="AAPL", model_name="logistic_regression", horizon_bars=39,
        metrics={"total_return": 0.01, "n_trades": 22, "sharpe": 0.4},
        baselines={"buy_and_hold": {"total_return": 0.02}},
        cost_bps=10.0, notes="synthetic",
    )

    harness = _run()
    assert not harness.exception
    text = " ".join(str(m.value) for m in harness.markdown)
    assert "AAPL" in text


def test_kill_switch_state_is_reflected(app_env):
    _settings, db = app_env
    db.set_state("kill_switch", True)
    harness = _run()
    errors = " ".join(str(e.value) for e in harness.error)
    assert "Kill switch: ENGAGED" in errors


# -- data_access unit tests (no Streamlit runtime needed) --------------------
def test_age_text_formats_relative_times():
    now = datetime.now(timezone.utc)
    assert da.age_text(None) == "never"
    assert da.age_text((now - timedelta(seconds=30)).isoformat()).endswith("s ago")
    assert da.age_text((now - timedelta(minutes=20)).isoformat()).endswith("m ago")
    assert da.age_text((now - timedelta(hours=5)).isoformat()).endswith("h ago")
    assert da.age_text((now + timedelta(minutes=8)).isoformat()).startswith("in ")


def test_parse_json_is_tolerant():
    assert da.parse_json(None, []) == []
    assert da.parse_json("not json", {}) == {}
    assert da.parse_json('{"a": 1}') == {"a": 1}
    assert da.parse_json({"a": 1}) == {"a": 1}


def test_performance_reports_unavailable_without_snapshots(app_env):
    _settings, db = app_env
    perf = da.paper_performance(db)
    assert perf["available"] is False
    assert "snapshot" in perf["reason"]


def test_performance_computed_from_snapshots(app_env):
    _settings, db = app_env
    for equity in (100_000.0, 101_000.0):
        db.snapshot_account({
            "equity": equity, "last_equity": 100_000.0, "cash": equity,
            "buying_power": equity, "long_market_value": 0.0, "daytrade_count": 0,
        })
    perf = da.paper_performance(db)
    assert perf["available"] is True
    assert perf["total_return"] == pytest.approx(0.01)
