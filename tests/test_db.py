"""Persistence layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from stockbot.db import Database


def test_schema_created(db):
    tables = {
        row["name"]
        for row in db.query("SELECT name FROM sqlite_master WHERE type='table'")
    }
    expected = {
        "bars", "features", "predictions", "signals", "model_versions",
        "backtest_runs", "paper_orders", "positions", "errors", "scheduler_runs",
        "account_snapshots", "app_state",
    }
    assert expected <= tables


def test_bar_upsert_is_idempotent(db, bars):
    rows = bars.to_dict("records")
    db.upsert_bars(rows)
    db.upsert_bars(rows)
    stored = db.get_bars("TEST", 10)
    assert len(stored) == len(rows)


def test_bar_upsert_overwrites_a_revised_bar(db, bars):
    rows = bars.to_dict("records")
    db.upsert_bars(rows)
    revised = dict(rows[-1])
    revised["close"] = 999.0
    db.upsert_bars([revised])
    assert db.get_bars("TEST", 10)[-1]["close"] == 999.0


def test_latest_bar_start(db, bars):
    db.upsert_bars(bars.to_dict("records"))
    latest = db.latest_bar_start("TEST", 10)
    assert latest is not None
    assert str(bars["bar_start"].iloc[-1].year) in latest


def test_signal_upsert_is_unique_per_bar(db):
    stamp = datetime.now(timezone.utc)
    for label in ("HOLD", "BUY"):
        db.save_signal(symbol="AAPL", bar_start=stamp, signal=label, confidence=0.6)
    rows = db.query("SELECT * FROM signals WHERE symbol='AAPL'")
    assert len(rows) == 1
    assert rows[0]["signal"] == "BUY"


def test_order_lifecycle_recorded(db):
    db.record_order(
        client_order_id="abc-1", symbol="AAPL", side="buy", qty=5, status="new",
        risk_snapshot={"decision": "ALLOW"}, prediction_snapshot={"prob_up": 0.7},
    )
    db.update_order_status("abc-1", status="partially_filled", filled_qty=2.0)
    db.update_order_status("abc-1", status="filled", filled_qty=5.0, filled_avg_price=101.0)

    order = db.recent_orders("AAPL")[0]
    assert order["status"] == "filled"
    assert order["filled_qty"] == 5.0
    assert "ALLOW" in order["risk_snapshot"]


def test_duplicate_client_order_id_rejected(db):
    db.record_order(client_order_id="dup", symbol="AAPL", side="buy", qty=1, status="new")
    with pytest.raises(Exception):
        db.record_order(client_order_id="dup", symbol="AAPL", side="buy", qty=1, status="new")


def test_orders_since_filters_by_time(db):
    db.record_order(client_order_id="o-1", symbol="AAPL", side="buy", qty=1, status="new")
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert db.orders_since("AAPL", past)
    assert not db.orders_since("AAPL", future)


def test_model_selection_is_exclusive(db):
    ids = [
        db.save_model_version(
            symbol="AAPL", model_name=name, horizon_bars=39, feature_names=["a"],
            train_rows=100, oos_metrics={}, baseline_metrics={}, selected=True,
        )
        for name in ("logistic_regression", "random_forest")
    ]
    db.mark_selected_model("AAPL", ids[1])
    selected = db.query("SELECT * FROM model_versions WHERE symbol='AAPL' AND selected=1")
    assert len(selected) == 1
    assert selected[0]["id"] == ids[1]


def test_app_state_round_trip(db):
    db.set_state("kill_switch", True)
    assert db.get_state("kill_switch") is True
    db.set_state("kill_switch", False)
    assert db.get_state("kill_switch") is False
    assert db.get_state("never_set", "fallback") == "fallback"


def test_scheduler_run_lifecycle(db):
    run_id = db.start_scheduler_run(reason="test")
    db.finish_scheduler_run(run_id, "OK", symbols_processed=3, signals_generated=3)
    run = db.recent_scheduler_runs(1)[0]
    assert run["status"] == "OK"
    assert run["symbols_processed"] == 3


def test_errors_recorded(db):
    db.log_error("test", "something failed", symbol="AAPL", detail={"k": "v"})
    errors = db.recent_errors()
    assert errors[0]["message"] == "something failed"


def test_read_only_database_cannot_write(tmp_settings, db):
    read_only = Database(tmp_settings.database_path, read_only=True)
    assert read_only.query("SELECT 1 AS one")[0]["one"] == 1
    with pytest.raises(Exception):
        read_only.execute("INSERT INTO app_state VALUES ('a','b','c')")
