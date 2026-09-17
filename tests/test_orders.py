"""Paper-order submission and reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from stockbot.orders import PaperOrderManager, make_client_order_id
from stockbot.risk import ALLOW, RiskDecision
from stockbot.signals import BUY, Signal
from tests.fakes import FakeAlpacaClient


@pytest.fixture
def signal():
    return Signal(
        symbol="AAPL",
        bar_start=datetime.now(timezone.utc),
        signal=BUY,
        confidence=0.7,
        prob_up=0.7,
        expected_return=0.005,
        uncertainty=0.05,
        last_price=100.0,
        horizon_bars=39,
    )


@pytest.fixture
def approved():
    decision = RiskDecision(symbol="AAPL")
    decision.decision = ALLOW
    decision.target_qty = 10
    decision.target_notional = 1000.0
    return decision


@pytest.fixture
def client(tmp_settings):
    return FakeAlpacaClient(tmp_settings)


def test_client_order_ids_are_unique():
    ids = {make_client_order_id("AAPL") for _ in range(200)}
    assert len(ids) == 200


def test_client_order_id_fits_alpaca_limits():
    order_id = make_client_order_id("AAPL")
    assert len(order_id) <= 48
    assert order_id.startswith("sb-aapl-")


def test_orders_disabled_by_default_blocks_submission(tmp_settings, db, client, signal, approved):
    manager = PaperOrderManager(tmp_settings, client, db)
    result = manager.submit(signal, approved)
    assert result.submitted is False
    assert result.status == "disabled"
    assert client.submitted == []


def test_blocked_risk_decision_is_not_submitted(tmp_settings, db, client, signal, monkeypatch):
    monkeypatch.setattr(tmp_settings, "enable_paper_orders", True)
    blocked = RiskDecision(symbol="AAPL")
    blocked.no_action("confidence_too_low")
    result = PaperOrderManager(tmp_settings, client, db).submit(signal, blocked)
    assert result.submitted is False
    assert client.submitted == []


def test_approved_order_is_submitted(tmp_settings, db, client, signal, approved, monkeypatch):
    monkeypatch.setattr(tmp_settings, "enable_paper_orders", True)
    result = PaperOrderManager(tmp_settings, client, db).submit(signal, approved)
    assert result.submitted is True
    assert len(client.submitted) == 1
    assert client.submitted[0]["qty"] == 10


def test_prediction_and_risk_snapshot_stored(tmp_settings, db, client, signal, approved, monkeypatch):
    monkeypatch.setattr(tmp_settings, "enable_paper_orders", True)
    PaperOrderManager(tmp_settings, client, db).submit(signal, approved)
    order = db.recent_orders("AAPL")[0]
    assert "ALLOW" in order["risk_snapshot"]
    assert "prob_up" in order["prediction_snapshot"]


def test_pending_broker_order_blocks_submission(tmp_settings, db, client, signal, approved, monkeypatch):
    monkeypatch.setattr(tmp_settings, "enable_paper_orders", True)
    client.open_orders.append({"symbol": "AAPL", "id": "x", "status": "new"})
    result = PaperOrderManager(tmp_settings, client, db).submit(signal, approved)
    assert result.submitted is False
    assert "pending order" in result.reason


def test_rejection_is_recorded_not_swallowed(tmp_settings, db, client, signal, approved, monkeypatch):
    monkeypatch.setattr(tmp_settings, "enable_paper_orders", True)
    client.reject_next = True
    result = PaperOrderManager(tmp_settings, client, db).submit(signal, approved)
    assert result.submitted is False
    assert result.status == "failed"
    assert db.recent_orders("AAPL")[0]["status"] == "failed"


def test_sub_one_share_order_refused(tmp_settings, db, client, signal, monkeypatch):
    monkeypatch.setattr(tmp_settings, "enable_paper_orders", True)
    tiny = RiskDecision(symbol="AAPL")
    tiny.decision = ALLOW
    tiny.target_qty = 0.4
    result = PaperOrderManager(tmp_settings, client, db).submit(signal, tiny)
    assert result.submitted is False


def test_reconcile_records_fills(tmp_settings, db, client, signal, approved, monkeypatch):
    monkeypatch.setattr(tmp_settings, "enable_paper_orders", True)
    manager = PaperOrderManager(tmp_settings, client, db)
    manager.submit(signal, approved)
    client.fill_all(price=101.5)
    counts = manager.reconcile()
    assert counts["filled"] == 1
    order = db.recent_orders("AAPL")[0]
    assert order["status"] == "filled"
    assert order["filled_avg_price"] == 101.5


def test_reconcile_records_partial_fills(tmp_settings, db, client, signal, approved, monkeypatch):
    monkeypatch.setattr(tmp_settings, "enable_paper_orders", True)
    manager = PaperOrderManager(tmp_settings, client, db)
    manager.submit(signal, approved)
    client.partially_fill_all(fraction=0.4)
    counts = manager.reconcile()
    assert counts["partial"] == 1
    order = db.recent_orders("AAPL")[0]
    assert order["status"] == "partially_filled"
    assert order["filled_qty"] == pytest.approx(4.0)


def test_reconcile_records_rejections(tmp_settings, db, client, signal, approved, monkeypatch):
    monkeypatch.setattr(tmp_settings, "enable_paper_orders", True)
    manager = PaperOrderManager(tmp_settings, client, db)
    manager.submit(signal, approved)
    client.reject_all()
    counts = manager.reconcile()
    assert counts["rejected"] == 1
    assert db.recent_errors()[0]["component"] == "orders"


def test_reconcile_records_cancellations(tmp_settings, db, client, signal, approved, monkeypatch):
    monkeypatch.setattr(tmp_settings, "enable_paper_orders", True)
    manager = PaperOrderManager(tmp_settings, client, db)
    manager.submit(signal, approved)
    client.cancel_all()
    counts = manager.reconcile()
    assert counts["canceled"] == 1


def test_reconcile_skips_terminal_orders(tmp_settings, db, client, monkeypatch):
    monkeypatch.setattr(tmp_settings, "enable_paper_orders", True)
    db.record_order(client_order_id="done-1", symbol="AAPL", side="buy", qty=1, status="filled")
    counts = PaperOrderManager(tmp_settings, client, db).reconcile()
    assert counts["checked"] == 0


def test_snapshot_persists_account_and_positions(tmp_settings, db, client):
    client.positions = [
        {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0, "market_value": 1010.0,
         "cost_basis": 1000.0, "unrealized_pl": 10.0, "unrealized_plpc": 0.01,
         "current_price": 101.0}
    ]
    PaperOrderManager(tmp_settings, client, db).snapshot()
    assert db.query("SELECT * FROM positions")
    assert db.query("SELECT * FROM account_snapshots")
