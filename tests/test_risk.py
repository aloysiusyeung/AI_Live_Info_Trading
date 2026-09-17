"""The deterministic risk layer. Every rule gets a test."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from stockbot.risk import ALLOW, BLOCK, NO_ACTION, RiskContext, RiskEngine
from stockbot.signals import AVOID, BUY, HOLD, INSUFFICIENT_EVIDENCE, Signal


@pytest.fixture
def engine(tmp_settings, db, monkeypatch):
    monkeypatch.setattr(tmp_settings, "enable_paper_orders", True)
    return RiskEngine(tmp_settings, db)


@pytest.fixture
def good_signal(tmp_settings):
    return Signal(
        symbol="AAPL",
        bar_start=datetime.now(timezone.utc),
        signal=BUY,
        confidence=0.72,
        prob_up=0.72,
        expected_return=0.004,
        uncertainty=0.05,
        last_price=100.0,
        model_name="random_forest",
        horizon_bars=39,
    )


@pytest.fixture
def good_context():
    return RiskContext(
        account={
            "equity": 100_000.0,
            "last_equity": 100_000.0,
            "buying_power": 200_000.0,
            "account_blocked": False,
            "trading_blocked": False,
        },
        positions={},
        open_orders=[],
        quote={"bid": 99.98, "ask": 100.02, "mid": 100.0, "spread_bps": 4.0},
        market_open=True,
        data_age_seconds=120.0,
        avg_dollar_volume=500_000_000.0,
        realised_volatility=0.002,
        atr_pct=0.004,
        now=datetime.now(timezone.utc),
    )


def test_happy_path_allows_an_order(engine, good_signal, good_context):
    decision = engine.evaluate(good_signal, good_context)
    assert decision.decision == ALLOW
    assert decision.target_qty >= 1
    assert all(decision.checks.values())


def test_kill_switch_blocks_everything(engine, good_signal, good_context):
    engine.engage_kill_switch(actor="test")
    decision = engine.evaluate(good_signal, good_context)
    assert decision.decision == BLOCK
    assert "kill_switch_engaged" in decision.reasons


def test_kill_switch_can_be_released(engine, good_signal, good_context):
    engine.engage_kill_switch(actor="test")
    engine.release_kill_switch(actor="test")
    assert engine.evaluate(good_signal, good_context).decision == ALLOW


def test_env_kill_switch_also_blocks(tmp_settings, db, good_signal, good_context, monkeypatch):
    monkeypatch.setattr(tmp_settings, "enable_paper_orders", True)
    monkeypatch.setattr(tmp_settings, "kill_switch", True)
    decision = RiskEngine(tmp_settings, db).evaluate(good_signal, good_context)
    assert decision.decision == BLOCK


def test_orders_disabled_means_no_order(tmp_settings, db, good_signal, good_context):
    assert tmp_settings.enable_paper_orders is False
    decision = RiskEngine(tmp_settings, db).evaluate(good_signal, good_context)
    assert decision.decision == NO_ACTION
    assert "paper_orders_disabled" in decision.reasons


def test_blocked_account_is_fatal(engine, good_signal, good_context):
    good_context.account["trading_blocked"] = True
    decision = engine.evaluate(good_signal, good_context)
    assert decision.decision == BLOCK


def test_closed_market_blocks(engine, good_signal, good_context):
    good_context.market_open = False
    assert engine.evaluate(good_signal, good_context).decision == NO_ACTION


@pytest.mark.parametrize("label", [HOLD, AVOID, INSUFFICIENT_EVIDENCE])
def test_only_buy_signals_reach_an_order(engine, good_signal, good_context, label):
    good_signal.signal = label
    decision = engine.evaluate(good_signal, good_context)
    assert decision.decision == NO_ACTION
    assert decision.target_qty == 0


def test_low_confidence_blocks(engine, good_signal, good_context):
    good_signal.prob_up = 0.51
    decision = engine.evaluate(good_signal, good_context)
    assert decision.decision == NO_ACTION
    assert decision.checks["confidence"] is False


def test_high_uncertainty_blocks(engine, good_signal, good_context):
    good_signal.uncertainty = 0.9
    assert engine.evaluate(good_signal, good_context).checks["uncertainty"] is False


def test_non_positive_expected_return_blocks(engine, good_signal, good_context):
    good_signal.expected_return = -0.001
    assert engine.evaluate(good_signal, good_context).checks["expected_return"] is False


def test_stale_data_blocks(engine, good_signal, good_context):
    good_context.data_age_seconds = 99_999.0
    decision = engine.evaluate(good_signal, good_context)
    assert decision.decision == BLOCK
    assert any("stale_data" in r for r in decision.reasons)


def test_missing_data_age_blocks(engine, good_signal, good_context):
    good_context.data_age_seconds = None
    assert engine.evaluate(good_signal, good_context).decision == BLOCK


def test_illiquid_symbol_blocks(engine, good_signal, good_context):
    good_context.avg_dollar_volume = 1_000.0
    assert engine.evaluate(good_signal, good_context).checks["liquidity"] is False


def test_wide_spread_blocks(engine, good_signal, good_context):
    good_context.quote["spread_bps"] = 500.0
    assert engine.evaluate(good_signal, good_context).checks["spread"] is False


def test_missing_quote_blocks(engine, good_signal, good_context):
    good_context.quote = None
    assert engine.evaluate(good_signal, good_context).checks["spread"] is False


def test_daily_loss_limit_blocks(engine, good_signal, good_context):
    good_context.account["equity"] = 97_000.0     # -3% against a 2% limit
    good_context.account["last_equity"] = 100_000.0
    decision = engine.evaluate(good_signal, good_context)
    assert decision.decision == BLOCK
    assert any("daily_loss" in r for r in decision.reasons)


def test_small_daily_loss_is_tolerated(engine, good_signal, good_context):
    good_context.account["equity"] = 99_500.0     # -0.5%
    assert engine.evaluate(good_signal, good_context).decision == ALLOW


def test_pending_order_blocks(engine, good_signal, good_context):
    good_context.open_orders = [{"symbol": "AAPL", "status": "new"}]
    decision = engine.evaluate(good_signal, good_context)
    assert decision.checks["no_pending_order"] is False


def test_pending_order_for_another_symbol_does_not_block(engine, good_signal, good_context):
    good_context.open_orders = [{"symbol": "TSLA", "status": "new"}]
    assert engine.evaluate(good_signal, good_context).decision == ALLOW


def test_duplicate_order_within_window_blocks(engine, db, good_signal, good_context):
    db.record_order(
        client_order_id="dupe-1", symbol="AAPL", side="buy", qty=10, status="filled"
    )
    decision = engine.evaluate(good_signal, good_context)
    assert decision.checks["no_duplicate_order"] is False


def test_rejected_order_does_not_count_as_duplicate(engine, db, good_signal, good_context):
    db.record_order(
        client_order_id="rej-1", symbol="AAPL", side="buy", qty=10, status="rejected"
    )
    assert engine.evaluate(good_signal, good_context).decision == ALLOW


def test_position_at_cap_blocks(engine, good_signal, good_context):
    good_context.positions = {"AAPL": {"symbol": "AAPL", "qty": 200, "market_value": 20_000.0}}
    decision = engine.evaluate(good_signal, good_context)
    assert decision.checks["position_cap"] is False


def test_portfolio_exposure_cap_blocks(engine, good_signal, good_context):
    good_context.positions = {
        f"SYM{i}": {"symbol": f"SYM{i}", "qty": 10, "market_value": 10_000.0}
        for i in range(7)
    }
    decision = engine.evaluate(good_signal, good_context)
    assert decision.checks["portfolio_exposure"] is False


def test_sizing_respects_per_symbol_cap(engine, good_signal, good_context):
    decision = engine.evaluate(good_signal, good_context)
    cap = good_context.account["equity"] * engine.settings.max_position_pct
    assert decision.target_notional <= cap + 1e-6


def test_higher_volatility_gives_a_smaller_position(engine, good_signal, good_context):
    calm = engine.evaluate(good_signal, good_context).target_qty
    good_context.atr_pct = 0.08
    volatile = engine.evaluate(good_signal, good_context).target_qty
    assert volatile < calm
    assert volatile >= 1


def test_sizing_falls_back_to_the_cap_without_volatility(engine, good_context):
    qty, notional, basis = engine.volatility_position_size(100_000.0, 100.0, None, None)
    assert basis == "cap_only"
    assert notional == pytest.approx(100_000.0 * engine.settings.max_position_pct)


def test_insufficient_buying_power_blocks(engine, good_signal, good_context):
    good_context.account["buying_power"] = 10.0
    assert engine.evaluate(good_signal, good_context).checks["buying_power"] is False


def test_order_below_minimum_notional_blocks(engine, good_signal, good_context):
    # last_equity must move with equity, or the daily-loss rule fires first.
    good_context.account["equity"] = 200.0
    good_context.account["last_equity"] = 200.0
    good_context.account["buying_power"] = 200.0
    good_signal.last_price = 5.0
    decision = engine.evaluate(good_signal, good_context)
    assert decision.decision == NO_ACTION


def test_default_is_no_trade_on_an_empty_context(engine, good_signal):
    decision = engine.evaluate(good_signal, RiskContext())
    assert decision.decision != ALLOW
    assert decision.target_qty == 0


def test_decision_is_serialisable(engine, good_signal, good_context):
    payload = engine.evaluate(good_signal, good_context).as_dict()
    assert payload["decision"] == ALLOW
    assert isinstance(payload["checks"], dict)
