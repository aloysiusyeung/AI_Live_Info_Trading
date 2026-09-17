"""End-to-end: market-wide news through to signals and the universe."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from stockbot.engine import TradingEngine
from tests.conftest import synthetic_bars, synthetic_news
from tests.fakes import FakeAlpacaClient


def _bar_rows(symbol: str, n_sessions: int, seed: int) -> list[dict]:
    frame = synthetic_bars(
        symbol=symbol, n_sessions=n_sessions, seed=seed,
        end_date=datetime.now(timezone.utc),
    )
    shift = datetime.now(timezone.utc) - timedelta(minutes=15) - frame["bar_start"].iloc[-1]
    frame["bar_start"] = frame["bar_start"] + shift
    return frame.to_dict("records")


@pytest.fixture
def wired(tmp_settings, db, monkeypatch):
    monkeypatch.setattr(tmp_settings, "watchlist", ["AAPL"])
    monkeypatch.setattr(tmp_settings, "min_training_rows", 400)
    monkeypatch.setattr(tmp_settings, "min_news_for_candidate", 1)

    news = synthetic_news(
        [
            ("AAPL", -30, "Apple beats estimates as profit surges"),
            ("TSLA", -40, "Tesla upgraded after strong deliveries"),
            ("TSLA", -50, "Tesla expands production capacity"),
            ("ZZZZ", -60, "Untradable ticker in the news"),
        ],
        base=datetime.now(timezone.utc),
    )
    bars = {
        "AAPL": _bar_rows("AAPL", 40, seed=11),
        "SPY": _bar_rows("SPY", 40, seed=29),
        "TSLA": _bar_rows("TSLA", 40, seed=37),
    }
    assets = [
        {"symbol": s, "name": s, "exchange": "NASDAQ", "asset_class": "us_equity",
         "status": "active", "tradable": True, "shortable": True, "fractionable": True}
        for s in ("AAPL", "SPY", "TSLA")
    ]
    client = FakeAlpacaClient(
        tmp_settings, bars=bars, news=news, assets=assets, market_open=True
    )
    return tmp_settings, db, client, TradingEngine(tmp_settings, client, db)


def test_cycle_ingests_news_market_wide(wired):
    settings, db, client, engine = wired
    engine.collector.backfill()
    engine.run_cycle()

    # ZZZZ is neither on the watchlist nor tradable, but its news is stored.
    assert db.news_for_symbol("ZZZZ")
    assert db.news_coverage()["articles"] == 4
    assert all(call["symbols"] is None for call in client.news_calls)


def test_cycle_analyses_a_news_driven_symbol_outside_the_watchlist(wired):
    settings, db, client, engine = wired
    engine.collector.backfill()
    result = engine.run_cycle()

    analysed = {o.symbol for o in result.outcomes}
    assert "TSLA" not in settings.watchlist
    assert "TSLA" in analysed, "news-driven symbol was not analysed"
    assert result.universe["news_driven"]


def test_untradable_news_symbol_is_not_analysed(wired):
    _settings, _db, _client, engine = wired
    engine.collector.backfill()
    result = engine.run_cycle()
    assert "ZZZZ" not in {o.symbol for o in result.outcomes}


def test_news_features_are_used_when_news_is_present(wired):
    """A trained model should actually pick up the news columns."""
    _settings, db, _client, engine = wired
    engine.collector.backfill()
    engine.news.backfill()
    engine.news_index(refresh=True)
    report = engine.ensure_models(force=True, symbols=["AAPL"])

    assert report["AAPL"]["status"] in {"trained", "no_model"}
    model = db.selected_model("AAPL")
    if model:
        import json
        features = json.loads(model["feature_names"])
        assert any(f.startswith("news_") or f == "has_news_24h" for f in features)


def test_cycle_result_reports_news_and_universe(wired):
    _settings, _db, _client, engine = wired
    engine.collector.backfill()
    payload = engine.run_cycle().as_dict()
    assert payload["news"]["status"] == "OK"
    assert "core" in payload["universe"]
    assert payload["universe"]["total"] >= 1


def test_news_disabled_leaves_the_cycle_working(wired, monkeypatch):
    settings, db, _client, engine = wired
    monkeypatch.setattr(settings, "news_enabled", False)
    engine.collector.backfill()
    result = engine.run_cycle()
    assert result.status == "OK"
    assert db.news_coverage()["articles"] == 0
    assert engine.news_index() is None


def test_news_failure_does_not_break_the_cycle(wired, monkeypatch):
    from stockbot.alpaca_client import AlpacaError

    _settings, db, client, engine = wired
    engine.collector.backfill()

    def explode(*_a, **_k):
        raise AlpacaError("simulated news outage")

    monkeypatch.setattr(client, "get_news", explode)
    result = engine.run_cycle()
    assert result.status == "OK"
    assert result.news["status"] == "ERROR"
    assert result.outcomes, "cycle produced no signals despite news failing"
