"""Market-wide news ingestion and the dynamic universe."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from stockbot.data.news_collector import NewsCollector
from stockbot.universe import UniverseManager
from tests.conftest import synthetic_news
from tests.fakes import FakeAlpacaClient

NOW = datetime.now(timezone.utc)


@pytest.fixture
def articles():
    return synthetic_news(
        [
            ("AAPL", -30, "Apple beats estimates as profit surges"),
            ("MSFT,NVDA", -45, "Microsoft and Nvidia announce partnership"),
            ("TSLA", -60, "Tesla recalls vehicles after investigation"),
            ("ZZZZ", -90, "Obscure ticker nobody watches issues update"),
            ("SPY,AAPL,MSFT", -120, "Broad market rally lifts major indices"),
        ],
        base=NOW,
    )


@pytest.fixture
def collector(tmp_settings, db, articles):
    client = FakeAlpacaClient(tmp_settings, news=articles)
    return NewsCollector(tmp_settings, client, db), client


# -- coverage is market-wide -------------------------------------------------
def test_ingestion_is_not_filtered_by_the_watchlist(collector, db, tmp_settings):
    """The whole point: symbols outside WATCHLIST must still be collected."""
    news, client = collector
    news.backfill()

    assert "ZZZZ" not in tmp_settings.watchlist
    assert "TSLA" not in tmp_settings.watchlist
    assert db.news_for_symbol("ZZZZ"), "off-watchlist symbol was not stored"
    assert db.news_for_symbol("TSLA")


def test_request_carries_no_symbol_filter(collector):
    """A symbol filter would silently narrow coverage."""
    news, client = collector
    news.backfill()
    assert client.news_calls
    assert all(call["symbols"] is None for call in client.news_calls)


def test_coverage_counts_every_symbol_seen(collector, db):
    news, _client = collector
    news.backfill()
    coverage = db.news_coverage()
    assert coverage["articles"] == 5
    # AAPL, MSFT, NVDA, TSLA, ZZZZ, SPY
    assert coverage["distinct_symbols"] == 6


def test_multi_symbol_article_is_linked_to_each_symbol(collector, db):
    news, _client = collector
    news.backfill()
    for symbol in ("SPY", "AAPL", "MSFT"):
        headlines = [a["headline"] for a in db.news_for_symbol(symbol)]
        assert any("Broad market rally" in h for h in headlines)


# -- idempotency and validation ---------------------------------------------
def test_reingestion_does_not_duplicate(collector, db):
    news, _client = collector
    news.backfill()
    news.backfill()
    assert db.news_coverage()["articles"] == 5


def test_watermark_advances_so_updates_resume(collector, db):
    news, client = collector
    news.backfill()
    client.news_calls.clear()
    news.update()
    # The incremental window must start near the newest stored article, not
    # at the full backfill horizon.
    start = client.news_calls[0]["start"]
    assert start > NOW - timedelta(hours=3)


def test_articles_without_an_id_are_rejected():
    bad = synthetic_news([("AAPL", -10, "No id here")], base=NOW)
    bad[0]["id"] = 0
    kept, report = NewsCollector.validate(bad)
    assert kept == []
    assert report["no_id"] == 1


def test_articles_without_a_headline_are_rejected():
    bad = synthetic_news([("AAPL", -10, "   ")], base=NOW)
    kept, report = NewsCollector.validate(bad)
    assert kept == []
    assert report["no_headline"] == 1


def test_future_dated_articles_are_rejected():
    bad = synthetic_news([("AAPL", 600, "Published in the future")], base=NOW)
    kept, report = NewsCollector.validate(bad)
    assert kept == []
    assert report["future_dated"] == 1


def test_duplicate_ids_within_one_batch_are_collapsed():
    batch = synthetic_news([("AAPL", -10, "One"), ("AAPL", -9, "Two")], base=NOW)
    batch[1]["id"] = batch[0]["id"]
    kept, report = NewsCollector.validate(batch)
    assert len(kept) == 1
    assert report["duplicate_id"] == 1


def test_polarity_stored_with_each_article(collector, db):
    news, _client = collector
    news.backfill()
    apple = db.news_for_symbol("AAPL")
    beats = [a for a in apple if "beats estimates" in a["headline"]]
    assert beats and beats[0]["polarity"] > 0
    tesla = db.news_for_symbol("TSLA")[0]
    assert tesla["polarity"] < 0


def test_truncation_is_recorded_not_hidden(tmp_settings, db, articles, monkeypatch):
    """Hitting the cap must be visible, not silently treated as full coverage."""
    monkeypatch.setattr(tmp_settings, "news_page_limit", 1)
    monkeypatch.setattr(tmp_settings, "news_max_pages", 2)
    client = FakeAlpacaClient(tmp_settings, news=articles)
    result = NewsCollector(tmp_settings, client, db).backfill()

    assert result["truncated"] is True
    assert db.recent_news_runs(1)[0]["truncated"] == 1
    warnings = [e for e in db.recent_errors() if e["component"] == "news"]
    assert warnings and "truncated" in warnings[0]["message"]


def test_disabled_news_is_a_no_op(tmp_settings, db, articles, monkeypatch):
    monkeypatch.setattr(tmp_settings, "news_enabled", False)
    client = FakeAlpacaClient(tmp_settings, news=articles)
    result = NewsCollector(tmp_settings, client, db).backfill()
    assert result["status"] == "DISABLED"
    assert db.news_coverage()["articles"] == 0


def test_fetch_failure_is_recorded(tmp_settings, db, monkeypatch):
    from stockbot.alpaca_client import AlpacaError

    client = FakeAlpacaClient(tmp_settings, news=[])

    def explode(*_a, **_k):
        raise AlpacaError("simulated news outage")

    monkeypatch.setattr(client, "get_news", explode)
    result = NewsCollector(tmp_settings, client, db).backfill()
    assert result["status"] == "ERROR"
    assert db.recent_news_runs(1)[0]["status"] == "ERROR"


# -- dynamic universe --------------------------------------------------------
@pytest.fixture
def universe(tmp_settings, db, articles):
    client = FakeAlpacaClient(tmp_settings, news=articles)
    NewsCollector(tmp_settings, client, db).backfill()
    manager = UniverseManager(tmp_settings, client, db)
    manager.refresh_assets(force=True)
    return manager


def test_core_watchlist_is_always_analysed(universe, tmp_settings):
    built = universe.build()
    for symbol in tmp_settings.watchlist:
        assert symbol in built.symbols


def test_benchmark_is_not_analysed_just_for_being_the_benchmark(universe, tmp_settings):
    """SPY's bars are needed for relative features; that is not a signal request."""
    assert tmp_settings.benchmark_symbol not in tmp_settings.watchlist
    built = universe.build()
    assert tmp_settings.benchmark_symbol not in built.symbols


def test_news_active_symbols_are_admitted_beyond_the_watchlist(universe, tmp_settings, db):
    """A symbol nobody configured should still get analysed if news is on it."""
    # Give TSLA enough articles to clear the candidate floor.
    extra = synthetic_news(
        [("TSLA", -20, "Tesla announces record deliveries"),
         ("TSLA", -25, "Tesla upgraded by analysts")],
        base=NOW,
    )
    for i, a in enumerate(extra):
        a["id"] = 5000 + i
    db.upsert_news(extra)

    built = universe.build()
    assert "TSLA" not in tmp_settings.watchlist
    assert "TSLA" in built.news_driven
    assert "TSLA" in built.symbols


def test_untradable_news_symbols_are_rejected_with_a_reason(tmp_settings, db, articles):
    """A news mention is not evidence a ticker is tradable."""
    client = FakeAlpacaClient(
        tmp_settings,
        news=articles,
        assets=[{
            "symbol": s, "name": s, "exchange": "NASDAQ", "asset_class": "us_equity",
            "status": "active", "tradable": True, "shortable": True, "fractionable": True,
        } for s in tmp_settings.all_symbols],   # ZZZZ deliberately absent
    )
    NewsCollector(tmp_settings, client, db).backfill()
    extra = synthetic_news([("ZZZZ", -5, "More on the obscure ticker")], base=NOW)
    extra[0]["id"] = 7001
    db.upsert_news(extra)

    manager = UniverseManager(tmp_settings, client, db)
    manager.refresh_assets(force=True)
    built = manager.build()

    assert "ZZZZ" not in built.symbols
    rejected = {e.symbol: e.reason for e in built.rejected}
    assert rejected.get("ZZZZ") == "not_tradable_on_alpaca"


def test_capacity_cap_is_respected_and_recorded(universe, tmp_settings, db, monkeypatch):
    monkeypatch.setattr(tmp_settings, "max_dynamic_symbols", 1)
    monkeypatch.setattr(tmp_settings, "min_news_for_candidate", 1)
    built = universe.build()
    assert len(built.news_driven) <= 1
    reasons = {e.reason for e in built.rejected}
    assert "capacity_cap_reached" in reasons


def test_candidate_floor_excludes_one_off_mentions(universe, tmp_settings, monkeypatch):
    monkeypatch.setattr(tmp_settings, "min_news_for_candidate", 99)
    assert universe.build().news_driven == []


def test_dynamic_universe_can_be_disabled(universe, tmp_settings, monkeypatch):
    monkeypatch.setattr(tmp_settings, "dynamic_universe_enabled", False)
    built = universe.build()
    assert built.news_driven == []
    assert built.symbols == tmp_settings.watchlist


def test_nothing_admitted_when_the_asset_list_is_unavailable(
    tmp_settings, db, articles, monkeypatch
):
    # Floor of 1 so off-watchlist symbols actually reach the tradability check.
    monkeypatch.setattr(tmp_settings, "min_news_for_candidate", 1)
    client = FakeAlpacaClient(tmp_settings, news=articles, assets=[])
    NewsCollector(tmp_settings, client, db).backfill()
    manager = UniverseManager(tmp_settings, client, db)
    manager.refresh_assets(force=True)
    built = manager.build()
    assert built.news_driven == []
    assert any(e.reason == "asset_list_unavailable" for e in built.rejected)


def test_universe_snapshot_is_persisted(universe, db):
    universe.build()
    snapshot = db.latest_universe()
    assert snapshot
    assert {row["source"] for row in snapshot} <= {"core", "news"}


def test_coverage_report_distinguishes_news_from_analysis(universe):
    report = universe.coverage_report()
    # News coverage is market-wide, so it should exceed what is analysed.
    assert report["news_symbols_in_window"] >= report["news_driven_symbols"]
    assert report["core_symbols"] >= 1
    assert report["assets"]["tradable"] >= 1
