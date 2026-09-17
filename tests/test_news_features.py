"""News features, with leakage as the central concern.

A news feature attached to a bar may only see articles published at or before
that bar's close. These tests assert that directly.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from stockbot.features import FEATURE_COLUMNS, build_features, prepare_training_frame
from stockbot.news_features import (
    NEWS_FEATURE_COLUMNS,
    NO_NEWS_RECENCY_MINUTES,
    NewsIndex,
    build_news_features,
    headline_polarity,
)

BASE = datetime(2025, 6, 20, 14, 0, tzinfo=timezone.utc)


def _rows(spec):
    """``spec`` is a list of (symbol, minutes_offset, polarity, article_id)."""
    return [
        {
            "symbol": symbol,
            "created_at": (BASE + timedelta(minutes=offset)).isoformat(),
            "polarity": polarity,
            "article_id": article_id,
        }
        for symbol, offset, polarity, article_id in spec
    ]


def _bar_starts(count: int, step_minutes: int = 10, start_offset: int = 0) -> pd.Series:
    return pd.Series(
        [BASE + timedelta(minutes=start_offset + i * step_minutes) for i in range(count)]
    )


# -- polarity lexicon -------------------------------------------------------
def test_polarity_is_signed_and_bounded():
    assert headline_polarity("Company beats estimates, profit surges") > 0
    assert headline_polarity("Company misses estimates, shares plunge") < 0
    assert -1.0 <= headline_polarity("beats miss") <= 1.0


def test_polarity_is_zero_without_lexicon_words():
    assert headline_polarity("Company announces annual general meeting date") == 0.0


def test_polarity_handles_empty_input():
    assert headline_polarity(None) == 0.0
    assert headline_polarity("") == 0.0
    assert headline_polarity("!!! ??? 123") == 0.0


# -- as-of correctness (leakage) --------------------------------------------
def test_features_never_see_articles_published_after_the_bar_closes():
    """The defining leakage test for news.

    An article published well after a bar must not appear in that bar's
    features, no matter that it exists in the index.
    """
    index = NewsIndex.from_rows(_rows([("AAPL", 500, 1.0, 1)]))
    bars = _bar_starts(5)   # bars at +0..+40, all closing long before +500
    out = build_news_features(bars, "AAPL", index, bar_minutes=10)
    assert (out["news_count_24h"] == 0).all()
    assert (out["has_news_24h"] == 0).all()


def test_article_inside_the_window_is_counted():
    # Bar starts at +0, closes at +10. An article at +5 is available by then.
    index = NewsIndex.from_rows(_rows([("AAPL", 5, 0.0, 1)]))
    out = build_news_features(_bar_starts(1), "AAPL", index, bar_minutes=10)
    assert out["news_count_1h"].iloc[0] == 1.0
    assert out["has_news_24h"].iloc[0] == 1.0


def test_article_after_bar_close_excluded_from_that_bar_but_seen_by_the_next():
    """An article at +15 is invisible to the bar closing at +10, visible at +20."""
    index = NewsIndex.from_rows(_rows([("AAPL", 15, 0.0, 1)]))
    out = build_news_features(_bar_starts(2), "AAPL", index, bar_minutes=10)
    assert out["news_count_1h"].iloc[0] == 0.0    # bar closes at +10
    assert out["news_count_1h"].iloc[1] == 1.0    # bar closes at +20


def test_truncating_future_news_does_not_change_earlier_features():
    """Recomputing with future articles removed must give identical values."""
    full = _rows([("AAPL", o, 0.5, i) for i, o in enumerate([1, 20, 40, 900, 2000])])
    past_only = [r for r in full if r["article_id"] < 3]

    bars = _bar_starts(4)
    with_future = build_news_features(bars, "AAPL", NewsIndex.from_rows(full), 10)
    without = build_news_features(bars, "AAPL", NewsIndex.from_rows(past_only), 10)

    for column in NEWS_FEATURE_COLUMNS:
        if column == "market_news_count_1h":
            continue    # market intensity legitimately differs between stores
        np.testing.assert_allclose(
            with_future[column].to_numpy(), without[column].to_numpy(),
            err_msg=f"{column} changed when future articles were removed",
        )


def test_updated_at_is_never_read_by_the_feature_code():
    """A later revision must not backdate information into an earlier bar.

    Checks the module's executable code rather than its prose, so the docstring
    explaining this rule does not itself trip the assertion.
    """
    import ast
    import inspect

    import stockbot.news_features as module

    tree = ast.parse(inspect.getsource(module))

    # Drop docstrings so only real code is inspected.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:]

    offenders = [
        n for n in ast.walk(tree)
        if (isinstance(n, ast.Constant) and n.value == "updated_at")
        or (isinstance(n, ast.Attribute) and n.attr == "updated_at")
    ]
    assert offenders == []


# -- window arithmetic ------------------------------------------------------
def test_windows_nest_correctly():
    index = NewsIndex.from_rows(
        _rows([("AAPL", -30, 0.0, 1), ("AAPL", -600, 0.0, 2), ("AAPL", -5000, 0.0, 3)])
    )
    out = build_news_features(_bar_starts(1), "AAPL", index, bar_minutes=10)
    row = out.iloc[0]
    assert row["news_count_1h"] == 1.0
    assert row["news_count_24h"] == 2.0
    assert row["news_count_7d"] == 3.0
    assert row["news_count_1h"] <= row["news_count_24h"] <= row["news_count_7d"]


def test_recency_measures_from_the_bar_close():
    index = NewsIndex.from_rows(_rows([("AAPL", -20, 0.0, 1)]))
    out = build_news_features(_bar_starts(1), "AAPL", index, bar_minutes=10)
    # Bar closes at +10; the article is at -20, so 30 minutes earlier.
    assert out["news_recency_minutes"].iloc[0] == pytest.approx(30.0)


def test_recency_capped_when_no_prior_article():
    index = NewsIndex.from_rows(_rows([("MSFT", -20, 0.0, 1)]))
    out = build_news_features(_bar_starts(1), "AAPL", index, bar_minutes=10)
    assert out["news_recency_minutes"].iloc[0] == NO_NEWS_RECENCY_MINUTES


def test_polarity_averages_articles_in_the_window():
    index = NewsIndex.from_rows(
        _rows([("AAPL", -5, 1.0, 1), ("AAPL", -6, -0.5, 2)])
    )
    out = build_news_features(_bar_starts(1), "AAPL", index, bar_minutes=10)
    assert out["news_polarity_1h"].iloc[0] == pytest.approx(0.25)


def test_burst_compares_against_the_weekly_baseline():
    # One article this hour against a quiet week gives a large burst value.
    spec = [("AAPL", -5, 0.0, 1)] + [("AAPL", -6000 - i * 10, 0.0, 100 + i) for i in range(3)]
    index = NewsIndex.from_rows(_rows(spec))
    out = build_news_features(_bar_starts(1), "AAPL", index, bar_minutes=10)
    assert out["news_burst"].iloc[0] > 1.0


def test_market_intensity_counts_each_article_once():
    """A story tagging many tickers is one market event, not many."""
    index = NewsIndex.from_rows(
        _rows([("AAPL", -5, 0.0, 1), ("MSFT", -5, 0.0, 1), ("NVDA", -5, 0.0, 1)])
    )
    out = build_news_features(_bar_starts(1), "AAPL", index, bar_minutes=10)
    assert out["market_news_count_1h"].iloc[0] == 1.0


# -- absence handling -------------------------------------------------------
def test_no_index_gives_nan_so_columns_are_dropped():
    out = build_news_features(_bar_starts(3), "AAPL", None, bar_minutes=10)
    assert out[NEWS_FEATURE_COLUMNS].isna().all().all()


def test_empty_store_gives_nan():
    out = build_news_features(_bar_starts(3), "AAPL", NewsIndex.from_rows([]), 10)
    assert out[NEWS_FEATURE_COLUMNS].isna().all().all()


def test_symbol_with_no_news_gets_zeros_not_nan():
    """Zero is informative and must not drop the row from training."""
    index = NewsIndex.from_rows(_rows([("MSFT", -5, 0.0, 1)]))
    out = build_news_features(_bar_starts(3), "AAPL", index, bar_minutes=10)
    assert (out["news_count_24h"] == 0).all()
    assert out[["news_count_1h", "news_polarity_24h", "has_news_24h"]].notna().all().all()


# -- integration with the full feature matrix -------------------------------
def test_news_columns_present_in_the_feature_list():
    for column in NEWS_FEATURE_COLUMNS:
        assert column in FEATURE_COLUMNS


def test_news_features_dropped_when_store_is_empty(bars, benchmark_bars):
    featured = build_features(bars, benchmark_bars, news_index=None, symbol="TEST")
    _, usable = prepare_training_frame(featured, FEATURE_COLUMNS)
    assert not any(c in usable for c in NEWS_FEATURE_COLUMNS)


def test_news_features_survive_when_store_has_data(bars, benchmark_bars):
    """With news present the columns must be usable, not silently all-NaN."""
    stamps = pd.to_datetime(bars["bar_start"], utc=True)
    rows = [
        {
            "symbol": "TEST",
            "created_at": stamps.iloc[i].isoformat(),
            "polarity": 0.4,
            "article_id": i,
        }
        for i in range(0, len(stamps), 25)
    ]
    featured = build_features(
        bars, benchmark_bars, news_index=NewsIndex.from_rows(rows), symbol="TEST"
    )
    _, usable = prepare_training_frame(featured, FEATURE_COLUMNS)
    assert "news_count_24h" in usable
    assert "has_news_24h" in usable
    # Adding news must not shrink the training set.
    assert featured[["news_count_24h", "news_polarity_24h"]].notna().all().all()


def test_adding_news_does_not_change_price_features(bars, benchmark_bars):
    """News columns must not disturb the price/volume features."""
    stamps = pd.to_datetime(bars["bar_start"], utc=True)
    rows = [{"symbol": "TEST", "created_at": stamps.iloc[5].isoformat(),
             "polarity": 0.0, "article_id": 1}]
    without = build_features(bars, benchmark_bars, symbol="TEST")
    with_news = build_features(
        bars, benchmark_bars, news_index=NewsIndex.from_rows(rows), symbol="TEST"
    )
    price_columns = [c for c in FEATURE_COLUMNS if c not in NEWS_FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(
        without[price_columns], with_news[price_columns]
    )
