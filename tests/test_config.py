"""Configuration and the paper-mode guard."""

from __future__ import annotations

import pytest

from stockbot.config import (
    ConfigError,
    LiveTradingRefused,
    env_bool,
    load_settings,
)


def test_refuses_to_start_when_paper_flag_missing(monkeypatch):
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    with pytest.raises(LiveTradingRefused):
        load_settings(dotenv_path="/nonexistent/.env")


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "LIVE", "", "  "])
def test_refuses_any_non_true_paper_value(monkeypatch, value):
    monkeypatch.setenv("ALPACA_PAPER", value)
    with pytest.raises(LiveTradingRefused):
        load_settings(dotenv_path="/nonexistent/.env")


@pytest.mark.parametrize("value", ["true", "TRUE", "True", "1", "yes", "on"])
def test_accepts_true_variants(monkeypatch, value, tmp_path):
    monkeypatch.setenv("ALPACA_PAPER", value)
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "l"))
    settings = load_settings()
    assert settings.paper is True


def test_missing_credentials_raise(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        load_settings(dotenv_path="/nonexistent/.env")


def test_orders_disabled_by_default(tmp_settings):
    assert tmp_settings.enable_paper_orders is False


def test_watchlist_parsed_and_uppercased(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCHLIST", " aapl, msft ,nvda ")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "db.sqlite"))
    settings = load_settings()
    assert settings.watchlist == ["AAPL", "MSFT", "NVDA"]


def test_benchmark_appended_to_all_symbols(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCHLIST", "AAPL")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "db.sqlite"))
    settings = load_settings()
    assert settings.all_symbols == ["AAPL", "SPY"]


def test_round_trip_cost_charges_both_sides(tmp_settings):
    # 3 bps spread + 2 bps slippage, entry and exit => 10 bps.
    assert tmp_settings.round_trip_cost_bps == pytest.approx(10.0)
    assert tmp_settings.round_trip_cost == pytest.approx(0.001)


def test_default_horizon_is_next_trading_day(tmp_settings):
    assert tmp_settings.prediction_horizon_bars == 39
    assert tmp_settings.horizon_label == "next trading day"


def test_horizon_is_configurable(monkeypatch, tmp_path):
    monkeypatch.setenv("PREDICTION_HORIZON_BARS", "6")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "db.sqlite"))
    settings = load_settings()
    assert settings.prediction_horizon_bars == 6
    assert settings.horizon_label == "60 minutes"


def test_secrets_are_not_in_repr(tmp_settings):
    text = repr(tmp_settings)
    assert "test-secret-not-real" not in text
    assert "test-key-not-real" not in text


def test_masked_config_hides_secret(tmp_settings):
    masked = tmp_settings.masked()
    assert masked["secret_key"] == "<set>"
    assert "test-key-not-real" not in masked["api_key"]


def test_invalid_confidence_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("MIN_CONFIDENCE", "0.2")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "db.sqlite"))
    with pytest.raises(ConfigError):
        load_settings()


def test_invalid_feed_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPACA_DATA_FEED", "bloomberg")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "db.sqlite"))
    with pytest.raises(ConfigError):
        load_settings()


def test_env_bool_default_used_for_blank(monkeypatch):
    monkeypatch.setenv("SOME_FLAG", "   ")
    assert env_bool("SOME_FLAG", True) is True
    assert env_bool("MISSING_FLAG", False) is False
