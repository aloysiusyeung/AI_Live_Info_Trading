"""Paper-only guard rails on the Alpaca client.

No network call is made: the SDK clients are replaced with stubs.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from stockbot.alpaca_client import LIVE_HOST, PAPER_HOST, AlpacaClient
from stockbot.config import LiveTradingRefused


class _StubTrading:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self._base_url = f"https://{PAPER_HOST}"


class _StubData:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def stub_sdk(monkeypatch):
    import stockbot.alpaca_client as module

    monkeypatch.setattr(module, "TradingClient", _StubTrading)
    monkeypatch.setattr(module, "StockHistoricalDataClient", _StubData)
    return module


def test_trading_client_always_constructed_with_paper_true(tmp_settings, stub_sdk):
    client = AlpacaClient(tmp_settings)
    assert client.trading.kwargs["paper"] is True


def test_no_url_override_is_passed(tmp_settings, stub_sdk):
    """url_override is the only way to reach a non-paper host; it must be unused."""
    client = AlpacaClient(tmp_settings)
    assert "url_override" not in client.trading.kwargs


def test_live_endpoint_is_refused(tmp_settings, stub_sdk, monkeypatch):
    class LiveTrading(_StubTrading):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._base_url = f"https://{LIVE_HOST}"

    monkeypatch.setattr(stub_sdk, "TradingClient", LiveTrading)
    with pytest.raises(LiveTradingRefused):
        AlpacaClient(tmp_settings)


def test_client_refuses_construction_outside_paper_mode(tmp_settings, stub_sdk):
    not_paper = replace(tmp_settings, paper=False)
    with pytest.raises(LiveTradingRefused):
        AlpacaClient(not_paper)


def test_timeframe_matches_configured_bar_size(tmp_settings, stub_sdk):
    client = AlpacaClient(tmp_settings)
    assert str(client.timeframe) == "10Min"


def test_submit_refuses_when_paper_flag_flipped(tmp_settings, stub_sdk):
    client = AlpacaClient(tmp_settings)
    client.settings = replace(tmp_settings, paper=False)
    with pytest.raises(LiveTradingRefused):
        client.submit_market_order("AAPL", 1, "buy", "cid-1")


def test_placeholder_credentials_are_detected(tmp_settings, stub_sdk):
    """A template value must be named as such, not reported as a bare 401."""
    client = AlpacaClient(tmp_settings)
    client.settings = replace(
        tmp_settings,
        alpaca_api_key="YOUR_ALPACA_API_KEY_HERE",
        alpaca_secret_key="YOUR_ALPACA_SECRET_KEY_HERE",
    )
    assert client.credentials_look_like_placeholders() is True

    report = client.connection_test()
    assert report["ok"] is False
    assert report["credentials"] == "placeholder"
    assert "placeholder" in report["error"]
    # The report must never echo the configured values.
    assert "YOUR_ALPACA_SECRET_KEY_HERE" not in str(report)


def test_realistic_credentials_are_not_flagged(tmp_settings, stub_sdk):
    client = AlpacaClient(tmp_settings)
    client.settings = replace(
        tmp_settings,
        alpaca_api_key="PKABCDEF1234567890XY",
        alpaca_secret_key="abcdefGHIJ1234567890abcdefGHIJ1234567890",
    )
    assert client.credentials_look_like_placeholders() is False


def test_paper_endpoint_enum_is_read_correctly(tmp_settings, stub_sdk, monkeypatch):
    """alpaca-py stores the base URL as an enum; str() on it is the enum name."""
    from enum import Enum

    class BaseURL(Enum):
        TRADING_PAPER = f"https://{PAPER_HOST}"

    class EnumTrading(_StubTrading):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._base_url = BaseURL.TRADING_PAPER

    monkeypatch.setattr(stub_sdk, "TradingClient", EnumTrading)
    client = AlpacaClient(tmp_settings)   # must not warn or raise
    assert client.trading.kwargs["paper"] is True
