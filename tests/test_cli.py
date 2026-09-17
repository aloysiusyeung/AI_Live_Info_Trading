"""CLI wiring. No network calls: the Alpaca client is stubbed."""

from __future__ import annotations

import json

import pytest

from stockbot import cli
from tests.fakes import FakeAlpacaClient


@pytest.fixture
def stub_client(monkeypatch, tmp_settings):
    """Replace AlpacaClient everywhere the CLI constructs one."""
    created: list[FakeAlpacaClient] = []

    def factory(settings):
        client = FakeAlpacaClient(settings, bars={}, market_open=False)
        created.append(client)
        return client

    monkeypatch.setattr(cli, "AlpacaClient", factory)
    return created


def test_parser_exposes_every_command():
    parser = cli.build_parser()
    for command in ("check", "backfill", "train", "cycle", "run", "status", "kill-switch"):
        assert parser.parse_args([command] if command != "kill-switch"
                                else [command, "engage"])


def test_command_is_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_check_reports_placeholder_credentials(tmp_path, monkeypatch, capsys):
    """The environment's template keys must be reported, not hidden."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "l"))
    monkeypatch.setenv("ALPACA_API_KEY", "YOUR_ALPACA_API_KEY_HERE")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY_HERE")

    exit_code = cli.main(["check"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["credentials"] == "placeholder"
    assert payload["note"].startswith("Connection test is read-only")


def test_check_never_prints_the_secret(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "l"))
    cli.main(["check"])
    out = capsys.readouterr().out
    assert "test-secret-not-real" not in out
    assert "test-key-not-real" not in out


def test_refuses_to_run_outside_paper_mode(monkeypatch, capsys):
    monkeypatch.setenv("ALPACA_PAPER", "false")
    assert cli.main(["status"]) == 2
    assert "ALPACA_PAPER must be 'true'" in capsys.readouterr().err


def test_status_reports_state(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "l"))
    assert cli.main(["status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kill_switch"] is False
    assert payload["scheduler"]["alive"] is False


def test_kill_switch_engage_and_release(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "l"))

    cli.main(["kill-switch", "engage"])
    assert json.loads(capsys.readouterr().out)["kill_switch"] is True

    cli.main(["kill-switch", "release"])
    assert json.loads(capsys.readouterr().out)["kill_switch"] is False


def test_cycle_command_runs_and_skips_closed_market(tmp_path, monkeypatch, stub_client, capsys):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "m"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "l"))
    assert cli.main(["cycle"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "SKIPPED"
    assert payload["reason"] == "market_closed"
    assert payload["orders_submitted"] == 0
