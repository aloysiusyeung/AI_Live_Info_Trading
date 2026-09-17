"""Structured logging and credential redaction."""

from __future__ import annotations

import json
import logging

from stockbot.logging_setup import JsonFormatter, RedactingFilter, setup_logging


def test_records_are_valid_json():
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", None, None)
    record.symbol = "AAPL"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "hello"
    assert payload["symbol"] == "AAPL"
    assert payload["level"] == "INFO"


def test_secret_value_is_redacted(monkeypatch):
    monkeypatch.setenv("ALPACA_SECRET_KEY", "super-secret-value-123")
    filt = RedactingFilter()
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1, "key is super-secret-value-123", None, None
    )
    filt.filter(record)
    assert "super-secret-value-123" not in record.getMessage()
    assert "REDACTED" in record.getMessage()


def test_key_value_patterns_are_redacted():
    filt = RedactingFilter()
    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1, "api_key=ABCDEF123456", None, None
    )
    filt.filter(record)
    assert "ABCDEF123456" not in record.getMessage()


def test_setup_logging_writes_a_file(tmp_path):
    logger = setup_logging(str(tmp_path), "INFO", "unit-test")
    logger.info("a message", extra={"symbol": "AAPL"})
    for handler in logging.getLogger().handlers:
        handler.flush()
    log_file = tmp_path / "unit-test.log"
    assert log_file.exists()
    line = json.loads(log_file.read_text().strip().splitlines()[-1])
    assert line["message"] == "a message"
    assert line["symbol"] == "AAPL"


def test_console_logs_go_to_stderr_not_stdout(tmp_path, capsys):
    """stdout must stay clean so CLI JSON output is pipeable."""
    logger = setup_logging(str(tmp_path), "INFO", "stream-test")
    logger.info("diagnostic line")
    captured = capsys.readouterr()
    assert "diagnostic line" not in captured.out
    assert "diagnostic line" in captured.err
