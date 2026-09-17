"""Command-line entry points.

    python -m stockbot.cli check        read-only Alpaca paper connection test
    python -m stockbot.cli backfill     download history into SQLite
    python -m stockbot.cli news         ingest market-wide news (all US symbols)
    python -m stockbot.cli universe     show news vs analysis coverage
    python -m stockbot.cli train        walk-forward validate and select models
    python -m stockbot.cli cycle        run one analysis cycle
    python -m stockbot.cli run          start the scheduler (long-running)
    python -m stockbot.cli status       print current state
    python -m stockbot.cli kill-switch  engage / release the emergency stop

Every command loads settings through :func:`stockbot.config.load_settings`,
which refuses to start unless ALPACA_PAPER is true.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .alpaca_client import AlpacaClient
from .config import ConfigError, load_settings
from .db import Database
from .engine import TradingEngine
from .logging_setup import setup_logging
from .scheduler import SchedulerService, scheduler_status


def _bootstrap(component: str):
    settings = load_settings()
    logger = setup_logging(settings.log_dir, settings.log_level, component)
    db = Database(settings.database_path)
    return settings, logger, db


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_check(_args: argparse.Namespace) -> int:
    """Read-only connection test. Never submits an order."""
    settings, logger, db = _bootstrap("check")
    client = AlpacaClient(settings)
    report = client.connection_test()
    report["config"] = settings.masked()
    report["orders_enabled"] = settings.enable_paper_orders
    report["note"] = "Connection test is read-only; no order was submitted."
    logger.info("Connection test finished", extra={"ok": report["ok"]})
    _emit(report)
    return 0 if report["ok"] else 1


def cmd_backfill(args: argparse.Namespace) -> int:
    settings, logger, db = _bootstrap("backfill")
    client = AlpacaClient(settings)
    engine = TradingEngine(settings, client, db)
    counts = engine.collector.backfill(days=args.days)
    logger.info("Backfill complete", extra={"counts": counts})
    _emit({"stored_bars": counts, "days": args.days or settings.history_days})
    return 0


def cmd_news(args: argparse.Namespace) -> int:
    """Ingest the market-wide news feed. Not limited to the watchlist."""
    settings, logger, db = _bootstrap("news")
    client = AlpacaClient(settings)
    engine = TradingEngine(settings, client, db)
    result = (
        engine.news.backfill(days=args.days) if args.backfill else engine.news.update()
    )
    result["coverage"] = engine.news.coverage()
    logger.info("News command finished", extra={"status": result.get("status")})
    _emit(result)
    return 0 if result.get("status") in {"OK", "DISABLED"} else 1


def cmd_universe(args: argparse.Namespace) -> int:
    """Show which symbols news covers versus which are actually analysed."""
    settings, _logger, db = _bootstrap("universe")
    client = AlpacaClient(settings)
    engine = TradingEngine(settings, client, db)
    if args.refresh_assets:
        engine.universe.refresh_assets(force=True)
    built = engine.universe.build(persist=False)
    _emit(
        {
            "news_coverage": engine.news.coverage(),
            "analysis_universe": built.as_dict(),
            "coverage_report": engine.universe.coverage_report(),
            "note": (
                "News collection is market-wide. Analysis is capped at "
                f"{settings.max_dynamic_symbols} news-driven symbols plus the "
                "configured watchlist, because each analysed symbol needs its own "
                "validated model."
            ),
        }
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    settings, logger, db = _bootstrap("train")
    client = AlpacaClient(settings)
    engine = TradingEngine(settings, client, db)
    if args.refresh:
        engine.collector.update()
    report = engine.ensure_models(force=args.force)
    _emit(report)
    return 0


def cmd_cycle(args: argparse.Namespace) -> int:
    settings, logger, db = _bootstrap("cycle")
    client = AlpacaClient(settings)
    engine = TradingEngine(settings, client, db)
    result = engine.run_cycle(force_when_closed=args.force)
    _emit(result.as_dict())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    settings, logger, db = _bootstrap("scheduler")
    client = AlpacaClient(settings)
    service = SchedulerService(settings, client, db, blocking=True)
    if args.train_on_start:
        service.engine.collector.update()
        service.engine.ensure_models()
    service.start(run_immediately=not args.no_startup_cycle)
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    settings, _logger, db = _bootstrap("status")
    payload = {
        "config": settings.masked(),
        "scheduler": scheduler_status(db),
        "latest_signals": db.latest_signals(),
        "recent_orders": db.recent_orders(limit=10),
        "recent_errors": db.recent_errors(limit=10),
        "kill_switch": bool(db.get_state("kill_switch", False)),
        "news_coverage": db.news_coverage(),
        "analysis_universe": [row["symbol"] for row in db.latest_universe()],
    }
    _emit(payload)
    return 0


def cmd_kill_switch(args: argparse.Namespace) -> int:
    from .risk import RiskEngine

    settings, _logger, db = _bootstrap("kill-switch")
    risk = RiskEngine(settings, db)
    if args.action == "engage":
        risk.engage_kill_switch(actor="cli")
    else:
        risk.release_kill_switch(actor="cli")
    _emit({"kill_switch": bool(db.get_state("kill_switch", False))})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stockbot",
        description="Alpaca paper-trading stock analyser (paper mode only).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="read-only Alpaca paper connection test").set_defaults(
        func=cmd_check
    )

    backfill = sub.add_parser("backfill", help="download bar history into SQLite")
    backfill.add_argument("--days", type=int, default=None)
    backfill.set_defaults(func=cmd_backfill)

    news = sub.add_parser("news", help="ingest market-wide news (all US symbols)")
    news.add_argument("--backfill", action="store_true", help="fetch a full trailing window")
    news.add_argument("--days", type=int, default=None)
    news.set_defaults(func=cmd_news)

    universe = sub.add_parser("universe", help="show news vs analysis coverage")
    universe.add_argument("--refresh-assets", action="store_true")
    universe.set_defaults(func=cmd_universe)

    train = sub.add_parser("train", help="walk-forward validate and select models")
    train.add_argument("--force", action="store_true", help="retrain even if a model is current")
    train.add_argument("--refresh", action="store_true", help="update bars first")
    train.set_defaults(func=cmd_train)

    cycle = sub.add_parser("cycle", help="run one analysis cycle")
    cycle.add_argument(
        "--force", action="store_true", help="run even when the market is closed (no orders)"
    )
    cycle.set_defaults(func=cmd_cycle)

    run = sub.add_parser("run", help="start the scheduler (long-running)")
    run.add_argument("--train-on-start", action="store_true")
    run.add_argument("--no-startup-cycle", action="store_true")
    run.set_defaults(func=cmd_run)

    sub.add_parser("status", help="print current state").set_defaults(func=cmd_status)

    kill = sub.add_parser("kill-switch", help="engage or release the emergency stop")
    kill.add_argument("action", choices=["engage", "release"])
    kill.set_defaults(func=cmd_kill_switch)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
