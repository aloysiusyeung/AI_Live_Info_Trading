"""Scheduling.

The cycle runs shortly after every completed 10-minute bar: bars close at
:00, :10, :20 ... so the job fires at ``bar_settle_seconds`` past each of those
minutes, by which point Alpaca has the finished bar.

Market hours are decided by Alpaca's clock, not by a local timetable, so
weekends, holidays, early closes and the DST transitions in New York are all
handled by the broker's own calendar. The job still fires outside market hours;
it simply records a SKIPPED run, which keeps the dashboard's "last run" honest.
"""

from __future__ import annotations

import logging
import signal as signal_module
import threading
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .alpaca_client import AlpacaClient
from .config import Settings
from .db import Database
from .engine import TradingEngine

logger = logging.getLogger(__name__)

ANALYSIS_JOB_ID = "analysis_cycle"
RETRAIN_JOB_ID = "retrain_models"


class SchedulerService:
    """Wraps APScheduler around :class:`~stockbot.engine.TradingEngine`."""

    def __init__(
        self,
        settings: Settings,
        client: AlpacaClient,
        db: Database,
        blocking: bool = True,
    ) -> None:
        self.settings = settings
        self.db = db
        self.engine = TradingEngine(settings, client, db)
        self.tz = ZoneInfo(settings.timezone)
        self._lock = threading.Lock()
        self.scheduler = (
            BlockingScheduler(timezone=self.tz) if blocking else BackgroundScheduler(timezone=self.tz)
        )
        self._configure_jobs()

    # -- job wiring ---------------------------------------------------------
    def _configure_jobs(self) -> None:
        interval = self.settings.scheduler_interval_minutes
        second = min(max(self.settings.bar_settle_seconds, 0), 59)

        self.scheduler.add_job(
            self.run_cycle,
            CronTrigger(
                day_of_week="mon-fri",
                minute=f"*/{interval}",
                second=second,
                timezone=self.tz,
            ),
            id=ANALYSIS_JOB_ID,
            name="10-minute analysis cycle",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=120,
            replace_existing=True,
        )

        # Nightly retrain, well after the close so the day's bars are final.
        self.scheduler.add_job(
            self.retrain,
            CronTrigger(day_of_week="mon-fri", hour=18, minute=15, timezone=self.tz),
            id=RETRAIN_JOB_ID,
            name="daily model retrain",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600,
            replace_existing=True,
        )

    # -- jobs ---------------------------------------------------------------
    def run_cycle(self) -> dict[str, Any]:
        """Guarded cycle run. Overlapping invocations are dropped, not queued."""
        if not self._lock.acquire(blocking=False):
            logger.warning("Previous cycle still running; skipping this tick")
            self.db.log_error(
                "scheduler", "cycle overlap skipped", severity="WARNING"
            )
            return {"status": "SKIPPED", "reason": "overlap"}
        try:
            result = self.engine.run_cycle()
            logger.info(
                "Cycle complete",
                extra={
                    "status": result.status,
                    "market_open": result.market_open,
                    "orders_submitted": result.orders_submitted,
                },
            )
            self._record_next_run()
            return result.as_dict()
        except Exception as exc:  # noqa: BLE001 - the scheduler must survive
            logger.exception("Cycle raised")
            self.db.log_error("scheduler", f"cycle failed: {exc}")
            return {"status": "ERROR", "reason": str(exc)}
        finally:
            self._lock.release()

    def retrain(self) -> dict[str, Any]:
        logger.info("Starting scheduled retrain")
        run_id = self.db.start_scheduler_run(reason="retrain")
        try:
            self.engine.collector.update()
            report = self.engine.ensure_models(force=True)
            self.engine.invalidate_model_cache()
            self.db.finish_scheduler_run(run_id, "OK", reason="retrain", detail=report)
            logger.info("Retrain complete", extra={"report": report})
            return report
        except Exception as exc:  # noqa: BLE001
            logger.exception("Retrain failed")
            self.db.log_error("scheduler", f"retrain failed: {exc}")
            self.db.finish_scheduler_run(run_id, "ERROR", reason=str(exc))
            return {"status": "ERROR", "reason": str(exc)}

    # -- state for the dashboard -------------------------------------------
    def _record_next_run(self) -> None:
        job = self.scheduler.get_job(ANALYSIS_JOB_ID)
        next_run = getattr(job, "next_run_time", None) if job else None
        self.db.set_state(
            "scheduler",
            {
                "alive": True,
                "heartbeat": datetime.now(timezone.utc).isoformat(),
                "next_run_at": next_run.isoformat() if next_run else None,
                "interval_minutes": self.settings.scheduler_interval_minutes,
            },
        )

    def next_run_time(self) -> datetime | None:
        job = self.scheduler.get_job(ANALYSIS_JOB_ID)
        return getattr(job, "next_run_time", None) if job else None

    # -- lifecycle ----------------------------------------------------------
    def start(self, run_immediately: bool = True) -> None:
        logger.info(
            "Starting scheduler",
            extra={
                "interval_minutes": self.settings.scheduler_interval_minutes,
                "timezone": self.settings.timezone,
                "paper": True,
                "orders_enabled": self.settings.enable_paper_orders,
            },
        )
        self.db.set_state("scheduler", {"alive": True, "heartbeat": datetime.now(timezone.utc).isoformat()})
        self._install_signal_handlers()

        if run_immediately:
            self.scheduler.add_job(
                self.run_cycle,
                "date",
                run_date=datetime.now(self.tz) + timedelta(seconds=5),
                id="startup_cycle",
                name="startup cycle",
                replace_existing=True,
            )
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.shutdown()

    def shutdown(self) -> None:
        logger.info("Shutting down scheduler")
        self.db.set_state(
            "scheduler",
            {"alive": False, "heartbeat": datetime.now(timezone.utc).isoformat()},
        )
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass

    def _install_signal_handlers(self) -> None:
        def handler(signum, _frame):
            logger.info("Received signal; shutting down", extra={"signal": signum})
            self.shutdown()

        for sig in (signal_module.SIGINT, signal_module.SIGTERM):
            try:
                signal_module.signal(sig, handler)
            except (ValueError, OSError):  # not on the main thread
                pass


def scheduler_status(db: Database) -> dict[str, Any]:
    """Scheduler health as seen from another process (the dashboard)."""
    state = db.get_state("scheduler", {}) or {}
    runs = db.recent_scheduler_runs(limit=1)
    last = runs[0] if runs else None

    heartbeat = state.get("heartbeat")
    alive = False
    if heartbeat:
        try:
            beat = datetime.fromisoformat(heartbeat)
            if beat.tzinfo is None:
                beat = beat.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - beat).total_seconds()
            alive = bool(state.get("alive")) and age < 3600
        except ValueError:
            alive = False

    return {
        "alive": alive,
        "heartbeat": heartbeat,
        "next_run_at": state.get("next_run_at"),
        "interval_minutes": state.get("interval_minutes"),
        "last_run": last,
    }
