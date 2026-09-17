"""Scheduler wiring and status reporting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from stockbot.scheduler import ANALYSIS_JOB_ID, RETRAIN_JOB_ID, SchedulerService, scheduler_status
from tests.fakes import FakeAlpacaClient


@pytest.fixture
def service(tmp_settings, db):
    client = FakeAlpacaClient(tmp_settings, bars={}, market_open=False)
    return SchedulerService(tmp_settings, client, db, blocking=False)


def test_jobs_registered(service):
    assert service.scheduler.get_job(ANALYSIS_JOB_ID) is not None
    assert service.scheduler.get_job(RETRAIN_JOB_ID) is not None


def test_analysis_job_fires_on_the_bar_boundary(service, tmp_settings):
    job = service.scheduler.get_job(ANALYSIS_JOB_ID)
    fields = {f.name: str(f) for f in job.trigger.fields}
    assert fields["minute"] == f"*/{tmp_settings.scheduler_interval_minutes}"
    assert fields["second"] == str(tmp_settings.bar_settle_seconds)
    assert fields["day_of_week"] == "mon-fri"


def test_scheduler_uses_exchange_timezone(service, tmp_settings):
    assert str(service.scheduler.timezone) == tmp_settings.timezone


def test_cycle_records_a_run_when_market_closed(service, db):
    service.run_cycle()
    run = db.recent_scheduler_runs(1)[0]
    assert run["status"] == "SKIPPED"
    assert run["reason"] == "market_closed"


def test_overlapping_cycles_are_skipped(service, db):
    service._lock.acquire()
    try:
        result = service.run_cycle()
    finally:
        service._lock.release()
    assert result["status"] == "SKIPPED"
    assert result["reason"] == "overlap"


def test_status_reports_dead_scheduler_by_default(db):
    status = scheduler_status(db)
    assert status["alive"] is False


def test_status_reports_alive_on_a_fresh_heartbeat(db):
    db.set_state(
        "scheduler",
        {"alive": True, "heartbeat": datetime.now(timezone.utc).isoformat(),
         "next_run_at": None, "interval_minutes": 10},
    )
    assert scheduler_status(db)["alive"] is True


def test_status_reports_dead_on_a_stale_heartbeat(db):
    old = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
    db.set_state("scheduler", {"alive": True, "heartbeat": old})
    assert scheduler_status(db)["alive"] is False


def test_shutdown_marks_scheduler_not_alive(service, db):
    service.db.set_state("scheduler", {"alive": True, "heartbeat": datetime.now(timezone.utc).isoformat()})
    service.shutdown()
    assert db.get_state("scheduler")["alive"] is False
