"""The analysis cycle.

One cycle: refresh bars -> validate -> engineer features -> predict -> signal ->
risk-check -> (optionally) submit a paper order -> persist everything.

The cycle is idempotent per bar: re-running it for the same bar overwrites the
prediction and signal rows rather than creating duplicates, and the risk layer's
duplicate-order window prevents a repeat submission.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from .alpaca_client import AlpacaClient, AlpacaError
from .config import Settings
from .data.collector import BarCollector
from .data.news_collector import NewsCollector
from .db import Database
from .features import build_features, latest_feature_row
from .models.trainer import ModelTrainer
from .news_features import NewsIndex
from .orders import OrderResult, PaperOrderManager
from .risk import RiskContext, RiskDecision, RiskEngine
from .signals import Signal, SignalGenerator
from .universe import UniverseManager

logger = logging.getLogger(__name__)


@dataclass
class SymbolOutcome:
    symbol: str
    signal: Signal | None = None
    risk: RiskDecision | None = None
    order: OrderResult | None = None
    validation: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class CycleResult:
    started_at: datetime
    finished_at: datetime | None = None
    status: str = "OK"
    reason: str = ""
    market_open: bool = False
    outcomes: list[SymbolOutcome] = field(default_factory=list)
    orders_submitted: int = 0
    news: dict[str, Any] = field(default_factory=dict)
    universe: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "status": self.status,
            "reason": self.reason,
            "market_open": self.market_open,
            "orders_submitted": self.orders_submitted,
            "news": self.news,
            "universe": self.universe,
            "training": self.training,
            "symbols": [
                {
                    "symbol": o.symbol,
                    "signal": o.signal.signal if o.signal else None,
                    "risk": o.risk.decision if o.risk else None,
                    "order": o.order.status if o.order else None,
                    "error": o.error,
                }
                for o in self.outcomes
            ],
        }


class TradingEngine:
    """Owns one analysis cycle end to end."""

    def __init__(self, settings: Settings, client: AlpacaClient, db: Database) -> None:
        self.settings = settings
        self.client = client
        self.db = db
        self.collector = BarCollector(settings, client, db)
        self.news = NewsCollector(settings, client, db)
        self.universe = UniverseManager(settings, client, db)
        self.trainer = ModelTrainer(settings, db)
        self.signals = SignalGenerator(settings)
        self.risk = RiskEngine(settings, db)
        self.orders = PaperOrderManager(settings, client, db)
        self._model_cache: dict[str, dict] = {}
        self._session_minutes: dict = {}
        self._session_minutes_fetched: datetime | None = None
        self._news_index: NewsIndex | None = None

    # -- model lifecycle ----------------------------------------------------
    def ensure_models(
        self,
        force: bool = False,
        symbols: list[str] | None = None,
        budget: int | None = None,
    ) -> dict[str, Any]:
        """Train any symbol that has no current model.

        ``budget`` caps how many models this call may train. When the budget is
        spent the remaining symbols are reported as ``deferred`` and pick up on
        a later call — they emit INSUFFICIENT_EVIDENCE until then, which is the
        honest outcome for a symbol with no validated model.
        """
        report: dict[str, Any] = {}
        benchmark = self.collector.load_frame(self.settings.benchmark_symbol)
        targets = symbols if symbols is not None else self.settings.watchlist
        trained = 0
        for symbol in targets:
            existing = self.db.selected_model(symbol)
            if existing and not force and not self._model_is_stale(existing):
                report[symbol] = {"status": "current", "model": existing["model_name"]}
                continue
            if budget is not None and trained >= budget:
                report[symbol] = {
                    "status": "deferred",
                    "reason": f"training budget of {budget} spent this cycle",
                }
                continue
            bars, validation = self.collector.load_validated(symbol)
            if bars.empty:
                report[symbol] = {"status": "skipped", "reason": "no validated bars"}
                continue
            outcome = self.trainer.train_symbol(
                symbol, bars, benchmark, session_minutes=self.session_minutes(),
                news_index=self.news_index(refresh=False),
            )
            trained += 1
            self._model_cache.pop(symbol, None)
            report[symbol] = {
                "status": "trained" if outcome.selected_model else "no_model",
                "model": outcome.selected_model,
                "rows": outcome.n_rows,
                "reason": outcome.reason,
                "verdict": outcome.comparison.get("verdict"),
            }
        return report

    def session_minutes(self) -> dict:
        """Scheduled session lengths from Alpaca's calendar, refreshed daily.

        Falls back to an empty map (i.e. a nominal 390-minute session) if the
        calendar is unavailable, rather than guessing at early closes.
        """
        now = datetime.now(timezone.utc)
        if (
            self._session_minutes_fetched
            and (now - self._session_minutes_fetched) < timedelta(hours=12)
        ):
            return self._session_minutes
        try:
            today = now.astimezone(self.client.tz).date()
            self._session_minutes = self.client.session_minutes_map(
                today - timedelta(days=self.settings.history_days + 5),
                today + timedelta(days=5),
            )
            self._session_minutes_fetched = now
        except Exception as exc:  # noqa: BLE001 - nominal sessions are an acceptable fallback
            logger.warning("Calendar unavailable; assuming full sessions",
                           extra={"error": str(exc)})
            self.db.log_error("engine", f"calendar unavailable: {exc}", severity="WARNING")
        return self._session_minutes

    def news_index(self, refresh: bool = True) -> NewsIndex | None:
        """Market-wide news index for as-of feature computation."""
        if not self.settings.news_enabled:
            return None
        if refresh or self._news_index is None:
            since = datetime.now(timezone.utc) - timedelta(
                days=self.settings.news_backfill_days + 1
            )
            try:
                self._news_index = NewsIndex.from_db(self.db, since=since)
            except Exception as exc:  # noqa: BLE001 - features degrade, cycle continues
                logger.warning("News index build failed", extra={"error": str(exc)})
                self.db.log_error("news", f"index build failed: {exc}", severity="WARNING")
                return self._news_index
        return self._news_index

    def _model_is_stale(self, record: dict) -> bool:
        try:
            created = datetime.fromisoformat(record["created_at"])
        except (KeyError, ValueError):
            return True
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - created
        return age > timedelta(hours=self.settings.retrain_interval_hours)

    def _get_model(self, symbol: str) -> dict | None:
        if symbol not in self._model_cache:
            bundle = self.trainer.load_model(symbol)
            if bundle is None:
                return None
            self._model_cache[symbol] = bundle
        return self._model_cache[symbol]

    def invalidate_model_cache(self) -> None:
        self._model_cache.clear()

    # -- the cycle ----------------------------------------------------------
    def run_cycle(self, force_when_closed: bool = False) -> CycleResult:
        now = datetime.now(timezone.utc)
        result = CycleResult(started_at=now)
        run_id = self.db.start_scheduler_run(reason="scheduled_cycle")

        try:
            clock = self.client.get_clock()
            result.market_open = bool(clock["is_open"])
        except AlpacaError as exc:
            result.status = "ERROR"
            result.reason = f"clock unavailable: {exc}"
            result.finished_at = datetime.now(timezone.utc)
            self.db.log_error("engine", result.reason)
            self.db.finish_scheduler_run(run_id, "ERROR", reason=result.reason)
            return result

        if not result.market_open and not force_when_closed:
            result.status = "SKIPPED"
            result.reason = "market_closed"
            result.finished_at = datetime.now(timezone.utc)
            next_open = clock.get("next_open")
            self.db.finish_scheduler_run(
                run_id,
                "SKIPPED",
                reason="market_closed",
                next_run_at=next_open.isoformat() if next_open else None,
            )
            logger.info("Market closed; cycle skipped", extra={"next_open": str(next_open)})
            return result

        # News first: it is market-wide and feeds both the features and the
        # dynamic universe, so it must land before either is built.
        if self.settings.news_enabled:
            try:
                result.news = self.news.update()
            except Exception as exc:  # noqa: BLE001
                logger.exception("News update failed")
                self.db.log_error("engine", f"news update failed: {exc}", severity="WARNING")
                result.news = {"status": "ERROR", "error": str(exc)}

        try:
            self.universe.refresh_assets()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Asset refresh failed", extra={"error": str(exc)})

        universe = self.universe.build()
        result.universe = universe.as_dict()
        symbols = universe.symbols

        try:
            self.collector.update(symbols=[*symbols, self.settings.benchmark_symbol])
        except Exception as exc:  # noqa: BLE001
            logger.exception("Bar update failed")
            self.db.log_error("engine", f"bar update failed: {exc}")

        # Any newly admitted symbol needs a validated model before it can
        # produce anything other than INSUFFICIENT_EVIDENCE.
        try:
            result.training = self.ensure_models(
                symbols=symbols, budget=self.settings.max_trainings_per_cycle
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Model provisioning failed")
            self.db.log_error("engine", f"ensure_models failed: {exc}", severity="WARNING")

        self.news_index(refresh=True)

        # Shared context fetched once per cycle.
        try:
            account = self.client.get_account()
            positions_list = self.client.get_positions()
            open_orders = self.client.get_open_orders()
        except AlpacaError as exc:
            result.status = "ERROR"
            result.reason = f"account state unavailable: {exc}"
            result.finished_at = datetime.now(timezone.utc)
            self.db.log_error("engine", result.reason)
            self.db.finish_scheduler_run(run_id, "ERROR", reason=result.reason)
            return result

        self.db.snapshot_account(account)
        self.db.snapshot_positions(positions_list)
        positions = {p["symbol"]: p for p in positions_list}

        try:
            quotes = self.client.get_latest_quotes(symbols)
        except AlpacaError as exc:
            logger.warning("Quote fetch failed", extra={"error": str(exc)})
            quotes = {}

        benchmark = self.collector.load_frame(self.settings.benchmark_symbol)
        signals_generated = 0

        for symbol in symbols:
            outcome = self._process_symbol(
                symbol,
                benchmark=benchmark,
                account=account,
                positions=positions,
                open_orders=open_orders,
                quote=quotes.get(symbol),
                market_open=result.market_open,
                now=now,
            )
            result.outcomes.append(outcome)
            if outcome.signal is not None:
                signals_generated += 1
            if outcome.order and outcome.order.submitted:
                result.orders_submitted += 1

        try:
            self.orders.reconcile()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Order reconciliation failed")
            self.db.log_error("engine", f"reconcile failed: {exc}", severity="WARNING")

        result.finished_at = datetime.now(timezone.utc)
        next_run = result.finished_at + timedelta(minutes=self.settings.scheduler_interval_minutes)
        self.db.finish_scheduler_run(
            run_id,
            result.status,
            symbols_processed=len(result.outcomes),
            signals_generated=signals_generated,
            orders_submitted=result.orders_submitted,
            next_run_at=next_run.isoformat(),
            detail=result.as_dict(),
        )
        return result

    def _process_symbol(
        self,
        symbol: str,
        benchmark: pd.DataFrame,
        account: dict,
        positions: dict[str, dict],
        open_orders: list[dict],
        quote: dict | None,
        market_open: bool,
        now: datetime,
    ) -> SymbolOutcome:
        outcome = SymbolOutcome(symbol=symbol)
        try:
            bars, validation = self.collector.load_validated(symbol, now=now)
            outcome.validation = validation.as_dict()

            if not validation.ok or bars.empty:
                reason = ", ".join(validation.reasons) or "no validated data"
                outcome.signal = self.signals.insufficient(symbol, f"data rejected ({reason})")
                self._persist(outcome, has_position=symbol in positions)
                return outcome

            featured = build_features(
                bars, benchmark, tz=self.settings.timezone,
                session_minutes=self.session_minutes(),
                news_index=self.news_index(refresh=False),
                bar_minutes=self.settings.bar_minutes,
                symbol=symbol,
            )
            bundle = self._get_model(symbol)
            if bundle is None:
                outcome.signal = self.signals.insufficient(
                    symbol,
                    "no validated model has been selected for this symbol",
                    bar_start=bars["bar_start"].iloc[-1],
                    last_price=float(bars["close"].iloc[-1]),
                )
                self._persist(outcome, has_position=symbol in positions)
                return outcome

            row = latest_feature_row(featured, bundle["feature_names"])
            if row is None:
                outcome.signal = self.signals.insufficient(
                    symbol,
                    "not enough history to compute a complete feature vector",
                    bar_start=bars["bar_start"].iloc[-1],
                    last_price=float(bars["close"].iloc[-1]),
                )
                self._persist(outcome, has_position=symbol in positions)
                return outcome

            has_position = symbol in positions and abs(positions[symbol].get("qty", 0)) > 0
            realised_vol = _safe_float(row.get("realised_vol_39"))
            signal = self.signals.generate(
                symbol, bundle, row, has_position=has_position, recent_volatility=realised_vol
            )
            outcome.signal = signal

            context = RiskContext(
                account=account,
                positions=positions,
                open_orders=open_orders,
                quote=quote,
                market_open=market_open,
                data_age_seconds=validation.age_seconds,
                avg_dollar_volume=_avg_dollar_volume(bars),
                realised_volatility=realised_vol,
                atr_pct=_safe_float(row.get("atr_14_pct")),
                now=now,
            )
            outcome.risk = self.risk.evaluate(signal, context)

            signal_id, prediction_id = self._persist(outcome, has_position=has_position)

            if outcome.risk.allowed:
                outcome.order = self.orders.submit(
                    signal, outcome.risk, signal_id=signal_id, prediction_id=prediction_id
                )
            else:
                outcome.order = OrderResult(
                    submitted=False,
                    status="no_order",
                    reason=", ".join(outcome.risk.reasons),
                )
        except Exception as exc:  # noqa: BLE001 - one symbol must not kill the cycle
            logger.exception("Symbol processing failed", extra={"symbol": symbol})
            self.db.log_error("engine", str(exc), symbol=symbol)
            outcome.error = str(exc)
            if outcome.signal is None:
                outcome.signal = self.signals.insufficient(symbol, f"processing error: {exc}")
        return outcome

    def _persist(self, outcome: SymbolOutcome, has_position: bool) -> tuple[int | None, int | None]:
        signal = outcome.signal
        if signal is None:
            return None, None

        prediction_id = None
        if signal.prob_up is not None and signal.bar_start is not None:
            prediction_id = self.db.save_prediction(
                symbol=signal.symbol,
                bar_start=signal.bar_start,
                model_version_id=signal.model_version_id,
                model_name=signal.model_name,
                horizon_bars=signal.horizon_bars,
                prob_up=signal.prob_up,
                expected_return=signal.expected_return,
                uncertainty=signal.uncertainty,
                top_features=signal.top_features,
            )

        bar_start = signal.bar_start or datetime.now(timezone.utc)
        signal_id = self.db.save_signal(
            symbol=signal.symbol,
            bar_start=bar_start,
            prediction_id=prediction_id,
            signal=signal.signal,
            confidence=signal.confidence,
            expected_return=signal.expected_return,
            uncertainty=signal.uncertainty,
            last_price=signal.last_price,
            explanation=signal.explanation,
            risk_decision=outcome.risk.decision if outcome.risk else None,
            risk_reasons=outcome.risk.reasons if outcome.risk else [],
            target_qty=outcome.risk.target_qty if outcome.risk else None,
            target_notional=outcome.risk.target_notional if outcome.risk else None,
        )
        return signal_id, prediction_id


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _avg_dollar_volume(bars: pd.DataFrame, window: int = 39) -> float | None:
    """Average dollar volume over the trailing window, used for the liquidity check."""
    if bars.empty or len(bars) < 2:
        return None
    tail = bars.tail(window)
    value = float((tail["close"].astype(float) * tail["volume"].astype(float)).mean())
    # Scale a per-bar average to a session-equivalent figure so the configured
    # threshold can be expressed in familiar daily-volume terms.
    return value * window if np.isfinite(value) else None
