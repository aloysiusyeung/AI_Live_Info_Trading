"""Deterministic risk-management layer.

No machine learning runs in this module. Every check is an explicit rule with a
named reason, evaluated in a fixed order, and the default outcome is **no
trade**. A signal only becomes an order if every single check passes.

Checks, in order:
  1. Emergency kill switch
  2. Paper-order toggle (ENABLE_PAPER_ORDERS)
  3. Account health (blocked / trading suspended)
  4. Market must be open
  5. Signal must be BUY
  6. Confidence floor and uncertainty ceiling
  7. Positive expected return after estimated costs
  8. Data freshness (no trading on stale bars)
  9. Liquidity (average dollar volume) and spread
 10. Daily loss limit
 11. No pending order for the symbol
 12. No duplicate order inside the dedupe window
 13. Existing position must leave room under the per-symbol cap
 14. Portfolio exposure cap
 15. Volatility-based position sizing produces a viable order
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

import numpy as np

from .config import Settings
from .db import Database
from .signals import BUY, Signal

logger = logging.getLogger(__name__)

ALLOW = "ALLOW"
BLOCK = "BLOCK"
NO_ACTION = "NO_ACTION"


@dataclass
class RiskContext:
    """Everything the risk engine needs, gathered by the caller."""

    account: dict = field(default_factory=dict)
    positions: dict[str, dict] = field(default_factory=dict)
    open_orders: Sequence[dict] = field(default_factory=list)
    quote: dict | None = None
    market_open: bool = False
    data_age_seconds: float | None = None
    avg_dollar_volume: float | None = None
    realised_volatility: float | None = None
    atr_pct: float | None = None
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RiskDecision:
    symbol: str
    decision: str = NO_ACTION
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    target_qty: float = 0.0
    target_notional: float = 0.0
    position_fraction: float = 0.0
    sizing_basis: str = ""
    limits: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == ALLOW

    def block(self, reason: str) -> "RiskDecision":
        self.decision = BLOCK
        self.reasons.append(reason)
        return self

    def no_action(self, reason: str) -> "RiskDecision":
        self.decision = NO_ACTION
        self.reasons.append(reason)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "checks": dict(self.checks),
            "target_qty": self.target_qty,
            "target_notional": self.target_notional,
            "position_fraction": self.position_fraction,
            "sizing_basis": self.sizing_basis,
            "limits": self.limits,
        }


class RiskEngine:
    """Applies the deterministic rule set to a signal."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db

    # -- kill switch --------------------------------------------------------
    def kill_switch_engaged(self) -> bool:
        """True when either the env flag or the dashboard toggle is set."""
        if self.settings.kill_switch:
            return True
        return bool(self.db.get_state("kill_switch", False))

    def engage_kill_switch(self, actor: str = "dashboard") -> None:
        self.db.set_state("kill_switch", True)
        self.db.set_state("kill_switch_actor", actor)
        self.db.log_error(
            "risk", "Emergency kill switch ENGAGED", severity="CRITICAL", detail={"actor": actor}
        )
        logger.critical("Kill switch engaged", extra={"actor": actor})

    def release_kill_switch(self, actor: str = "dashboard") -> None:
        self.db.set_state("kill_switch", False)
        self.db.set_state("kill_switch_actor", actor)
        self.db.log_error(
            "risk", "Emergency kill switch released", severity="WARNING", detail={"actor": actor}
        )
        logger.warning("Kill switch released", extra={"actor": actor})

    # -- daily loss ---------------------------------------------------------
    def daily_pnl_fraction(self, account: dict) -> float | None:
        """Today's P&L as a fraction of yesterday's closing equity."""
        equity = account.get("equity")
        last_equity = account.get("last_equity")
        if not equity or not last_equity or last_equity <= 0:
            return None
        return (equity - last_equity) / last_equity

    # -- sizing -------------------------------------------------------------
    def volatility_position_size(
        self, equity: float, price: float, volatility: float | None, atr_pct: float | None
    ) -> tuple[float, float, str]:
        """Risk-parity style sizing capped by the per-symbol allocation limit.

        Target risk is ``risk_per_trade_pct`` of equity. The expected adverse
        move is estimated from ATR% first (an actual price range) and realised
        volatility second. Without either, sizing falls back to the per-symbol
        cap, which is the most conservative fixed allocation available.
        """
        cap_notional = equity * self.settings.max_position_pct
        if price <= 0 or equity <= 0:
            return 0.0, 0.0, "invalid_inputs"

        # Both inputs are per-bar quantities, so scale them to the holding
        # horizon before sizing against a per-trade risk budget.
        horizon_scale = np.sqrt(self.settings.prediction_horizon_bars)
        risk_move = None
        basis = "cap_only"
        if atr_pct and np.isfinite(atr_pct) and atr_pct > 0:
            risk_move = float(atr_pct) * horizon_scale
            basis = "atr"
        elif volatility and np.isfinite(volatility) and volatility > 0:
            risk_move = float(volatility) * horizon_scale
            basis = "realised_volatility"

        if risk_move is None or risk_move <= 0:
            notional = cap_notional
        else:
            risk_budget = equity * self.settings.risk_per_trade_pct
            notional = min(risk_budget / risk_move, cap_notional)

        qty = float(np.floor(notional / price))
        return qty, qty * price, basis

    # -- main evaluation ----------------------------------------------------
    def evaluate(self, signal: Signal, context: RiskContext) -> RiskDecision:
        settings = self.settings
        decision = RiskDecision(symbol=signal.symbol)
        decision.limits = {
            "max_position_pct": settings.max_position_pct,
            "max_portfolio_exposure_pct": settings.max_portfolio_exposure_pct,
            "max_daily_loss_pct": settings.max_daily_loss_pct,
            "min_confidence": settings.min_confidence,
            "max_spread_bps": settings.max_spread_bps,
            "min_avg_dollar_volume": settings.min_avg_dollar_volume,
        }

        def check(name: str, passed: bool, reason: str, fatal: bool = False) -> bool:
            decision.checks[name] = passed
            if not passed:
                (decision.block if fatal else decision.no_action)(reason)
            return passed

        # 1. Kill switch.
        if not check("kill_switch", not self.kill_switch_engaged(),
                     "kill_switch_engaged", fatal=True):
            return decision

        # 2. Paper-order toggle.
        if not check("paper_orders_enabled", settings.enable_paper_orders,
                     "paper_orders_disabled"):
            return decision

        # 3. Account health.
        account = context.account or {}
        healthy = bool(account) and not account.get("account_blocked") and not account.get("trading_blocked")
        if not check("account_healthy", healthy, "account_blocked_or_unavailable", fatal=True):
            return decision

        # 4. Market open.
        if not check("market_open", context.market_open, "market_closed"):
            return decision

        # 5. Actionable signal.
        if not check("signal_actionable", signal.signal == BUY,
                     f"signal_is_{signal.signal.lower()}"):
            return decision

        # 6. Confidence and uncertainty.
        prob = signal.prob_up or 0.0
        if not check("confidence", prob >= settings.min_confidence,
                     f"confidence_{prob:.3f}_below_{settings.min_confidence:.3f}"):
            return decision
        uncertainty = signal.uncertainty
        if not check(
            "uncertainty",
            uncertainty is not None and uncertainty <= settings.uncertainty_max,
            f"uncertainty_{uncertainty}_above_{settings.uncertainty_max}",
        ):
            return decision

        # 7. Positive expected return after costs.
        expected = signal.expected_return or 0.0
        if not check("expected_return", expected > 0,
                     f"expected_return_{expected * 10_000:.1f}bps_not_positive"):
            return decision

        # 8. Data freshness.
        age = context.data_age_seconds
        fresh = age is not None and age <= settings.max_data_age_seconds
        if not check("data_fresh", fresh, f"stale_data_age_{age}", fatal=True):
            return decision

        # 9. Liquidity and spread.
        adv = context.avg_dollar_volume
        if not check(
            "liquidity",
            adv is not None and adv >= settings.min_avg_dollar_volume,
            f"avg_dollar_volume_{adv}_below_{settings.min_avg_dollar_volume}",
        ):
            return decision

        quote = context.quote or {}
        spread_bps = quote.get("spread_bps")
        spread_ok = (
            spread_bps is not None
            and np.isfinite(spread_bps)
            and 0 <= spread_bps <= settings.max_spread_bps
        )
        if not check("spread", spread_ok, f"spread_{spread_bps}_bps_outside_limit"):
            return decision

        # 10. Daily loss limit.
        pnl_fraction = self.daily_pnl_fraction(account)
        within_loss_limit = pnl_fraction is None or pnl_fraction > -abs(settings.max_daily_loss_pct)
        if not check(
            "daily_loss_limit",
            within_loss_limit,
            f"daily_loss_{(pnl_fraction or 0) * 100:.2f}pct_breached_limit",
            fatal=True,
        ):
            return decision

        # 11. No pending order for this symbol.
        pending = [o for o in context.open_orders if o.get("symbol") == signal.symbol]
        if not check("no_pending_order", not pending,
                     f"pending_order_exists_{len(pending)}"):
            return decision

        # 12. Duplicate-order window.
        since = (context.now - timedelta(minutes=settings.duplicate_order_window_minutes)).isoformat()
        recent = [
            o for o in self.db.orders_since(signal.symbol, since)
            if (o.get("status") or "").lower() not in {"rejected", "canceled", "cancelled", "failed"}
        ]
        if not check("no_duplicate_order", not recent,
                     f"duplicate_order_within_{settings.duplicate_order_window_minutes}min"):
            return decision

        # 13. Per-symbol allocation cap given the existing position.
        equity = float(account.get("equity") or 0.0)
        if not check("equity_known", equity > 0, "equity_unavailable", fatal=True):
            return decision

        position = context.positions.get(signal.symbol) or {}
        existing_value = abs(float(position.get("market_value") or 0.0))
        cap_notional = equity * settings.max_position_pct
        room = cap_notional - existing_value
        if not check("position_cap", room > 0,
                     f"position_cap_reached_{existing_value:.0f}_of_{cap_notional:.0f}"):
            return decision

        # 14. Portfolio exposure cap.
        gross_exposure = sum(abs(float(p.get("market_value") or 0.0)) for p in context.positions.values())
        exposure_cap = equity * settings.max_portfolio_exposure_pct
        exposure_room = exposure_cap - gross_exposure
        if not check("portfolio_exposure", exposure_room > 0,
                     f"exposure_{gross_exposure:.0f}_at_cap_{exposure_cap:.0f}"):
            return decision

        # 15. Volatility-based sizing.
        price = signal.last_price or quote.get("mid") or 0.0
        if not check("price_known", bool(price) and price > 0, "price_unavailable"):
            return decision

        qty, notional, basis = self.volatility_position_size(
            equity, float(price), context.realised_volatility, context.atr_pct
        )
        notional = min(notional, room, exposure_room)
        qty = float(np.floor(notional / float(price))) if price > 0 else 0.0
        notional = qty * float(price)
        decision.sizing_basis = basis

        if not check("min_order_size", qty >= 1 and notional >= settings.min_order_notional,
                     f"order_too_small_qty_{qty}_notional_{notional:.2f}"):
            return decision

        buying_power = float(account.get("buying_power") or 0.0)
        if not check("buying_power", notional <= buying_power,
                     f"notional_{notional:.0f}_exceeds_buying_power_{buying_power:.0f}"):
            return decision

        decision.decision = ALLOW
        decision.target_qty = qty
        decision.target_notional = notional
        decision.position_fraction = notional / equity if equity else 0.0
        decision.reasons.append("all_checks_passed")
        logger.info(
            "Risk approved order",
            extra={
                "symbol": signal.symbol,
                "qty": qty,
                "notional": round(notional, 2),
                "sizing_basis": basis,
            },
        )
        return decision
