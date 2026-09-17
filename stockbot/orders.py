"""Paper-order submission and reconciliation.

Submission is disabled unless ``ENABLE_PAPER_ORDERS=true`` **and** the risk
engine allowed the trade. Every order carries a unique client order ID and is
written to SQLite together with the prediction and risk decision that produced
it, so any fill can be traced back to its cause.

Order states handled on reconciliation: filled, partially filled, rejected,
canceled, expired.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .alpaca_client import AlpacaClient, AlpacaError
from .config import LiveTradingRefused, Settings
from .db import Database
from .risk import RiskDecision
from .signals import Signal

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {
    "filled", "canceled", "cancelled", "expired", "rejected", "done_for_day", "replaced"
}
PENDING_STATUSES = {
    "new", "accepted", "pending_new", "partially_filled", "accepted_for_bidding",
    "pending_cancel", "pending_replace", "held", "calculated", "stopped", "suspended",
}


@dataclass
class OrderResult:
    submitted: bool
    client_order_id: str | None = None
    broker_order_id: str | None = None
    status: str = "not_submitted"
    reason: str = ""
    error: str | None = None
    db_id: int | None = None


def make_client_order_id(symbol: str) -> str:
    """Unique, human-readable, and stable-length client order ID.

    Alpaca requires uniqueness per account. The random suffix guarantees it even
    if two runs land on the same bar.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"sb-{symbol.lower()}-{stamp}-{suffix}"[:48]


class PaperOrderManager:
    """Submits and reconciles paper orders. Never touches live trading."""

    def __init__(self, settings: Settings, client: AlpacaClient, db: Database) -> None:
        if not settings.paper:
            raise LiveTradingRefused("PaperOrderManager requires paper mode.")
        self.settings = settings
        self.client = client
        self.db = db

    # -- submission ---------------------------------------------------------
    def submit(
        self,
        signal: Signal,
        decision: RiskDecision,
        signal_id: int | None = None,
        prediction_id: int | None = None,
    ) -> OrderResult:
        """Submit the order the risk engine approved, or explain why not."""
        if not self.settings.enable_paper_orders:
            return OrderResult(
                submitted=False,
                status="disabled",
                reason="ENABLE_PAPER_ORDERS is false; order not submitted.",
            )
        if not decision.allowed:
            return OrderResult(
                submitted=False,
                status="blocked",
                reason=f"risk decision {decision.decision}: {', '.join(decision.reasons)}",
            )
        if decision.target_qty < 1:
            return OrderResult(
                submitted=False, status="blocked", reason="approved quantity below 1 share"
            )

        # Final pre-flight re-check against the broker: positions and pending
        # orders may have changed between the risk evaluation and now.
        guard = self._preflight(signal.symbol)
        if guard:
            return OrderResult(submitted=False, status="blocked", reason=guard)

        client_order_id = make_client_order_id(signal.symbol)
        db_id = self.db.record_order(
            client_order_id=client_order_id,
            symbol=signal.symbol,
            side="buy",
            qty=decision.target_qty,
            notional=decision.target_notional,
            status="submitting",
            signal_id=signal_id,
            prediction_id=prediction_id,
            risk_snapshot=decision.as_dict(),
            prediction_snapshot=signal.as_dict(),
        )

        try:
            order = self.client.submit_market_order(
                symbol=signal.symbol,
                qty=decision.target_qty,
                side="buy",
                client_order_id=client_order_id,
            )
        except AlpacaError as exc:
            self.db.update_order_status(client_order_id, status="failed", error=str(exc))
            self.db.log_error("orders", f"submit failed: {exc}", symbol=signal.symbol)
            logger.error("Order submission failed", extra={"symbol": signal.symbol})
            return OrderResult(
                submitted=False,
                client_order_id=client_order_id,
                status="failed",
                reason="submission error",
                error=str(exc),
                db_id=db_id,
            )

        self.db.update_order_status(
            client_order_id,
            status=order.get("status") or "submitted",
            broker_order_id=order.get("id"),
            filled_qty=order.get("filled_qty"),
            filled_avg_price=order.get("filled_avg_price"),
        )
        logger.info(
            "Paper order submitted",
            extra={
                "symbol": signal.symbol,
                "qty": decision.target_qty,
                "client_order_id": client_order_id,
                "status": order.get("status"),
            },
        )
        return OrderResult(
            submitted=True,
            client_order_id=client_order_id,
            broker_order_id=order.get("id"),
            status=order.get("status") or "submitted",
            reason="submitted to paper account",
            db_id=db_id,
        )

    def _preflight(self, symbol: str) -> str | None:
        """Broker-side duplicate guard. Returns a reason string when blocked."""
        try:
            open_orders = self.client.get_open_orders([symbol])
        except AlpacaError as exc:
            return f"could not verify open orders: {exc}"
        if open_orders:
            return f"pending order already exists for {symbol}"
        return None

    # -- reconciliation -----------------------------------------------------
    def reconcile(self, limit: int = 100) -> dict[str, int]:
        """Refresh non-terminal local orders from the broker.

        Handles partial fills (status stays ``partially_filled`` with the filled
        quantity updated), rejections and cancellations.
        """
        counts = {"checked": 0, "updated": 0, "filled": 0, "partial": 0, "rejected": 0,
                  "canceled": 0, "missing": 0}
        rows = self.db.recent_orders(limit=limit)
        for row in rows:
            status = (row.get("status") or "").lower()
            if status in TERMINAL_STATUSES:
                continue
            counts["checked"] += 1
            remote = self.client.get_order_by_client_id(row["client_order_id"])
            if remote is None:
                counts["missing"] += 1
                # A submitting order the broker never acknowledged is dead.
                if status == "submitting":
                    self.db.update_order_status(
                        row["client_order_id"],
                        status="unknown",
                        error="order not found at broker",
                    )
                continue

            new_status = (remote.get("status") or "").lower()
            self.db.update_order_status(
                row["client_order_id"],
                status=new_status,
                broker_order_id=remote.get("id"),
                filled_qty=remote.get("filled_qty"),
                filled_avg_price=remote.get("filled_avg_price"),
            )
            counts["updated"] += 1
            if new_status == "filled":
                counts["filled"] += 1
            elif new_status == "partially_filled":
                counts["partial"] += 1
            elif new_status == "rejected":
                counts["rejected"] += 1
                self.db.log_error(
                    "orders",
                    f"order rejected for {row['symbol']}",
                    severity="WARNING",
                    symbol=row["symbol"],
                    detail={"client_order_id": row["client_order_id"]},
                )
            elif new_status in {"canceled", "cancelled", "expired"}:
                counts["canceled"] += 1
        return counts

    def snapshot(self) -> dict[str, Any]:
        """Persist the current account and positions; used by the scheduler."""
        account = self.client.get_account()
        positions = self.client.get_positions()
        self.db.snapshot_account(account)
        self.db.snapshot_positions(positions)
        return {"account": account, "positions": positions}
