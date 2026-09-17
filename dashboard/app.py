"""Streamlit dashboard.

Read-only over the SQLite store, with one exception: the emergency stop, which
writes the kill-switch flag that the risk engine consults before every order.

The dashboard does not place orders and cannot enable live trading.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import charts, data_access as da  # noqa: E402
from stockbot.alpaca_client import AlpacaClient, AlpacaError  # noqa: E402
from stockbot.config import ConfigError, load_settings  # noqa: E402
from stockbot.db import Database  # noqa: E402
from stockbot.risk import RiskEngine  # noqa: E402
from stockbot.scheduler import scheduler_status  # noqa: E402
from stockbot.signals import AVOID, BUY, HOLD, INSUFFICIENT_EVIDENCE  # noqa: E402

SIGNAL_STYLE = {
    BUY: ("🟢", "#1a7f37"),
    HOLD: ("🟡", "#9a6700"),
    AVOID: ("🔴", "#cf222e"),
    INSUFFICIENT_EVIDENCE: ("⚪", "#57606a"),
}

st.set_page_config(
    page_title="Alpaca Paper Trading Analyser",
    page_icon="📈",
    layout="wide",
)


@st.cache_resource
def get_settings():
    return load_settings()


@st.cache_resource
def get_db(path: str) -> Database:
    return Database(path)


@st.cache_data(ttl=30)
def get_broker_state(_settings) -> dict:
    """Live connection probe, cached briefly so reruns do not hammer the API."""
    try:
        client = AlpacaClient(_settings)
        account = client.get_account()
        clock = client.get_clock()
        return {
            "connected": True,
            "account": account,
            "clock": {
                "is_open": clock["is_open"],
                "next_open": clock["next_open"],
                "next_close": clock["next_close"],
            },
        }
    except (AlpacaError, ConfigError, Exception) as exc:  # noqa: BLE001
        return {"connected": False, "error": str(exc)}


def render_header(settings, broker: dict, db: Database) -> None:
    st.markdown(
        """
        <div style="background:#0b5ed7;color:#fff;padding:14px 18px;border-radius:8px;
                    font-size:1.25rem;font-weight:700;letter-spacing:.06em;
                    display:flex;justify-content:space-between;align-items:center;">
          <span>📄 PAPER TRADING — SIMULATED ORDERS ONLY</span>
          <span style="font-size:.8rem;font-weight:500;opacity:.9;">
            Live trading is not implemented and cannot be enabled from this app
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("Stock Analysis & Paper Trading")

    cols = st.columns(5)
    with cols[0]:
        if broker.get("connected"):
            st.success("Alpaca: connected")
        else:
            st.error("Alpaca: disconnected")
            st.caption(str(broker.get("error", ""))[:120])

    with cols[1]:
        clock = broker.get("clock") or {}
        if not broker.get("connected"):
            st.info("Market: unknown")
        elif clock.get("is_open"):
            close = clock.get("next_close")
            st.success("Market: OPEN")
            st.caption(f"closes {da.age_text(close)}" if close else "")
        else:
            nxt = clock.get("next_open")
            st.warning("Market: CLOSED")
            st.caption(f"opens {da.age_text(nxt)}" if nxt else "")

    with cols[2]:
        if settings.enable_paper_orders:
            st.warning("Paper orders: ENABLED")
        else:
            st.info("Paper orders: disabled")
        st.caption("ENABLE_PAPER_ORDERS")

    with cols[3]:
        killed = bool(db.get_state("kill_switch", False)) or settings.kill_switch
        if killed:
            st.error("Kill switch: ENGAGED")
        else:
            st.success("Kill switch: clear")

    with cols[4]:
        account = (broker.get("account") or {})
        equity = account.get("equity")
        st.metric("Paper equity", f"${equity:,.2f}" if equity else "—")
        st.caption(f"account {account.get('status', 'unknown')}")


def render_sidebar(settings, db: Database) -> None:
    st.sidebar.header("Configuration")
    st.sidebar.write(
        {
            "watchlist": settings.watchlist,
            "benchmark": settings.benchmark_symbol,
            "bar": f"{settings.bar_minutes} min",
            "horizon": settings.horizon_label,
            "data feed": settings.data_feed,
            "min confidence": settings.min_confidence,
            "cost assumption": f"{settings.round_trip_cost_bps:.0f} bps round trip",
        }
    )

    st.sidebar.header("Risk limits")
    st.sidebar.write(
        {
            "max per stock": f"{settings.max_position_pct:.0%}",
            "max portfolio exposure": f"{settings.max_portfolio_exposure_pct:.0%}",
            "daily loss limit": f"{settings.max_daily_loss_pct:.1%}",
            "risk per trade": f"{settings.risk_per_trade_pct:.2%}",
            "max spread": f"{settings.max_spread_bps:.0f} bps",
        }
    )

    st.sidebar.header("🛑 Emergency stop")
    engaged = bool(db.get_state("kill_switch", False))
    risk = RiskEngine(settings, db)

    if engaged:
        st.sidebar.error("Kill switch is ENGAGED. No orders will be submitted.")
        confirm = st.sidebar.checkbox("I want to allow paper orders again")
        if st.sidebar.button("Release kill switch", disabled=not confirm, type="secondary"):
            risk.release_kill_switch(actor="dashboard")
            st.rerun()
    else:
        if st.sidebar.button("ENGAGE KILL SWITCH", type="primary"):
            risk.engage_kill_switch(actor="dashboard")
            st.rerun()
        st.sidebar.caption("Blocks every order immediately. Takes effect on the next cycle.")

    if settings.kill_switch:
        st.sidebar.warning("KILL_SWITCH=true in the environment: orders are blocked regardless.")

    st.sidebar.divider()
    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()


def render_scheduler(db: Database, settings) -> None:
    status = scheduler_status(db)
    cols = st.columns(4)
    with cols[0]:
        if status["alive"]:
            st.success("Scheduler: running")
        else:
            st.error("Scheduler: not running")
        st.caption(f"heartbeat {da.age_text(status.get('heartbeat'))}")
    with cols[1]:
        last = status.get("last_run") or {}
        st.metric("Last update", da.age_text(last.get("started_at")))
        st.caption(f"status: {last.get('status', '—')}")
    with cols[2]:
        st.metric("Next update", da.age_text(status.get("next_run_at")))
        st.caption(f"every {settings.scheduler_interval_minutes} min")
    with cols[3]:
        st.metric("Orders last run", last.get("orders_submitted", 0) or 0)
        st.caption(f"signals: {last.get('signals_generated', 0) or 0}")

    if not status["alive"]:
        st.info(
            "Start the scheduler with `./scripts/run_scheduler.sh` "
            "(or `python -m stockbot.cli run`) for updates every "
            f"{settings.scheduler_interval_minutes} minutes."
        )


def render_signals(db: Database, settings) -> None:
    st.subheader("Signals")
    signals = da.latest_signals(db)
    if signals.empty:
        st.info("No signals yet. Run `python -m stockbot.cli cycle` or start the scheduler.")
        return

    st.caption(
        f"Horizon: {settings.horizon_label} · recomputed every "
        f"{settings.scheduler_interval_minutes} minutes · "
        f"expected return is net of {settings.round_trip_cost_bps:.0f} bps estimated costs."
    )

    for _, row in signals.iterrows():
        icon, colour = SIGNAL_STYLE.get(row["signal"], ("⚪", "#57606a"))
        with st.container(border=True):
            cols = st.columns([1.4, 1, 1, 1, 1.1])
            with cols[0]:
                st.markdown(
                    f"### {row['symbol']}<br>"
                    f"<span style='color:{colour};font-size:1rem;font-weight:700;'>"
                    f"{icon} {row['signal'].replace('_', ' ')}</span>",
                    unsafe_allow_html=True,
                )
            with cols[1]:
                price = row.get("last_price")
                st.metric("Last price", f"${price:,.2f}" if pd.notna(price) else "—")
                st.caption(f"bar {da.age_text(row.get('bar_start'))}")
            with cols[2]:
                exp = row.get("expected_return")
                st.metric(
                    "Expected return",
                    f"{exp * 10_000:+.0f} bps" if pd.notna(exp) else "—",
                )
                st.caption("after estimated costs")
            with cols[3]:
                conf = row.get("confidence")
                st.metric("Confidence", f"{conf:.0%}" if pd.notna(conf) else "—")
                unc = row.get("uncertainty")
                st.caption(f"uncertainty {unc:.3f}" if pd.notna(unc) else "uncertainty —")
            with cols[4]:
                decision = row.get("risk_decision") or "—"
                st.metric("Risk decision", decision)
                qty = row.get("target_qty")
                if pd.notna(qty) and qty:
                    st.caption(f"sized {qty:.0f} sh (${row.get('target_notional', 0):,.0f})")

            st.write(row.get("explanation") or "")
            reasons = row.get("risk_reasons") or []
            if reasons:
                st.caption("Risk engine: " + ", ".join(str(r) for r in reasons))


def render_symbol_detail(db: Database, settings) -> None:
    st.subheader("Charts and contributing features")
    symbol = st.selectbox("Symbol", settings.watchlist, key="detail_symbol")

    bars = da.bars_frame(db, symbol, settings.bar_minutes, limit=400)
    if bars.empty:
        st.info(f"No bars stored for {symbol} yet. Run `python -m stockbot.cli backfill`.")
        return

    benchmark = da.bars_frame(db, settings.benchmark_symbol, settings.bar_minutes, limit=400)
    latest = bars.iloc[-1]
    cols = st.columns(4)
    cols[0].metric("Last close", f"${float(latest['close']):,.2f}")
    cols[1].metric("Bars stored", f"{len(bars):,}")
    cols[2].metric("Latest bar", da.age_text(latest["bar_start"]))
    cols[3].metric("Feed", (latest.get("feed") or settings.data_feed).upper())

    st.plotly_chart(charts.price_and_indicators(bars, benchmark), use_container_width=True)

    preds = da.prediction_history(db, limit=500)
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("**Model probability over time**")
        st.plotly_chart(charts.probability_history(preds, symbol), use_container_width=True)
    with right:
        st.markdown("**Top contributing features (latest prediction)**")
        subset = preds[preds["symbol"] == symbol] if not preds.empty else pd.DataFrame()
        if subset.empty:
            st.info("No prediction recorded for this symbol yet.")
        else:
            latest_pred = subset.iloc[0]
            contributions = latest_pred.get("top_features") or []
            st.plotly_chart(
                charts.feature_contributions(contributions, "signed"),
                use_container_width=True,
            )
            st.caption(
                "Linear models show signed contributions; tree ensembles show "
                "global importances, which indicate influence but not direction."
            )


def render_positions_and_orders(db: Database, broker: dict) -> None:
    st.subheader("Paper positions and orders")
    tabs = st.tabs(["Positions", "Pending orders", "Order history"])

    with tabs[0]:
        positions = da.latest_positions(db)
        if positions.empty:
            st.info("No positions recorded. Positions are snapshotted on every cycle.")
        else:
            display = positions[
                ["symbol", "qty", "avg_entry_price", "current_price", "market_value",
                 "unrealized_pl", "unrealized_plpc"]
            ].copy()
            display.columns = [
                "Symbol", "Qty", "Avg entry", "Current", "Market value", "Unrealised P/L", "P/L %"
            ]
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.caption(f"Snapshot {da.age_text(positions['snapshot_at'].iloc[0])}")

    with tabs[1]:
        pending = da.pending_orders(db)
        if pending.empty:
            st.info("No pending paper orders.")
        else:
            st.dataframe(
                pending[["created_at", "symbol", "side", "qty", "status", "filled_qty",
                         "client_order_id"]],
                use_container_width=True,
                hide_index=True,
            )

    with tabs[2]:
        orders = da.order_history(db, limit=200)
        if orders.empty:
            st.info(
                "No paper orders yet. Orders are only submitted when "
                "ENABLE_PAPER_ORDERS=true and the risk engine approves."
            )
        else:
            st.dataframe(
                orders[["created_at", "symbol", "side", "qty", "status", "filled_qty",
                        "filled_avg_price", "error"]],
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Why was an order placed? (prediction + risk snapshot)"):
                choice = st.selectbox("Client order ID", orders["client_order_id"].tolist())
                record = orders[orders["client_order_id"] == choice].iloc[0]
                left, right = st.columns(2)
                left.markdown("**Prediction that caused it**")
                left.json(da.parse_json(record.get("prediction_snapshot"), {}))
                right.markdown("**Risk decision**")
                right.json(da.parse_json(record.get("risk_snapshot"), {}))


def render_performance(db: Database) -> None:
    st.subheader("Paper-trading performance")
    perf = da.paper_performance(db)
    if not perf.get("available"):
        st.info(f"Performance not available yet: {perf.get('reason')}")
        return

    cols = st.columns(4)
    cols[0].metric("Current equity", f"${perf['current_equity']:,.2f}")
    total = perf.get("total_return")
    cols[1].metric("Return since first snapshot", f"{total:+.2%}" if total is not None else "—")
    dd = perf.get("max_drawdown")
    cols[2].metric("Max drawdown", f"{dd:.2%}" if dd is not None else "—")
    cols[3].metric("Filled orders", perf.get("orders_filled", 0))

    st.plotly_chart(charts.equity_curve(perf["history"]), use_container_width=True)
    st.caption(
        "Measured from stored account snapshots over a short window. This is not "
        "a validated performance record and should not be read as evidence of edge."
    )


def render_backtests(db: Database) -> None:
    st.subheader("Backtest and walk-forward results")
    runs = da.backtest_runs(db)
    if runs.empty:
        st.info("No backtest runs recorded. Run `python -m stockbot.cli train`.")
        return

    st.warning(
        "These are out-of-sample walk-forward results net of estimated costs on a "
        "small sample. They are a sanity check, not evidence of profitability.",
        icon="⚠️",
    )

    rows = []
    for _, run in runs.iterrows():
        metrics = run["metrics"] or {}
        baselines = run["baselines"] or {}
        rows.append(
            {
                "Run at": run["created_at"],
                "Symbol": run["symbol"],
                "Model": run["model_name"],
                "OOS trades": metrics.get("n_trades"),
                "Total return": metrics.get("total_return"),
                "Sharpe": metrics.get("sharpe"),
                "Max DD": metrics.get("max_drawdown"),
                "Win rate": metrics.get("win_rate"),
                "Buy & hold": (baselines.get("buy_and_hold") or {}).get("total_return"),
                "Momentum": (baselines.get("momentum") or {}).get("total_return"),
                "Cost bps": run.get("cost_bps"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("**Selected models**")
    models = da.model_versions(db)
    if models.empty:
        st.info("No model has been selected yet.")
    else:
        display = models[["created_at", "symbol", "model_name", "train_rows", "selected", "notes"]]
        st.dataframe(display, use_container_width=True, hide_index=True)


def render_history(db: Database) -> None:
    st.subheader("Prediction and signal history")
    tabs = st.tabs(["Signals", "Predictions", "Scheduler runs", "Errors"])

    with tabs[0]:
        history = da.signal_history(db)
        if history.empty:
            st.info("No signals recorded yet.")
        else:
            st.dataframe(
                history[["created_at", "symbol", "signal", "confidence", "expected_return",
                         "risk_decision", "last_price"]],
                use_container_width=True, hide_index=True,
            )

    with tabs[1]:
        preds = da.prediction_history(db)
        if preds.empty:
            st.info("No predictions recorded yet.")
        else:
            st.dataframe(
                preds[["created_at", "symbol", "model_name", "prob_up", "expected_return",
                       "uncertainty", "horizon_bars"]],
                use_container_width=True, hide_index=True,
            )

    with tabs[2]:
        runs = da.scheduler_runs(db)
        if runs.empty:
            st.info("No scheduler activity recorded yet.")
        else:
            st.dataframe(
                runs[["started_at", "finished_at", "status", "reason", "symbols_processed",
                      "signals_generated", "orders_submitted", "next_run_at"]],
                use_container_width=True, hide_index=True,
            )

    with tabs[3]:
        errors = da.recent_errors(db)
        if errors.empty:
            st.success("No errors recorded.")
        else:
            st.dataframe(
                errors[["created_at", "component", "symbol", "severity", "message"]],
                use_container_width=True, hide_index=True,
            )


def render_risk_notice() -> None:
    with st.expander("Known risks and limitations — read before trusting any number here"):
        st.markdown(
            """
- **Overfitting.** Several candidate models are compared on the same short history;
  the selected one may simply be the luckiest, not the best.
- **Leakage.** Preprocessing is fitted inside each training fold and an embargo
  separates train from test, but any residual leakage would inflate results.
- **Regime change.** A model fitted on recent months can fail immediately when
  volatility, liquidity or correlation structure shifts.
- **Survivorship bias.** The watchlist is chosen today, by hand, from companies
  that still exist and have done well enough to be worth watching.
- **Data quality.** The IEX feed is a partial view of consolidated volume; bars,
  spreads and volumes differ from SIP data.
- **Paper fills are not real fills.** Alpaca's paper engine does not reproduce
  queue position, partial-fill dynamics or market impact.
            """
        )


def main() -> None:
    try:
        settings = get_settings()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
        return

    db = get_db(settings.database_path)
    broker = get_broker_state(settings)

    render_header(settings, broker, db)
    render_sidebar(settings, db)
    st.divider()
    render_scheduler(db, settings)
    st.divider()
    render_signals(db, settings)
    st.divider()
    render_symbol_detail(db, settings)
    st.divider()
    render_positions_and_orders(db, broker)
    st.divider()
    render_performance(db)
    st.divider()
    render_backtests(db)
    st.divider()
    render_history(db)
    st.divider()
    render_risk_notice()
    st.caption(
        f"Paper trading only · rendered {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC"
    )


main()
