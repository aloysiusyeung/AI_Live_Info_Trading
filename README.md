# Alpaca Paper-Trading Stock Analyser

A stock-analysis and **paper-trading** application built on Alpaca's official
Python SDK (`alpaca-py`). It pulls 10-minute bars, engineers features, trains
and walk-forward validates several models, produces BUY / HOLD / AVOID /
INSUFFICIENT EVIDENCE signals with plain-English explanations, and — only when
explicitly enabled — submits simulated orders to an Alpaca **paper** account.

> **This application is paper-trading only.**
> It refuses to start unless `ALPACA_PAPER=true`, the trading client is always
> constructed with `paper=True`, `url_override` is never passed, and a runtime
> assertion rejects any non-paper endpoint. There is no live-trading code path,
> and no setting in this project enables one.

> **No claim is made that this is profitable.** The models are validated on a
> few months of 10-minute bars for a handful of hand-picked symbols. That is
> nowhere near enough evidence to conclude an edge exists. Read
> [Known risks](#known-risks-and-limitations) before trusting any number the
> application prints.

---

## Architecture

```
Alpaca (paper trading API + market data, IEX feed by default)
        │
 alpaca_client.py ──── paper-only guard; refuses a live endpoint
        │
 data/collector.py ─── 10-minute bars; a bar is analysed only after
        │              bar_end + BAR_SETTLE_SECONDS
 data/validation.py ── staleness, duplicate timestamps, gaps, OHLC sanity,
        │              non-positive prices, incomplete trailing bars
 features.py ───────── returns, momentum, SMA/EMA, RSI, MACD, ATR, realised
        │              vol, relative volume, VWAP distance, SPY-relative
        │              performance, time of day
 labeling.py ───────── forward return over the horizon, net of spread+slippage
        │
 models/registry.py ── logistic regression, random forest, gradient boosting
 models/walkforward.py chronological expanding-window folds with an embargo;
        │              preprocessing fitted inside each training fold only
 backtest.py ───────── cost-aware equity curve vs momentum and buy-and-hold
 models/trainer.py ─── selects the best model on out-of-sample net return
        │
 signals.py ────────── BUY / HOLD / AVOID / INSUFFICIENT_EVIDENCE, expected
        │              return, P(up), uncertainty, explanation, top features
 risk.py ───────────── deterministic gate, no ML (see Risk engine below)
 orders.py ─────────── paper-only submission, unique client IDs, reconciliation
 scheduler.py ──────── fires shortly after each completed 10-minute bar
 db.py (SQLite) ────── bars, features, predictions, signals, model_versions,
                       backtest_runs, paper_orders, positions, account
                       snapshots, errors, scheduler_runs
 dashboard/app.py ──── Streamlit UI (read-only, plus the emergency stop)
```

Two processes, one database:

| Process | Command | Writes |
|---|---|---|
| Scheduler | `python -m stockbot.cli run` | everything |
| Dashboard | `streamlit run dashboard/app.py` | the kill switch only |

### Prediction horizon

**Default: next trading day.** The target is the forward return from the
current 10-minute bar's close to the close one regular session ahead — 39
ten-minute bars — minus the estimated round-trip cost.

The **signal is recomputed every 10 minutes**; the horizon it predicts over is
one day. Those are independent: a ten-minute refresh cadence on a one-day
forecast. Change the horizon with `PREDICTION_HORIZON_BARS` (e.g. `6` for a
one-hour horizon) — no code change needed. The embargo in walk-forward
validation and the non-overlapping holding rule in the backtest both follow the
configured horizon automatically.

### Leakage controls

Financial time series are never shuffled. Specifically:

1. **Chronological folds only.** Expanding-window splits, ordered in time.
   `expanding_splits` is the only splitter in the project; `train_test_split`
   and `KFold` are not used anywhere.
2. **An embargo** of `PREDICTION_HORIZON_BARS` rows sits between each training
   block and its test block. Without it, the label on the last training rows
   would be computed from prices that also appear in the test set.
3. **Preprocessing inside the pipeline.** The imputer and scaler are pipeline
   steps, so `fit` on a training fold fits them on that fold alone. A test is
   included (`test_scaler_is_fitted_per_fold_not_globally`) that shifts the tail
   of the series by a large constant and asserts the first fold's predictions do
   not move.
4. **No forward-looking features.** `test_no_lookahead_features_change_when_future_is_truncated`
   recomputes every feature on a truncated series and asserts each value is
   bit-for-bit identical. This test caught a genuine bug during development:
   `session_progress` and `is_last_hour` were derived from the session's *last*
   bar, which is in the future relative to every earlier bar in that session.
   They now use the session open plus the scheduled session length from
   Alpaca's calendar.
5. **Unresolved rows dropped.** The final `PREDICTION_HORIZON_BARS` rows have no
   observable outcome and are removed, never imputed.

### Model selection

Candidates are scored on **out-of-sample net return per trade after costs**, and
a candidate is only eligible if it clears every guard:

- at least 15 out-of-sample trades,
- ROC AUC ≥ 0.52 (0.50 is a coin flip),
- positive net return per trade after the round-trip cost.

If nothing qualifies, **no model is selected** and every signal for that symbol
is `INSUFFICIENT_EVIDENCE`. Both baselines — buy-and-hold and trailing momentum
— are computed on the same out-of-sample window, and the recorded verdict states
plainly whether the model beat them.

### Risk engine

Deterministic, no ML, fixed evaluation order, **default deny**. A signal becomes
an order only if all 15 checks pass:

| # | Check | Blocked when |
|---|---|---|
| 1 | Emergency kill switch | `KILL_SWITCH=true` or the dashboard toggle is set |
| 2 | Paper-order toggle | `ENABLE_PAPER_ORDERS` is not true |
| 3 | Account health | account or trading blocked, or unreachable |
| 4 | Market open | Alpaca's clock says closed |
| 5 | Actionable signal | signal is not `BUY` |
| 6 | Confidence / uncertainty | `P(up) < MIN_CONFIDENCE`, or dispersion above `UNCERTAINTY_MAX`, or no uncertainty estimate at all |
| 7 | Expected return | not positive after estimated costs |
| 8 | Data freshness | newest bar older than `MAX_DATA_AGE_SECONDS`, or age unknown |
| 9 | Liquidity and spread | dollar volume below floor, or spread above `MAX_SPREAD_BPS` |
| 10 | Daily loss limit | today's P&L worse than `-MAX_DAILY_LOSS_PCT` |
| 11 | Pending order | an order for the symbol is already open at the broker |
| 12 | Duplicate order | an order for the symbol inside `DUPLICATE_ORDER_WINDOW_MINUTES` |
| 13 | Per-symbol cap | existing position already at `MAX_POSITION_PCT` of equity |
| 14 | Portfolio exposure | gross exposure at `MAX_PORTFOLIO_EXPOSURE_PCT` of equity |
| 15 | Position sizing | volatility sizing yields under 1 share, below `MIN_ORDER_NOTIONAL`, or over buying power |

Sizing is volatility-based: the target is `RISK_PER_TRADE_PCT` of equity divided
by the expected adverse move over the horizon (ATR% preferred, realised
volatility second, both scaled by `sqrt(horizon_bars)`), capped by
`MAX_POSITION_PCT`. With neither volatility input available it falls back to the
per-symbol cap, the most conservative fixed allocation on offer.

---

## Setup

```bash
git clone <this repo> && cd AI_Live_Info_Trading
./scripts/setup.sh          # venv, dependencies, .env scaffold, tests
```

Then put your **paper** keys in `.env` (or export them):

```bash
ALPACA_API_KEY=<your paper key>
ALPACA_SECRET_KEY=<your paper secret>
ALPACA_PAPER=true
```

Get them from <https://app.alpaca.markets/paper/dashboard/overview>. `.env` is
git-ignored; `.env.example` holds placeholders only. Credentials are never
logged — a redacting filter strips them from every log record, and
`Settings.__repr__` masks them.

### Verify the connection (read-only, submits no orders)

```bash
./scripts/check_connection.sh
# or: python -m stockbot.cli check
```

This probes the account, clock, calendar and market data. It never places an
order. If your keys are still placeholders, it says so explicitly rather than
reporting a bare 401.

### Load history and train

```bash
./scripts/bootstrap_data.sh
# or: python -m stockbot.cli backfill && python -m stockbot.cli train --force
```

---

## Running

### Dashboard

```bash
./scripts/run_dashboard.sh              # http://localhost:8501
# or: streamlit run dashboard/app.py
```

Shows connection status, a prominent PAPER TRADING banner, market open/closed
with next open or close, the watchlist, latest prices and data timestamps, each
symbol's signal with expected return and confidence, the plain-English
explanation, top contributing features, price/RSI/MACD charts, current paper
positions, pending orders, paper performance, backtest results, prediction and
order history, last and next scheduled update, scheduler status, and the
emergency stop.

The dashboard caches settings as a Streamlit resource, so restart it after
editing `.env`.

### Scheduler

```bash
./scripts/run_scheduler.sh
# or: python -m stockbot.cli run --train-on-start
```

The cycle fires at `BAR_SETTLE_SECONDS` past every 10th minute, Mon–Fri, in
`America/New_York`. Market hours come from **Alpaca's clock and calendar**, not
a local timetable, so weekends, holidays, early closes and the New York
daylight-saving transitions are handled by the broker's own schedule. Outside
market hours the run is recorded as `SKIPPED`, which keeps the dashboard's
"last update" honest. Models retrain nightly at 18:15 ET.

### Other commands

```bash
python -m stockbot.cli cycle --force          # one cycle, even when closed (no orders)
python -m stockbot.cli status                 # JSON state dump
python -m stockbot.cli kill-switch engage     # emergency stop
python -m stockbot.cli kill-switch release
```

Command output goes to stdout as JSON; logs go to stderr, so
`python -m stockbot.cli status | jq` works.

---

## Enabling simulated paper orders

Orders are **disabled by default**. After you have watched the signals for a
while and read the backtest verdicts:

1. Confirm the connection test passes and `"paper": true` in its output.
2. Confirm the kill switch is clear: `python -m stockbot.cli status`.
3. Set `ENABLE_PAPER_ORDERS=true` in `.env`.
4. Restart the scheduler.

Every submission is still gated by all 15 risk checks, so enabling the flag does
not by itself cause a trade. Each order gets a unique client order ID and is
stored with the prediction and risk decision that caused it — visible in the
dashboard under *Order history → Why was an order placed?*

To stop immediately, engage the kill switch (dashboard sidebar or
`./scripts/kill_switch.sh engage`). It takes effect on the next cycle and
outranks every other setting.

**There is no path from here to live trading.** `ENABLE_PAPER_ORDERS` controls
simulated orders against a paper account and nothing else.

---

## Deployment: keeping the scheduler running

### systemd (recommended on Linux)

`/etc/systemd/system/stockbot.service`:

```ini
[Unit]
Description=Alpaca paper-trading analyser scheduler
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=stockbot
WorkingDirectory=/opt/stockbot
EnvironmentFile=/opt/stockbot/.env
ExecStart=/opt/stockbot/.venv/bin/python -m stockbot.cli run --train-on-start
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

# The process needs no privileges beyond its own directory.
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths=/opt/stockbot/data /opt/stockbot/logs /opt/stockbot/models_store

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now stockbot
sudo systemctl status stockbot
journalctl -u stockbot -f
```

The dashboard as a second unit:

```ini
[Service]
ExecStart=/opt/stockbot/.venv/bin/streamlit run dashboard/app.py \
  --server.port 8501 --server.address 127.0.0.1 --server.headless true
Restart=always
```

Bind the dashboard to `127.0.0.1` and put it behind a reverse proxy with
authentication — it exposes the emergency stop and your positions.

### Docker Compose

```yaml
services:
  scheduler:
    build: .
    command: python -m stockbot.cli run --train-on-start
    env_file: .env
    volumes: ["./data:/app/data", "./logs:/app/logs", "./models_store:/app/models_store"]
    restart: unless-stopped

  dashboard:
    build: .
    command: >
      streamlit run dashboard/app.py --server.port 8501
      --server.address 0.0.0.0 --server.headless true
    env_file: .env
    ports: ["127.0.0.1:8501:8501"]
    volumes: ["./data:/app/data", "./logs:/app/logs", "./models_store:/app/models_store"]
    depends_on: [scheduler]
    restart: unless-stopped
```

Both containers must share the `data/` volume — they talk through the SQLite
file, which runs in WAL mode for concurrent readers.

### supervisor

```ini
[program:stockbot-scheduler]
command=/opt/stockbot/.venv/bin/python -m stockbot.cli run --train-on-start
directory=/opt/stockbot
autostart=true
autorestart=true
stopsignal=TERM
stderr_logfile=/var/log/stockbot/scheduler.err.log
```

The scheduler handles `SIGINT`/`SIGTERM` and marks itself not-alive on shutdown,
so the dashboard shows "Scheduler: not running" within a minute of a stop. The
dashboard treats a heartbeat older than an hour as dead.

### Operational notes

- **Back up `data/stockbot.sqlite`.** It holds every prediction, signal and
  order.
- **Watch the Errors tab** (or `python -m stockbot.cli status`). Data-validation
  failures and rejected orders land there.
- **Log rotation** is built in: 10 MB per file, 5 files, JSON lines.
- **Disk growth** is dominated by `bars` and `positions`; a single-symbol year of
  10-minute bars is roughly 1 MB.

---

## Tests

```bash
python -m pytest              # 211 tests
python -m pytest --cov=stockbot --cov-report=term-missing
```

No test touches the network. Alpaca is replaced by an in-process fake, and bar
data is a seeded random walk, clearly labelled as synthetic. **No test derives a
performance claim from synthetic data** — they assert code behaviour only.

The suite covers, among other things: the paper-mode refusal for every
non-`true` value of `ALPACA_PAPER`; that `url_override` is never passed and a
live endpoint raises; feature look-ahead; per-fold preprocessing; split ordering
and the embargo; every one of the 15 risk rules; order rejection, cancellation
and partial fills; and a headless render of the Streamlit app via `AppTest`.

---

## Known risks and limitations

These are real and they are not resolved by anything in this repository.

**Overfitting.** Three model families are compared on a few months of bars for a
handful of symbols. The winner may simply be the luckiest. The guards (minimum
trade count, AUC floor, positive net return) filter the worst cases but cannot
turn a small sample into evidence. Fold-level net returns and their standard
deviation are recorded so you can see how unstable the result is across folds.

**Leakage.** The controls above are deliberate and tested, and one genuine bug
was caught by them during development. That is evidence the tests work, not proof
that no leakage remains. Any residual leakage inflates every backtest number.

**Regime change.** A model fitted on recent months can fail the day volatility,
liquidity or correlation structure shifts. Walk-forward validation measures
performance across *past* regimes, which says nothing about the next one. There
is no regime detection here.

**Survivorship bias.** `WATCHLIST` is chosen today, by hand, from companies that
still exist and are well known enough to be worth watching. Any backtest over
such a list is biased upward, and this application does not correct for it.

**Data quality.** The default IEX feed is a partial view of consolidated volume,
so volumes, VWAP and spreads all differ from SIP data — which matters directly,
because the liquidity and spread checks are calibrated in those units. Bars can
also be revised after publication; the store upserts, so a revision overwrites,
but a model trained before a revision is not retrained because of it.

**Cost model.** Spread and slippage are flat configured constants
(`ESTIMATED_SPREAD_BPS`, `ESTIMATED_SLIPPAGE_BPS`). Real costs vary with time of
day, size and volatility. Market impact is not modelled at all. If the true cost
is higher than configured, every backtest is optimistic.

**Paper fills are not real fills.** Alpaca's paper engine does not reproduce
queue position, partial-fill dynamics, or the effect of your own order on the
price. Paper performance is an upper bound.

**Short samples.** `paper_performance` on the dashboard is computed from however
many account snapshots exist, which early on may be minutes apart. It is a
monitoring aid, not a track record.

**Long-only, single-name, no portfolio optimisation.** Signals are independent
per symbol; there is no correlation-aware sizing, no shorting, no hedging, and
exits are not modelled — the risk engine sizes entries and the caps limit
accumulation, but nothing closes a position.

**No walk-forward on hyperparameters.** Model hyperparameters are fixed
constants in `models/registry.py`. They were not tuned, which avoids one kind of
overfitting and accepts a worse fit in exchange.

---

## Licence and disclaimer

For research and education. Nothing here is financial advice. Alpaca paper
trading involves no real money; do not treat any output as a reason to risk
real money.
