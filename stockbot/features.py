"""Feature engineering.

Every feature is computed from information available **at or before** the close
of the bar it is attached to. Rolling windows use only trailing data, and no
feature uses ``shift(-n)``. The only forward-looking column in the project is
the label, produced in :mod:`stockbot.labeling`.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Ordered list of engineered feature columns handed to the models.
FEATURE_COLUMNS: list[str] = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_39",
    "momentum_12", "momentum_39",
    "sma_ratio_12", "sma_ratio_39", "ema_ratio_12", "sma_cross_12_39",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "atr_14_pct",
    "realised_vol_39", "realised_vol_78", "vol_ratio",
    "relative_volume_39", "dollar_volume_z",
    "vwap_distance", "session_vwap_distance",
    "high_low_range", "close_position_in_range",
    "spy_ret_12", "spy_relative_12", "spy_relative_39", "beta_78",
    "minutes_since_open", "session_progress", "is_first_hour", "is_last_hour",
    "day_of_week",
]

#: Human-readable descriptions used by the explanation generator.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "ret_1": "last bar return",
    "ret_3": "30-minute return",
    "ret_6": "1-hour return",
    "ret_12": "2-hour return",
    "ret_39": "1-session return",
    "momentum_12": "2-hour momentum",
    "momentum_39": "1-session momentum",
    "sma_ratio_12": "price vs 2-hour average",
    "sma_ratio_39": "price vs 1-session average",
    "ema_ratio_12": "price vs 2-hour exponential average",
    "sma_cross_12_39": "short vs long moving-average spread",
    "rsi_14": "RSI(14)",
    "macd": "MACD line",
    "macd_signal": "MACD signal line",
    "macd_hist": "MACD histogram",
    "atr_14_pct": "ATR(14) as % of price",
    "realised_vol_39": "1-session realised volatility",
    "realised_vol_78": "2-session realised volatility",
    "vol_ratio": "short vs long volatility ratio",
    "relative_volume_39": "volume vs 1-session average",
    "dollar_volume_z": "dollar-volume z-score",
    "vwap_distance": "distance from rolling VWAP",
    "session_vwap_distance": "distance from session VWAP",
    "high_low_range": "bar range as % of price",
    "close_position_in_range": "close position within the bar range",
    "spy_ret_12": "SPY 2-hour return",
    "spy_relative_12": "2-hour performance vs SPY",
    "spy_relative_39": "1-session performance vs SPY",
    "beta_78": "rolling beta to SPY",
    "minutes_since_open": "minutes since the open",
    "session_progress": "fraction of the session elapsed",
    "is_first_hour": "first hour of the session",
    "is_last_hour": "last hour of the session",
    "day_of_week": "day of week",
}


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, trailing only."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    # All-gain windows have no loss: RSI is 100 by definition.
    return out.where(avg_loss.notna() & (avg_loss != 0), other=100.0).where(avg_gain.notna())


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average true range (Wilder smoothing)."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _session_key(timestamps: pd.Series, tz: str = "America/New_York") -> pd.Series:
    """Trading-date key in exchange local time (handles DST correctly)."""
    return timestamps.dt.tz_convert(tz).dt.date


#: Minutes in a full regular US session (09:30-16:00 ET).
NOMINAL_SESSION_MINUTES = 390.0


def build_features(
    bars: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    tz: str = "America/New_York",
    session_minutes: dict | None = None,
) -> pd.DataFrame:
    """Engineer the full feature matrix for one symbol.

    Parameters
    ----------
    bars:
        Validated bars for the symbol, oldest first, with a tz-aware
        ``bar_start`` column.
    benchmark:
        Bars for the benchmark (SPY). When absent, the SPY-relative features are
        filled with NaN and later dropped by :func:`prepare_training_frame`.
    session_minutes:
        Optional ``{date: scheduled_session_length_in_minutes}`` map derived from
        Alpaca's calendar, which correctly shortens early-close days. The
        schedule is published in advance, so using it is not look-ahead. Without
        it, every session is assumed to be a full 390 minutes.
    """
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["bar_start", *FEATURE_COLUMNS])

    df = bars.copy().sort_values("bar_start").reset_index(drop=True)
    df["bar_start"] = pd.to_datetime(df["bar_start"], utc=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # --- returns & momentum ------------------------------------------------
    log_close = np.log(close)
    for window in (1, 3, 6, 12, 39):
        df[f"ret_{window}"] = log_close.diff(window)
    df["momentum_12"] = close / close.shift(12) - 1.0
    df["momentum_39"] = close / close.shift(39) - 1.0

    # --- moving averages ---------------------------------------------------
    sma_12 = close.rolling(12, min_periods=12).mean()
    sma_39 = close.rolling(39, min_periods=39).mean()
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    df["sma_ratio_12"] = close / sma_12 - 1.0
    df["sma_ratio_39"] = close / sma_39 - 1.0
    df["ema_ratio_12"] = close / ema_12 - 1.0
    df["sma_cross_12_39"] = sma_12 / sma_39 - 1.0

    # --- oscillators -------------------------------------------------------
    df["rsi_14"] = rsi(close, 14)
    macd_line, macd_signal, macd_hist = macd(close)
    scale = close.replace(0.0, np.nan)
    df["macd"] = macd_line / scale
    df["macd_signal"] = macd_signal / scale
    df["macd_hist"] = macd_hist / scale

    # --- volatility --------------------------------------------------------
    df["atr_14_pct"] = atr(high, low, close, 14) / scale
    bar_ret = log_close.diff()
    df["realised_vol_39"] = bar_ret.rolling(39, min_periods=39).std()
    df["realised_vol_78"] = bar_ret.rolling(78, min_periods=78).std()
    df["vol_ratio"] = df["realised_vol_39"] / df["realised_vol_78"].replace(0.0, np.nan)

    # --- volume ------------------------------------------------------------
    avg_volume_39 = volume.rolling(39, min_periods=39).mean()
    df["relative_volume_39"] = volume / avg_volume_39.replace(0.0, np.nan)
    dollar_volume = close * volume
    dv_mean = dollar_volume.rolling(78, min_periods=78).mean()
    dv_std = dollar_volume.rolling(78, min_periods=78).std()
    df["dollar_volume_z"] = (dollar_volume - dv_mean) / dv_std.replace(0.0, np.nan)

    # --- VWAP --------------------------------------------------------------
    typical = (high + low + close) / 3.0
    if "vwap" in df.columns and df["vwap"].notna().any():
        bar_vwap = df["vwap"].astype(float).fillna(typical)
    else:
        bar_vwap = typical
    rolling_vwap = (
        (bar_vwap * volume).rolling(39, min_periods=12).sum()
        / volume.rolling(39, min_periods=12).sum().replace(0.0, np.nan)
    )
    df["vwap_distance"] = close / rolling_vwap - 1.0

    session = _session_key(df["bar_start"], tz)
    df["_session"] = session
    cum_pv = (bar_vwap * volume).groupby(session).cumsum()
    cum_v = volume.groupby(session).cumsum().replace(0.0, np.nan)
    df["session_vwap_distance"] = close / (cum_pv / cum_v) - 1.0

    # --- bar shape ---------------------------------------------------------
    df["high_low_range"] = (high - low) / scale
    span = (high - low).replace(0.0, np.nan)
    df["close_position_in_range"] = (close - low) / span

    # --- benchmark-relative -------------------------------------------------
    if benchmark is not None and not benchmark.empty:
        bench = benchmark.copy()
        bench["bar_start"] = pd.to_datetime(bench["bar_start"], utc=True)
        bench = bench[["bar_start", "close"]].rename(columns={"close": "bench_close"})
        bench = bench.sort_values("bar_start").drop_duplicates("bar_start")
        df = df.merge(bench, on="bar_start", how="left")
        bench_log = np.log(df["bench_close"].astype(float))
        df["spy_ret_12"] = bench_log.diff(12)
        df["spy_relative_12"] = df["ret_12"] - df["spy_ret_12"]
        df["spy_relative_39"] = df["ret_39"] - bench_log.diff(39)
        bench_ret = bench_log.diff()
        cov = bar_ret.rolling(78, min_periods=78).cov(bench_ret)
        var = bench_ret.rolling(78, min_periods=78).var()
        df["beta_78"] = cov / var.replace(0.0, np.nan)
        df = df.drop(columns=["bench_close"])
    else:
        for col in ("spy_ret_12", "spy_relative_12", "spy_relative_39", "beta_78"):
            df[col] = np.nan

    # --- time of day --------------------------------------------------------
    # The session *open* is the first bar of the day, which is always already
    # observed. The session *close* must NOT be taken from the data, because the
    # last bar of a session is in the future relative to every earlier bar in
    # it — reading it would be look-ahead leakage. The scheduled length is used
    # instead, from Alpaca's calendar when supplied and 390 minutes otherwise.
    local = df["bar_start"].dt.tz_convert(tz)
    session_open = local.groupby(session).transform("min")
    minutes_since_open = (local - session_open).dt.total_seconds() / 60.0

    if session_minutes:
        length = pd.Series(
            [float(session_minutes.get(day, NOMINAL_SESSION_MINUTES)) for day in session],
            index=df.index,
        )
    else:
        length = pd.Series(NOMINAL_SESSION_MINUTES, index=df.index, dtype=float)

    df["minutes_since_open"] = minutes_since_open
    df["session_progress"] = (minutes_since_open / length.replace(0.0, np.nan)).clip(0.0, 1.0)
    df["session_progress"] = df["session_progress"].fillna(0.0)
    df["is_first_hour"] = (minutes_since_open < 60).astype(float)
    df["is_last_hour"] = (minutes_since_open >= (length - 60)).astype(float)
    df["day_of_week"] = local.dt.dayofweek.astype(float)

    df = df.drop(columns=["_session"])
    return df.replace([np.inf, -np.inf], np.nan)


def prepare_training_frame(
    featured: pd.DataFrame, feature_columns: Iterable[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Drop features that are entirely missing and rows without a full vector."""
    columns = [c for c in (feature_columns or FEATURE_COLUMNS) if c in featured.columns]
    usable = [c for c in columns if featured[c].notna().any()]
    dropped = sorted(set(columns) - set(usable))
    if dropped:
        logger.info("Dropping all-NaN features", extra={"features": dropped})
    return featured, usable


def latest_feature_row(featured: pd.DataFrame, feature_columns: Iterable[str]) -> pd.Series | None:
    """Most recent row that has every required feature present."""
    columns = list(feature_columns)
    if featured.empty:
        return None
    complete = featured.dropna(subset=columns)
    if complete.empty:
        return None
    return complete.iloc[-1]
