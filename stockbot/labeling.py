"""Prediction target construction.

The target is the forward return over ``horizon_bars`` bars, measured from the
current bar's close to the close ``horizon_bars`` ahead, then reduced by the
estimated round-trip cost (spread + slippage, charged on entry and exit).

With the default configuration ``horizon_bars`` is one regular session of
10-minute bars, i.e. a *next-trading-day* horizon that is re-evaluated every ten
minutes.

Two columns come out of this module:

``forward_return``
    Raw log-to-simple forward return, used for regression and for reporting the
    expected return.
``target``
    Binary label: 1 when the cost-adjusted forward return clears the configured
    threshold, else 0.

The forward window is the only place in the codebase that looks ahead, and the
final ``horizon_bars`` rows are dropped because their outcome is unknown.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_forward_return(frame: pd.DataFrame, horizon_bars: int, price_col: str = "close") -> pd.DataFrame:
    """Attach the raw forward simple return over ``horizon_bars``."""
    df = frame.copy()
    price = df[price_col].astype(float)
    df["forward_return"] = price.shift(-horizon_bars) / price - 1.0
    return df


def add_labels(
    frame: pd.DataFrame,
    horizon_bars: int,
    round_trip_cost: float,
    threshold: float = 0.0,
    price_col: str = "close",
) -> pd.DataFrame:
    """Attach ``forward_return``, ``net_forward_return`` and the binary ``target``.

    Parameters
    ----------
    round_trip_cost:
        Fractional cost of a full round trip, e.g. ``0.001`` for 10 bps.
    threshold:
        Extra fractional edge required on top of costs before a bar counts as a
        positive example.
    """
    df = add_forward_return(frame, horizon_bars, price_col=price_col)
    df["net_forward_return"] = df["forward_return"] - round_trip_cost
    df["target"] = np.where(
        df["net_forward_return"].isna(),
        np.nan,
        (df["net_forward_return"] > threshold).astype(float),
    )
    return df


def drop_unresolved(frame: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """Remove trailing rows whose forward window has not completed.

    ``horizon_bars`` rows at the end of the frame necessarily have a NaN target.
    Training on them, or imputing them, would be look-ahead leakage.
    """
    if frame.empty:
        return frame
    resolved = frame.dropna(subset=["target"])
    expected_drop = min(horizon_bars, len(frame))
    actual_drop = len(frame) - len(resolved)
    if actual_drop < expected_drop:
        logger.debug(
            "Fewer unresolved rows than the horizon implies",
            extra={"dropped": actual_drop, "horizon_bars": horizon_bars},
        )
    return resolved.reset_index(drop=True)


def build_supervised_frame(
    featured: pd.DataFrame,
    feature_columns: list[str],
    horizon_bars: int,
    round_trip_cost: float,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Feature matrix + labels, with unresolved and incomplete rows removed."""
    labelled = add_labels(featured, horizon_bars, round_trip_cost, threshold)
    labelled = drop_unresolved(labelled, horizon_bars)
    required = [*feature_columns, "target", "forward_return", "bar_start", "close"]
    present = [c for c in required if c in labelled.columns]
    out = labelled[present].dropna(subset=[*feature_columns, "target"])
    return out.sort_values("bar_start").reset_index(drop=True)
