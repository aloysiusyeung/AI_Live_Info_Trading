"""Plotly figures for the dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stockbot.features import build_features

# Colour-blind-safe, consistent across every figure.
UP = "#1f77b4"
DOWN = "#d62728"
NEUTRAL = "#7f7f7f"
ACCENT = "#ff7f0e"


def price_and_indicators(bars: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> go.Figure:
    """Candlesticks with moving averages, plus RSI and MACD panels."""
    featured = build_features(bars, benchmark)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.56, 0.22, 0.22],
        vertical_spacing=0.04,
        subplot_titles=("Price and moving averages", "RSI(14)", "MACD"),
    )

    fig.add_trace(
        go.Candlestick(
            x=featured["bar_start"],
            open=featured["open"],
            high=featured["high"],
            low=featured["low"],
            close=featured["close"],
            name="price",
            increasing_line_color=UP,
            decreasing_line_color=DOWN,
        ),
        row=1,
        col=1,
    )

    close = featured["close"].astype(float)
    for window, colour in ((12, ACCENT), (39, NEUTRAL)):
        fig.add_trace(
            go.Scatter(
                x=featured["bar_start"],
                y=close.rolling(window, min_periods=window).mean(),
                name=f"SMA {window}",
                line=dict(color=colour, width=1.4),
            ),
            row=1,
            col=1,
        )

    if "rsi_14" in featured:
        fig.add_trace(
            go.Scatter(x=featured["bar_start"], y=featured["rsi_14"], name="RSI(14)",
                       line=dict(color=UP, width=1.4)),
            row=2, col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color=DOWN, row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=UP, row=2, col=1)

    if "macd_hist" in featured:
        colours = [UP if v >= 0 else DOWN for v in featured["macd_hist"].fillna(0)]
        fig.add_trace(
            go.Bar(x=featured["bar_start"], y=featured["macd_hist"], name="MACD histogram",
                   marker_color=colours),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=featured["bar_start"], y=featured["macd"], name="MACD",
                       line=dict(color=NEUTRAL, width=1.2)),
            row=3, col=1,
        )

    fig.update_layout(
        height=680,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.06, x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    return fig


def equity_curve(history: pd.DataFrame) -> go.Figure:
    """Paper-account equity over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["snapshot_at"],
            y=history["equity"],
            name="paper equity",
            line=dict(color=UP, width=2),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.12)",
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Equity (USD)",
        xaxis_title=None,
        hovermode="x unified",
    )
    return fig


def probability_history(predictions: pd.DataFrame, symbol: str) -> go.Figure:
    """Model probability over time for one symbol."""
    subset = predictions[predictions["symbol"] == symbol].copy()
    fig = go.Figure()
    if subset.empty:
        fig.update_layout(height=260, annotations=[
            dict(text="No predictions recorded yet", showarrow=False, x=0.5, y=0.5,
                 xref="paper", yref="paper")
        ])
        return fig

    subset["created_at"] = pd.to_datetime(subset["created_at"], utc=True, format="mixed")
    subset = subset.sort_values("created_at")
    fig.add_trace(
        go.Scatter(x=subset["created_at"], y=subset["prob_up"], name="P(up)",
                   line=dict(color=UP, width=2), mode="lines+markers")
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color=NEUTRAL)
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="P(positive return)",
        yaxis_range=[0, 1],
        hovermode="x unified",
    )
    return fig


def feature_contributions(contributions: list, kind: str) -> go.Figure:
    """Horizontal bar chart of the top contributing features."""
    fig = go.Figure()
    if not contributions:
        fig.update_layout(height=240, annotations=[
            dict(text="No contributions available", showarrow=False, x=0.5, y=0.5,
                 xref="paper", yref="paper")
        ])
        return fig

    names = [c[0] for c in contributions][::-1]
    values = [float(c[1]) for c in contributions][::-1]
    colours = [UP if v >= 0 else DOWN for v in values] if kind == "signed" else [UP] * len(values)

    fig.add_trace(go.Bar(x=values, y=names, orientation="h", marker_color=colours))
    title = "Signed contribution" if kind == "signed" else "Global importance (unsigned)"
    fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis_title=title,
        yaxis_title=None,
    )
    return fig
