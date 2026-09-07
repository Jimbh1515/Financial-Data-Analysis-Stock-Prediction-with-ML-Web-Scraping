"""Plotly chart builders for the Streamlit app.

Colors: increasing/decreasing candles use a colorblind-safe teal/orange pair
(ColorBrewer Dark2) instead of the conventional red/green, since red-green
confusion is the most common form of color vision deficiency. Sentiment uses
the same two hues plus neutral gray as a status palette (positive/neutral/
negative), reserved for that meaning only.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UP_COLOR = "#1B9E77"      # teal-green: increasing candle / positive sentiment
DOWN_COLOR = "#D95F02"    # orange: decreasing candle / negative sentiment
NEUTRAL_COLOR = "#7F7F7F"
FORECAST_BG = "rgba(217, 95, 2, 0.06)"
GRID_COLOR = "rgba(128,128,128,0.2)"


def candlestick_with_forecast(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    ticker: str,
    history_days: int = 90,
) -> go.Figure:
    hist = history.tail(history_days)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )

    fig.add_trace(
        go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"],
            name="Historical",
            increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
            increasing_fillcolor=UP_COLOR, decreasing_fillcolor=DOWN_COLOR,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=hist.index, y=hist["Volume"], name="Volume",
            marker_color=NEUTRAL_COLOR, opacity=0.5, showlegend=False,
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Candlestick(
            x=forecast.index, open=forecast["Open"], high=forecast["High"],
            low=forecast["Low"], close=forecast["Close"],
            name="7-Day Forecast",
            increasing_line_color=UP_COLOR, decreasing_line_color=DOWN_COLOR,
            increasing_fillcolor=UP_COLOR, decreasing_fillcolor=DOWN_COLOR,
            opacity=0.55,
            increasing_line_width=2, decreasing_line_width=2,
        ),
        row=1, col=1,
    )

    fig.add_vrect(
        x0=forecast.index[0], x1=forecast.index[-1],
        fillcolor=FORECAST_BG, line_width=0, row=1, col=1,
    )
    fig.add_vline(
        x=hist.index[-1], line_width=1, line_dash="dash",
        line_color=NEUTRAL_COLOR, row=1, col=1,
    )
    fig.add_annotation(
        x=forecast.index[len(forecast) // 2], y=1.0, yref="y domain",
        text="Forecast (predicted, not historical data)", showarrow=False,
        font=dict(size=11, color=NEUTRAL_COLOR), row=1, col=1,
    )

    fig.update_layout(
        title=f"{ticker} — Price History & 7-Day Forecast",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=60, b=10),
        height=560,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price", gridcolor=GRID_COLOR, row=1, col=1)
    fig.update_yaxes(title_text="Volume", gridcolor=GRID_COLOR, row=2, col=1)
    fig.update_xaxes(gridcolor=GRID_COLOR)

    return fig


def sentiment_breakdown_chart(items) -> go.Figure:
    from collections import Counter

    counts = Counter(item.label for item in items)
    labels = ["Positive", "Neutral", "Negative"]
    values = [counts.get(l, 0) for l in labels]
    colors = [UP_COLOR, NEUTRAL_COLOR, DOWN_COLOR]

    fig = go.Figure(
        go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition="outside")
    )
    fig.update_layout(
        title="News Sentiment Breakdown",
        margin=dict(l=10, r=10, t=50, b=10),
        height=320,
        yaxis_title="Number of items",
        showlegend=False,
    )
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def actual_vs_predicted_chart(y_test, y_pred, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test, mode="lines", name="Actual", line=dict(color=NEUTRAL_COLOR, width=2)))
    fig.add_trace(go.Scatter(y=y_pred, mode="lines", name="Predicted", line=dict(color=UP_COLOR, width=2, dash="dot")))
    fig.update_layout(
        title=title,
        xaxis_title="Trading day (holdout period)",
        yaxis_title="Close price",
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def feature_importance_chart(importances) -> go.Figure:
    top = importances.head(10).sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=UP_COLOR))
    fig.update_layout(
        title="What drives the forecast (feature importance)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=340,
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, title_text="Relative importance")
    return fig
