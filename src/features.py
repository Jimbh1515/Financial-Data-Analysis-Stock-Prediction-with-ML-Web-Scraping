"""Technical-indicator feature engineering for the forecasting model.

Builds lag/rolling features from OHLCV data — a generalized version of the
lag-based regression setup in finalmodel.ipynb (which used other tickers'
closing prices as features for HDB). Here we derive everything from the
target ticker's own price history so the same pipeline works for any ticker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAG_DAYS = (1, 2, 3, 5, 10)
MA_WINDOWS = (5, 10, 20)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a feature matrix + 'target' (next day's Close) aligned by date.

    Row at date t holds features computed from data up to and including t,
    and target = Close at t+1, so a model trained on this frame learns to
    predict "tomorrow's close" from "today and earlier".
    """
    out = pd.DataFrame(index=df.index)
    close = df["Close"]

    for lag in LAG_DAYS:
        out[f"close_lag{lag}"] = close.shift(lag - 1)

    for window in MA_WINDOWS:
        out[f"ma{window}"] = close.rolling(window).mean()
        out[f"std{window}"] = close.rolling(window).std()

    out["daily_return"] = close.pct_change()
    out["momentum_5"] = close.pct_change(5)
    out["rsi14"] = _rsi(close)
    out["volume"] = df["Volume"]
    out["volume_ma10"] = df["Volume"].rolling(10).mean()

    out["target"] = close.shift(-1)

    out = out.dropna()
    return out


def intraday_range_stats(df: pd.DataFrame) -> dict:
    """Historical average relationship between Open/High/Low and Close.

    Used to reconstruct a plausible OHLC candle around a forecast Close
    price, since the ML model only predicts the closing price directly.
    All ratios are expressed relative to that day's Close.
    """
    close = df["Close"]
    prev_close = close.shift(1)

    open_pct = ((df["Open"] - prev_close) / prev_close).dropna()
    high_pct = ((df["High"] - df["Close"]) / df["Close"]).dropna()
    low_pct = ((df["Close"] - df["Low"]) / df["Close"]).dropna()

    return {
        "open_offset_mean": float(open_pct.mean()),
        "open_offset_std": float(open_pct.std()),
        "high_offset_mean": float(high_pct.clip(lower=0).mean()),
        "low_offset_mean": float(low_pct.clip(lower=0).mean()),
    }


def latest_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    """Feature row computed from the most recent available data (no target).

    Mirrors build_feature_frame's feature columns exactly, but keeps the
    final row (which build_feature_frame would drop for lacking a target).
    """
    feats = pd.DataFrame(index=df.index)
    close = df["Close"]
    for lag in LAG_DAYS:
        feats[f"close_lag{lag}"] = close.shift(lag - 1)
    for window in MA_WINDOWS:
        feats[f"ma{window}"] = close.rolling(window).mean()
        feats[f"std{window}"] = close.rolling(window).std()
    feats["daily_return"] = close.pct_change()
    feats["momentum_5"] = close.pct_change(5)
    feats["rsi14"] = _rsi(close)
    feats["volume"] = df["Volume"]
    feats["volume_ma10"] = df["Volume"].rolling(10).mean()
    return feats.iloc[[-1]]
