"""Live market data access via yfinance.

Replaces the deprecated Yahoo Finance CSV-download endpoint used in
Historical_Stock_Data_Yahoo_Finance.ipynb (that `/v7/finance/download/...`
URL no longer works) with the actively-maintained `yfinance` library, so any
ticker symbol can be pulled on demand instead of relying on a pre-downloaded
`result1.csv`.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st
import yfinance as yf

POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX",
    "JPM", "V", "HDB", "INFY", "TCS.NS", "RELIANCE.NS",
]

MIN_ROWS_REQUIRED = 120


class DataUnavailableError(RuntimeError):
    """Raised when a ticker cannot be resolved or has too little history."""


@st.cache_data(ttl=60 * 30, show_spinner=False)
def fetch_history(ticker: str, period: str = "3y", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV history for a ticker.

    Returns a DataFrame indexed by date with columns
    Open, High, Low, Close, Volume. Raises DataUnavailableError if the
    ticker is invalid or has insufficient history for modeling.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise DataUnavailableError("No ticker symbol provided.")

    df = None
    fetch_errors = []

    # yf.download() and Ticker.history() go through slightly different code
    # paths internally and have, in practice, failed independently of each
    # other when Yahoo's undocumented API changes — try both before giving up.
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception as exc:  # network / yfinance errors
        fetch_errors.append(f"download(): {exc}")
        df = None

    if df is None or df.empty:
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        except Exception as exc:
            fetch_errors.append(f"Ticker.history(): {exc}")
            df = None

    if df is None or df.empty:
        detail = f" ({'; '.join(fetch_errors)})" if fetch_errors else ""
        raise DataUnavailableError(
            f"No data returned for '{ticker}'. Check the symbol is correct "
            f"(e.g. 'AAPL', 'RELIANCE.NS', 'TCS.NS'){detail}"
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"

    if len(df) < MIN_ROWS_REQUIRED:
        raise DataUnavailableError(
            f"Only {len(df)} trading days of history for '{ticker}' — "
            f"need at least {MIN_ROWS_REQUIRED} to fit a forecasting model."
        )

    return df


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    """Best-effort fetch of a couple of fundamentals used by the risk tab.

    yfinance's `.info` can be slow or partially missing depending on the
    exchange, so every field is optional and defaults are supplied by the
    caller if a key is absent.
    """
    ticker = ticker.strip().upper()
    try:
        info = yf.Ticker(ticker).get_info()
    except Exception:
        return {}

    return {
        "P/E ratio": info.get("trailingPE"),
        "Debt/Equity ratio": info.get("debtToEquity"),
        "longName": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "currency": info.get("currency"),
    }


def last_trading_date(df: pd.DataFrame) -> dt.date:
    return df.index[-1].date()


def next_trading_days(last_date: dt.date, n: int) -> list[dt.date]:
    """Next n calendar business days (Mon-Fri) after last_date.

    A simple weekday-skip is used rather than a full exchange calendar —
    good enough for a forecast horizon label, not for trading operations.
    """
    days: list[dt.date] = []
    current = last_date
    while len(days) < n:
        current = current + dt.timedelta(days=1)
        if current.weekday() < 5:
            days.append(current)
    return days
