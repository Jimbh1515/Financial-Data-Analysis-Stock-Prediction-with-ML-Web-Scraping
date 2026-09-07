"""Interactive financial analysis & prediction platform.

Run with:  streamlit run app.py

Ties together the repo's existing ideas — ML-based price prediction
(finalmodel.ipynb), historical data access (Historical_Stock_Data_Yahoo_
Finance.ipynb), and news sentiment analysis (Stock Analysis.py) — into one
app: pick any ticker, see its historical candlestick chart, get a 7-day
OHLC forecast, check live news sentiment, and get a risk-assessment
decision. See the "About / Legacy" tab for how this maps to the original
notebooks and scripts.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.data import POPULAR_TICKERS, DataUnavailableError, fetch_fundamentals, fetch_history
from src.predictor import MODEL_FACTORIES, run_forecast
from src.risk import build_assessment

st.set_page_config(page_title="Financial Prediction Platform", page_icon="📈", layout="wide")


def sidebar_controls():
    st.sidebar.header("Settings")
    choice = st.sidebar.selectbox(
        "Pick a popular ticker, or choose Custom below",
        ["Custom"] + POPULAR_TICKERS,
        index=1,
    )
    if choice == "Custom":
        ticker = st.sidebar.text_input("Ticker symbol", value="AAPL").strip().upper()
    else:
        ticker = choice

    period = st.sidebar.selectbox(
        "History window used for training", ["1y", "2y", "3y", "5y"], index=2,
    )
    model_name = st.sidebar.selectbox("Forecast model", list(MODEL_FACTORIES.keys()))
    horizon = st.sidebar.slider("Forecast horizon (trading days)", 3, 14, 7)

    st.sidebar.divider()
    st.sidebar.caption(
        "Price data via yfinance (Yahoo Finance). Forecast is a statistical "
        "estimate from historical patterns, not financial advice."
    )
    return ticker, period, model_name, horizon


def render_forecast_tab(ticker: str, period: str, model_name: str, horizon: int):
    if not ticker:
        st.info("Enter a ticker symbol in the sidebar to get started.")
        return None

    try:
        with st.spinner(f"Fetching {ticker} history..."):
            history = fetch_history(ticker, period=period)
    except DataUnavailableError as exc:
        st.error(str(exc))
        return None

    with st.spinner("Training model and forecasting..."):
        result = run_forecast(history, model_name=model_name, horizon=horizon)

    fundamentals = fetch_fundamentals(ticker)
    name = fundamentals.get("longName") or ticker
    st.subheader(f"{name} ({ticker})")

    last_close = history["Close"].iloc[-1]
    forecast_close = result.forecast["Close"].iloc[-1]
    change_pct = (forecast_close - last_close) / last_close

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close", f"{last_close:,.2f}")
    c2.metric(f"Forecast close (+{horizon}d)", f"{forecast_close:,.2f}", f"{change_pct:+.2%}")
    c3.metric("Backtest MAE", f"{result.backtest.mae:.3f}")
    c4.metric("Directional accuracy", f"{result.backtest.directional_accuracy:.1f}%")

    fig = charts.candlestick_with_forecast(result.history, result.forecast, ticker)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Forecast table"):
        st.dataframe(result.forecast.style.format("{:.2f}"), use_container_width=True)

    with st.expander("Model backtest detail & feature importance"):
        b = result.backtest
        st.write(
            f"Evaluated on the most recent {b.n_test} trading days, holdout not used "
            f"for training (chronological split — no look-ahead)."
        )
        st.write(f"MAE: {b.mae:.3f}   RMSE: {b.rmse:.3f}   MAPE: {b.mape:.2f}%")
        if not result.feature_importances.empty:
            st.plotly_chart(
                charts.feature_importance_chart(result.feature_importances),
                use_container_width=True,
            )

    st.caption(
        "Methodology: a regressor predicts tomorrow's Close from lagged prices, "
        "moving averages, RSI and volume; it's applied recursively for the "
        "forecast horizon. Open/High/Low are reconstructed from the ticker's own "
        "historical average intraday range around each forecast Close — not "
        "predicted independently. This is a statistical projection from past "
        "patterns, not a guarantee of future prices."
    )

    return history, result


def render_sentiment_tab():
    st.subheader("Live Market News Sentiment")
    st.caption(
        "Scrapes livemint.com/market headlines and classifies each with the "
        "cardiffnlp/twitter-roberta-base-sentiment transformer model — same "
        "source and model as Stock Analysis.py. First run downloads the model "
        "(~500MB) and may take a while."
    )

    if not st.button("Fetch latest news & analyze sentiment", type="primary"):
        st.info("Click the button to scrape LiveMint and run sentiment analysis.")
        return

    from src.sentiment import analyze_market_sentiment

    try:
        with st.spinner("Scraping LiveMint and classifying sentiment..."):
            items = analyze_market_sentiment()
    except Exception as exc:
        st.error(
            f"Could not complete sentiment analysis: {exc}\n\n"
            "This is a live web-scraping dependency — LiveMint may be "
            "unreachable, rate-limiting, or have changed its page structure."
        )
        return

    if not items:
        st.warning("No news items found.")
        return

    st.plotly_chart(charts.sentiment_breakdown_chart(items), use_container_width=True)

    df = pd.DataFrame(
        [{"Text": i.text, "Source": i.source, "Sentiment": i.label, "Score": round(i.score, 3)} for i in items]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_risk_tab(history, result, ticker: str):
    st.subheader("Investment Risk Assessment")
    st.caption(
        "Adapted from finalmodel.ipynb's assess_risks(): compares the "
        "forecast's expected return and the stock's historical volatility "
        "against fixed thresholds, factoring in P/E ratio."
    )

    if history is None or result is None:
        st.info("Pick a valid ticker in the Forecast tab first.")
        return

    fundamentals = fetch_fundamentals(ticker)
    default_pe = fundamentals.get("P/E ratio")
    default_de = fundamentals.get("Debt/Equity ratio")

    c1, c2 = st.columns(2)
    pe = c1.number_input(
        "P/E ratio (auto-filled from Yahoo Finance if available)",
        value=float(default_pe) if default_pe else 20.0, min_value=0.0,
    )
    de = c2.number_input(
        "Debt/Equity ratio (auto-filled if available)",
        value=float(default_de) if default_de else 1.0, min_value=0.0,
    )

    assessment = build_assessment(result.forecast["Close"], history["Close"], pe, de)

    color = {"Invest": "green", "Hold": "orange", "Avoid": "red"}[assessment.decision]
    st.markdown(f"### Decision: :{color}[{assessment.decision}]")
    st.write(assessment.rationale)

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected return (forecast horizon)", f"{assessment.expected_return:+.2%}")
    c2.metric("Historical volatility (daily std)", f"{assessment.historical_volatility:.2%}")
    c3.metric("P/E ratio used", f"{assessment.pe_ratio:.1f}")

    st.caption(
        "This reproduces the notebook's rule-based decision logic exactly "
        "(same thresholds). It is a simplified educational heuristic, not "
        "financial advice."
    )


def render_legacy_tab():
    st.subheader("What's already in this repository")
    st.write(
        "This platform builds directly on three pieces already in the repo. "
        "Each is summarized below with a live equivalent where applicable."
    )

    with st.expander("📓 Historical_Stock_Data_Yahoo_Finance.ipynb", expanded=False):
        st.write(
            "Builds a Yahoo Finance CSV-download URL "
            "(`query1.finance.yahoo.com/v7/finance/download/...`) for a given "
            "ticker and date range. **That endpoint has since been deprecated "
            "by Yahoo** and no longer serves data — this platform's `src/data.py` "
            "replaces it with the actively-maintained `yfinance` library, which "
            "is what powers ticker selection on the Forecast tab."
        )

    with st.expander("📓 finalmodel.ipynb — Random Forest price prediction & risk", expanded=False):
        st.write(
            "Trains a `RandomForestRegressor` to predict HDB's Close price from "
            "other tickers' Close prices plus sentiment scores (from a fixed "
            "`result1.csv`), reports Mean Squared Error, plots actual vs. "
            "predicted, and defines `assess_risks()` for an Invest/Hold/Avoid "
            "decision. This platform's Forecast tab generalizes that model to "
            "any ticker using the ticker's own lagged/technical features "
            "(`src/predictor.py`), and the Risk tab reuses `assess_risks()` "
            "unchanged (`src/risk.py`)."
        )
        st.write("Live reproduction of the notebook's actual-vs-predicted chart, using the current ticker's backtest:")
        ticker = st.session_state.get("last_ticker")
        result = st.session_state.get("last_result")
        if result is not None:
            b = result.backtest
            st.caption(
                f"Backtest for {ticker}: MAE {b.mae:.3f}, RMSE {b.rmse:.3f}, "
                f"MAPE {b.mape:.2f}%, directional accuracy {b.directional_accuracy:.1f}% "
                f"over the last {b.n_test} trading days."
            )
        else:
            st.info("Pick a ticker in the Forecast tab to populate this chart.")

    with st.expander("🐍 Stock Analysis.py — web scraping & sentiment analysis", expanded=False):
        st.write(
            "Scrapes market news headlines from livemint.com/market with "
            "BeautifulSoup and classifies each with the "
            "`cardiffnlp/twitter-roberta-base-sentiment` transformer model, "
            "printing labels and scores. This platform's News Sentiment tab "
            "runs the same scraping + model live and visualizes the results "
            "(`src/sentiment.py`), rather than only printing to console."
        )


def main():
    st.title("📈 Financial Data Analysis & Stock Prediction Platform")
    st.caption(
        "Pick a ticker, get a 7-day OHLC forecast as an interactive candlestick "
        "chart, check live news sentiment, and get a risk-assessment decision — "
        "all built on the methods already in this repository."
    )

    ticker, period, model_name, horizon = sidebar_controls()

    tab_forecast, tab_sentiment, tab_risk, tab_legacy = st.tabs(
        ["🕯️ Price & Forecast", "📰 News Sentiment", "⚖️ Risk Assessment", "🗂️ About / Legacy"]
    )

    with tab_forecast:
        outcome = render_forecast_tab(ticker, period, model_name, horizon)
        if outcome is not None:
            history, result = outcome
            st.session_state["last_ticker"] = ticker
            st.session_state["last_history"] = history
            st.session_state["last_result"] = result

    with tab_sentiment:
        render_sentiment_tab()

    with tab_risk:
        render_risk_tab(
            st.session_state.get("last_history"),
            st.session_state.get("last_result"),
            st.session_state.get("last_ticker", ticker),
        )

    with tab_legacy:
        render_legacy_tab()


if __name__ == "__main__":
    main()
