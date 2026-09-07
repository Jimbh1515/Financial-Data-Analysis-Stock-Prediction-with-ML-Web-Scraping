# Financial Data Analysis & Stock Prediction with Machine Learning and Web Scraping

This project integrates financial data analysis, stock price prediction using machine learning, sentiment analysis on market news headlines from LiveMint via web scraping, and investment risk assessment based on predictive models.

It now includes an **interactive Streamlit platform** (`app.py`) that ties all of this together: pick any stock ticker, view its historical candlestick chart, get a 7-day OHLC forecast, check live news sentiment, and get a rule-based Invest/Hold/Avoid risk assessment — all in one app.

## Table of Contents

- [Interactive Platform](#interactive-platform)
- [Deploying to Render](#deploying-to-render)
- [Original Notebooks & Scripts](#original-notebooks--scripts)
- [Installation](#installation)
- [Usage](#usage)
- [How This Project is Useful](#how-this-project-is-useful)
- [License](#license)

## Interactive Platform

Run with:

```sh
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

### Features

- **Ticker selection** — pick from a shortlist of popular tickers or type any symbol yourself (e.g. `AAPL`, `RELIANCE.NS`, `TCS.NS`). Historical OHLCV data is fetched live via [`yfinance`](https://pypi.org/project/yfinance/).
- **🕯️ Price & Forecast tab** — an interactive Plotly candlestick chart of recent price history plus a **7-day-ahead OHLC forecast**, shaded and outlined separately from historical candles so predicted data is never mistaken for actual data. A regressor (Random Forest or Gradient Boosting, selectable in the sidebar) is trained on lagged prices, moving averages, RSI and volume to predict the next closing price, applied recursively for the forecast horizon; Open/High/Low are reconstructed from the ticker's own historical intraday-range statistics around each forecast Close. Backtest metrics (MAE, RMSE, MAPE, directional accuracy) are shown so forecast quality is transparent, not just the chart.
- **📰 News Sentiment tab** — scrapes live market headlines from LiveMint and classifies each with the `cardiffnlp/twitter-roberta-base-sentiment` transformer model, visualized as a sentiment breakdown chart and table.
- **⚖️ Risk Assessment tab** — expected return (from the forecast) vs. historical volatility vs. P/E ratio, producing an Invest / Hold / Avoid decision using the same threshold logic as `finalmodel.ipynb`'s `assess_risks()`.
- **🗂️ About / Legacy tab** — explains how each part of the platform maps back to the original notebooks/scripts in this repo.

The app's Python modules live in `src/`: `data.py` (yfinance access), `features.py` (technical indicators), `predictor.py` (forecast model), `sentiment.py` (scraping + sentiment), `risk.py` (risk logic), `charts.py` (Plotly figures).

**Network note:** the platform needs outbound HTTPS access to Yahoo Finance (via `yfinance`), `livemint.com` (news scraping), and `huggingface.co` (to download the sentiment model on first use, ~500MB). If any of these are unreachable from your network, the corresponding tab will show an error but the rest of the app still works.

**Disclaimer:** the forecast is a statistical projection from historical patterns using a small set of technical features — it is not investment advice, and the risk assessment reproduces a simplified educational heuristic from the original notebook, not a rigorous financial model.

## Deploying to Render

No local Python install needed — Render builds and runs the app entirely in the cloud from this repo. A `render.yaml` blueprint is included at the repo root, pinned to Render's **free** instance plan (`plan: free`) so no payment method should be required to deploy. It also pins the Python version to **3.11.9** (via `PYTHON_VERSION` in `render.yaml` and a `runtime.txt` fallback) — this repo's pinned package versions (`pandas`, `torch`, etc.) have prebuilt wheels for Python 3.11, so pip installs them directly instead of compiling from source. Without this pin, Render may default to a newer Python for which some of these packages have no prebuilt wheel, forcing a source compile that's prone to failing on the free tier's limited CPU.

**Option A — Blueprint (one click, uses `render.yaml`):**

1. Push/merge this branch so `render.yaml` is on the branch you want deployed.
2. In the [Render dashboard](https://dashboard.render.com/), click **New +** → **Blueprint**.
3. Connect this GitHub repo and select the branch (`claude/financial-prediction-platform-7fj295`, or `main` once merged).
4. Render reads `render.yaml` and provisions a **Web Service** automatically:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
5. Click **Apply** / **Create**. First deploy will take a while — `torch` + `transformers` are large installs.

**Option B — Manual Web Service (no blueprint):**

1. **New +** → **Web Service** → connect this repo/branch.
2. Environment: **Python 3**.
3. Instance Type: **Free**.
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
6. Under **Environment** → **Environment Variables**, add `PYTHON_VERSION` = `3.11.9` (see note above on why this matters).
7. Deploy.

### Things to check once it's live (I could not verify these myself — no Render account access from this session)

- **Instance size / RAM on the free tier:** this app installs `torch` + `transformers` for the sentiment tab, which is a meaningfully heavier footprint than a typical Streamlit app, and the free tier is deliberately kept on the smaller side. I don't have a verified, current figure for Render's free-tier RAM limit — please check Render's pricing page directly. **If the app crashes or restarts specifically when the News Sentiment tab is opened, that's almost certainly an out-of-memory kill on the free instance.** The rest of the app (Forecast, Risk, Legacy tabs) doesn't load torch/transformers and should be unaffected either way. If this happens, come back and we can swap the sentiment model for a lightweight lexicon-based method (e.g. VADER, no model download) that fits free-tier RAM comfortably — no need to pay for a bigger instance just for this one tab.
- **Cold starts / sleep:** on lower tiers, Render may spin the service down after inactivity, so the first request after idle can be slow. Confirm current behavior on Render's site for the plan you pick.
- **Model re-download on restart:** the sentiment model (~500MB) downloads from HuggingFace the first time the News Sentiment tab is used after each deploy/restart, since Render's default filesystem isn't guaranteed to persist across deploys. This just means the first click of that tab after a restart will be slow, not that anything is broken.
- **Outbound network:** confirm Render's egress allows Yahoo Finance, LiveMint, and HuggingFace — normal Render services have unrestricted outbound HTTPS, but worth a quick smoke test after deploy.

## Original Notebooks & Scripts

These are the pieces the platform above is built on:

### Stock Price Prediction (`finalmodel.ipynb`)

- **Objective:** Predict closing prices of HDB stock based on historical data and sentiment scores.
- **Techniques Used:**
  - **Random Forest Regressor:** Machine learning algorithm for regression tasks.
  - **Data Preparation:** Cleaning and preprocessing of financial data (`result1.csv`, not included in this repo).
  - **Evaluation:** Mean Squared Error (MSE) calculation to assess prediction accuracy.
  - **Visualization:** Plotting of predicted vs. actual values using matplotlib.
  - Also defines `assess_risks()`, the investment risk heuristic reused by the platform's Risk tab.

### Web Scraping & Sentiment Analysis (`Stock Analysis.py`)

- **Objective:** Extract market news headlines from LiveMint and analyze sentiment using transformer models.
- **Technologies & Libraries:**
  - **Web Scraping:** BeautifulSoup for parsing HTML and extracting news headlines from LiveMint's market section.
  - **Sentiment Analysis:** Transformer model (`cardiffnlp/twitter-roberta-base-sentiment`) for sentiment classification of headlines and news items.
  - **Output:** Printing sentiment labels (Negative, Neutral, Positive) and scores for each headline and news item.

### Historical Data Access (`Historical_Stock_Data_Yahoo_Finance.ipynb`)

- Builds a Yahoo Finance CSV-download URL for a given ticker and date range. **Note:** that specific download endpoint (`query1.finance.yahoo.com/v7/finance/download/...`) has since been deprecated by Yahoo; the platform's `src/data.py` uses the actively-maintained `yfinance` library instead.

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/atchudhansg/Financial-Data-Analysis-Stock-Prediction-with-ML-Web-Scraping.git
   cd Financial-Data-Analysis-Stock-Prediction-with-ML-Web-Scraping
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

- **Interactive platform (recommended):**
  ```sh
  streamlit run app.py
  ```
- **Original notebooks:** open `finalmodel.ipynb` (requires `result1.csv` in the root directory, not included) or `Historical_Stock_Data_Yahoo_Finance.ipynb` in Jupyter.
- **Original scraping script:**
  ```sh
  python "Stock Analysis.py"
  ```
  Scrapes market news headlines from LiveMint and prints sentiment labels and scores for each headline.

## How This Project is Useful

This project serves several practical purposes:

- **Investment Decision Support:** By predicting stock prices and assessing investment risks, investors can make informed decisions on buying, selling, or holding stocks.
- **Market Sentiment Analysis:** Analyzing market sentiment from news headlines helps in understanding public perception and potential market trends.
- **Educational Purposes:** The project demonstrates practical applications of machine learning in finance, including data preprocessing, model training, evaluation, and visualization.
- **Automation:** Automated scripts and an interactive app reduce manual effort and streamline information gathering.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
