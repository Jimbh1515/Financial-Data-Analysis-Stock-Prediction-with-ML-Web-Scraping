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
- **📰 News Sentiment tab** — scrapes live market headlines from LiveMint and classifies each with [VADER](https://github.com/cjhutto/vaderSentiment), a lightweight lexicon-based sentiment scorer, visualized as a sentiment breakdown chart and table. (`Stock Analysis.py`'s original transformer-model approach is described in [Original Notebooks & Scripts](#original-notebooks--scripts) below; VADER was used here instead specifically to keep memory usage low enough for free-tier hosting.)
- **⚖️ Risk Assessment tab** — expected return (from the forecast) vs. historical volatility vs. P/E ratio, producing an Invest / Hold / Avoid decision using the same threshold logic as `finalmodel.ipynb`'s `assess_risks()`.
- **🗂️ About / Legacy tab** — explains how each part of the platform maps back to the original notebooks/scripts in this repo.

The app's Python modules live in `src/`: `data.py` (yfinance access), `features.py` (technical indicators), `predictor.py` (forecast model), `sentiment.py` (scraping + sentiment), `risk.py` (risk logic), `charts.py` (Plotly figures).

**Network note:** the platform needs outbound HTTPS access to Yahoo Finance (via `yfinance`) and `livemint.com` (news scraping). If either is unreachable from your network, the corresponding tab will show an error but the rest of the app still works.

**Disclaimer:** the forecast is a statistical projection from historical patterns using a small set of technical features — it is not investment advice, and the risk assessment reproduces a simplified educational heuristic from the original notebook, not a rigorous financial model.

## Deploying to Render

No local Python install needed — Render builds and runs the app entirely in the cloud from this repo. A `render.yaml` blueprint is included at the repo root, pinned to Render's **free** instance plan (`plan: free`) so no payment method should be required to deploy. It also pins the Python version to **3.11.9** (via `PYTHON_VERSION` in `render.yaml` and a `runtime.txt` fallback) — this repo's pinned package versions (`pandas`, etc.) have prebuilt wheels for Python 3.11, so pip installs them directly instead of compiling from source. Without this pin, Render may default to a newer Python for which some of these packages have no prebuilt wheel, forcing a source compile that's prone to failing on the free tier's limited CPU.

The platform's News Sentiment tab uses a lightweight sentiment classifier (VADER) rather than a transformer model, specifically so it fits within free-tier memory — see [Interactive Platform](#interactive-platform) above.

**Option A — Blueprint (one click, uses `render.yaml`):**

1. Push/merge this branch so `render.yaml` is on the branch you want deployed.
2. In the [Render dashboard](https://dashboard.render.com/), click **New +** → **Blueprint**.
3. Connect this GitHub repo and select the branch (`claude/financial-prediction-platform-7fj295`, or `main` once merged).
4. Render reads `render.yaml` and provisions a **Web Service** automatically:
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
5. Click **Apply** / **Create**.

**Option B — Manual Web Service (no blueprint):**

1. **New +** → **Web Service** → connect this repo/branch.
2. Environment: **Python 3**.
3. Instance Type: **Free**.
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`
6. Under **Environment** → **Environment Variables**, add `PYTHON_VERSION` = `3.11.9` (see note above on why this matters).
7. Deploy.

### Things to check once it's live (I could not verify these myself — no Render account access from this session)

- **Cold starts / sleep:** free-tier Render instances spin down after inactivity (Render's own dashboard notes requests can be delayed 50+ seconds after idle). Nothing to fix here, just expected behavior on the free tier.
- **Outbound network to Yahoo Finance / LiveMint:** confirm Render's egress reaches both — normal Render services have unrestricted outbound HTTPS, but this is worth a smoke test after each deploy, since Yahoo Finance's undocumented API has a track record of changing its anti-bot/auth requirements without notice (this already caused one "No data returned" failure during setup — see `src/data.py`'s fallback fetch path).
- **LiveMint markup changes:** the scraping selectors in `src/sentiment.py` are copied from the original `Stock Analysis.py` and will break if LiveMint changes its page structure; the tab will show a clear error rather than crash if that happens.

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
  pip install transformers torch scipy  # not in requirements.txt — see note below
  python "Stock Analysis.py"
  ```
  Scrapes market news headlines from LiveMint and prints sentiment labels and scores for each headline, using the `cardiffnlp/twitter-roberta-base-sentiment` transformer model. This script's dependencies (`transformers`, `torch`, `scipy`) were removed from `requirements.txt` when the platform's own sentiment tab switched to a lighter classifier (VADER) for free-tier hosting compatibility — install them separately to run this script directly.

## How This Project is Useful

This project serves several practical purposes:

- **Investment Decision Support:** By predicting stock prices and assessing investment risks, investors can make informed decisions on buying, selling, or holding stocks.
- **Market Sentiment Analysis:** Analyzing market sentiment from news headlines helps in understanding public perception and potential market trends.
- **Educational Purposes:** The project demonstrates practical applications of machine learning in finance, including data preprocessing, model training, evaluation, and visualization.
- **Automation:** Automated scripts and an interactive app reduce manual effort and streamline information gathering.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
