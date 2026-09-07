"""LiveMint news scraping + sentiment analysis.

The scraping logic (selectors, headline extraction) is a direct refactor of
`Stock Analysis.py` into reusable, cached functions. The sentiment
*classifier* is intentionally different from that script's: `Stock
Analysis.py` uses the `cardiffnlp/twitter-roberta-base-sentiment` transformer
model (torch + transformers, a ~500MB download, meaningful RAM at inference
time). This platform is meant to run on free-tier hosting (e.g. Render's free
web service instance), where that footprint caused out-of-memory restarts —
so this module uses VADER instead: a small, self-contained lexicon/rule-based
sentiment scorer (no model download, negligible memory). It's less nuanced
than a transformer model but fits comfortably in a constrained environment.
The original transformer-based script is untouched in `Stock Analysis.py` if
you want to run that approach directly (needs `transformers`/`torch`
installed separately — no longer in requirements.txt).
"""

from __future__ import annotations

from dataclasses import dataclass

import requests
import streamlit as st
from bs4 import BeautifulSoup

LIVEMINT_URL = "https://www.livemint.com/market"
LABELS = ["Negative", "Neutral", "Positive"]
REQUEST_TIMEOUT = 10
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


@dataclass
class SentimentItem:
    text: str
    source: str  # "headline" or "market_news"
    label: str
    score: float


@st.cache_data(ttl=60 * 15, show_spinner=False)
def scrape_livemint_news() -> tuple[list[str], list[str]]:
    """Scrape headlines and top market-news items from LiveMint.

    Returns (headlines, market_news). Raises requests.RequestException or
    ValueError on failure — callers should handle both.
    """
    response = requests.get(LIVEMINT_URL, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code != 200:
        raise ValueError(f"LiveMint returned HTTP {response.status_code}")

    soup = BeautifulSoup(response.content, "html.parser")

    headlines = []
    for news_block in soup.find_all("li", class_="newsBlock"):
        headline_element = news_block.find("h2")
        if headline_element:
            headlines.append(headline_element.text.strip())

    market_news = []
    h3_elements = soup.select(".market-new-common-collection_contentBox__leEBU h3")
    for i, h3_element in enumerate(h3_elements):
        if i == 7:
            break
        a_tag = h3_element.find("a")
        if a_tag:
            market_news.append(a_tag.text.strip())

    if not headlines and not market_news:
        raise ValueError(
            "No headlines found — LiveMint's page structure may have changed."
        )

    return headlines, market_news


@st.cache_resource(show_spinner=False)
def _load_sentiment_model():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def classify_sentiment(text: str) -> tuple[str, float]:
    analyzer = _load_sentiment_model()
    compound = analyzer.polarity_scores(text)["compound"]

    if compound >= POSITIVE_THRESHOLD:
        label = "Positive"
    elif compound <= NEGATIVE_THRESHOLD:
        label = "Negative"
    else:
        label = "Neutral"

    return label, float(abs(compound))


def analyze_market_sentiment() -> list[SentimentItem]:
    """Scrape LiveMint and classify sentiment for every item found."""
    headlines, market_news = scrape_livemint_news()

    items: list[SentimentItem] = []
    for h in headlines:
        label, score = classify_sentiment(h)
        items.append(SentimentItem(text=h, source="headline", label=label, score=score))
    for n in market_news:
        label, score = classify_sentiment(n)
        items.append(SentimentItem(text=n, source="market_news", label=label, score=score))
    return items
