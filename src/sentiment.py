"""LiveMint news scraping + sentiment analysis.

A refactor of `Stock Analysis.py` into reusable, cached functions: the
scraping and sentiment-classification logic is unchanged (same selectors,
same `cardiffnlp/twitter-roberta-base-sentiment` model, same headline
pre-processing), but it's wrapped so the Streamlit app can call it safely —
LiveMint's markup or availability can change at any time, and the transformer
model is a genuine multi-hundred-MB download on first use, so every failure
mode here degrades gracefully instead of crashing the app.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests
import streamlit as st
from bs4 import BeautifulSoup

LIVEMINT_URL = "https://www.livemint.com/market"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LABELS = ["Negative", "Neutral", "Positive"]
REQUEST_TIMEOUT = 10


@dataclass
class SentimentItem:
    text: str
    source: str  # "headline" or "market_news"
    label: str
    score: float


def _preprocess(text: str) -> str:
    words = []
    for word in text.split(" "):
        if word.startswith("@") and len(word) > 1:
            word = "@user"
        elif word.startswith("http"):
            word = "http"
        words.append(word)
    return " ".join(words)


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
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model


def classify_sentiment(text: str) -> tuple[str, float]:
    from scipy.special import softmax

    tokenizer, model = _load_sentiment_model()
    encoded = tokenizer(_preprocess(text), return_tensors="pt", truncation=True)
    output = model(**encoded)
    scores = softmax(output[0][0].detach().numpy())
    idx = scores.argmax()
    return LABELS[idx], float(scores[idx])


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
