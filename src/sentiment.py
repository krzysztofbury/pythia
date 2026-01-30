"""Sentiment analysis using Grok and Polymarket."""
import logging

import requests
from pydantic import SecretStr
from xai_sdk import Client

logger = logging.getLogger(__name__)


class SentimentOracle:
    """Oracle for fetching market sentiment and predictions."""

    def __init__(self, xai_key: SecretStr) -> None:
        """Initialize the Grok client."""
        self._grok = Client(api_key=xai_key.get_secret_value())

    def analyze_ticker_sentiment(self, ticker: str) -> str:
        """Query Grok for a sentiment analysis of the given ticker.

        Returns:
            str: Analysis text (e.g., 'BULLISH: ...')

        """
        try:
            prompt = (
                f"Analyze the current market sentiment for {ticker} (Crypto) based on recent "
                "news and Twitter/X discussions. "
                "Start with 'BULLISH', 'BEARISH', or 'NEUTRAL'. "
                "Then provide a one-sentence reasoning."
            )

            response = self._grok.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a conservative crypto market analyst."},
                    {"role": "user", "content": prompt},
                ],
                model="grok-2-latest",  # Using a recent stable model alias
                stream=False,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            logger.exception("Grok Analysis failed")
            return "NEUTRAL: Analysis failed."

    def get_prediction_market_odds(self, query: str) -> float | None:
        """Fetch probability for a 'Yes' outcome on a Polymarket event matching the query.

        Returns:
            float: Probability (0.0 to 1.0) or None if not found.

        """
        try:
            # Polymarket Gamma API (Public)
            url = "https://gamma-api.polymarket.com/events"
            params = {
                "limit": 3,
                "closed": "false",  # API expects string 'false' often
                "q": query,
            }
            # Search for markets
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # Simple heuristic: Find the first market with prices
            for event in data:
                markets = event.get("markets", [])
                for market in markets:
                    # outcomePrices is a JSON string in some versions, or list in others.
                    # Usually list of strings: ["0.45", "0.55"]
                    prices = market.get("outcomePrices")
                    if prices and len(prices) > 0:
                        try:
                            # Return price of the first outcome (usually 'Yes' or the main outcome)
                            return float(prices[0])
                        except (ValueError, TypeError):
                            continue
            return None  # noqa: TRY300
        except Exception:
            logger.exception("Polymarket fetch failed")
            return None
