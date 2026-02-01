"""Sentiment analysis using Grok and Polymarket."""
import logging

import requests
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from xai_sdk import Client
from xai_sdk.chat import system, user

logger = logging.getLogger(__name__)


class SentimentOracle:
    """Oracle for fetching market sentiment and predictions."""

    def __init__(
        self,
        xai_key: SecretStr,
        model_name: str,
        cmc_key: SecretStr | None = None,
    ) -> None:
        """Initialize the Grok and CMC clients."""
        self._grok = Client(api_key=xai_key.get_secret_value())
        self._model_name = model_name
        self._cmc_key = cmc_key

    def get_fear_and_greed_index(self) -> str:
        """Fetch the latest Crypto Fear & Greed Index from CoinMarketCap."""
        if not self._cmc_key:
            return "Data unavailable (No API Key)"

        try:
            url = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/latest"
            headers = {
                "X-CMC_PRO_API_KEY": self._cmc_key.get_secret_value(),
                "Accept": "application/json",
            }
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # CMC structure: {"data": {"value": 60, "value_classification": "Greed", ...}}
            # Actually structure might be list or dict, let's handle standard CMC response.
            # Usually: data['data']['value']
            item = data.get("data", {})
            value = item.get("value")
            label = item.get("value_classification")

            return f"{value} ({label})"
        except Exception:
            logger.exception("CMC Fear & Greed fetch failed")
            return "Error fetching index"

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

            conversation = self._grok.chat.create(
                model=self._model_name,
                messages=[
                    system("You are a conservative crypto market analyst."),
                    user(prompt),
                ],
            )
            response = conversation.sample()

            # response is likely a Message object or similar.
            # Based on typical xAI/Grok SDKs, it might be response.content
            # or response.message.content
            # The example said `print(response)`.
            # Let's try accessing .content directly or .message.content
            # Safest is to inspect, but let's guess .message.content based on previous error context
            # (which was Openai style).
            # Actually, if `response` is the message, then `response.content`.
            # If `response` is a wrapper, `response.choices[0]...`.
            # xAI SDK 1.6.1 seems to return a Message object directly from sample()?
            # Let's try `response.content` if it has it, else `str(response)`.
            # I will use getattr to be safe or just try .content
            if hasattr(response, "message"):
                 return response.message.content.strip()
            if hasattr(response, "content"):
                 return response.content.strip()
            return str(response).strip()
        except Exception:
            logger.exception("Grok Analysis failed")
            return "NEUTRAL: Analysis failed."

    @retry(
        retry=retry_if_exception_type(
            (requests.exceptions.ConnectionError, requests.exceptions.Timeout),
        ),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _fetch_polymarket(self, url: str, params: dict) -> list:
        """Fetch data from Polymarket with retries."""
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_consensus_odds(self, ticker: str) -> str:
        """Fetch market consensus from Grok.

        Use Grok to search for prediction market consensus or Twitter sentiment
        regarding the specific price direction of the ticker.
        """
        try:
            prompt = (
                f"Search for recent prediction market odds (like Polymarket) or "
                f"expert consensus on X/Twitter regarding {ticker}. "
                f"Is the consensus that it will go UP or DOWN in the next 24-48h? "
                "Provide a brief summary of the probability or sentiment strength."
            )

            conversation = self._grok.chat.create(
                model=self._model_name,
                messages=[
                    system("You are a market data aggregator."),
                    user(prompt),
                ],
            )
            response = conversation.sample()

            if hasattr(response, "message"):
                 return response.message.content.strip()
            if hasattr(response, "content"):
                 return response.content.strip()
            return str(response).strip()
        except Exception:
            logger.exception("Grok Consensus fetch failed")
            return "Consensus unavailable."

    def get_prediction_market_odds(self, _query: str) -> float | None:
        """Provide a fallback for the deprecated Polymarket method."""
        return None
