"""Strategy layer powered by Google Gemini."""
import json
import logging
from typing import Literal

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel, SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class TradeSignal(BaseModel):
    """Structured output for the strategy decision."""

    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float  # 0.0 to 1.0
    reasoning: str
    target_price: float | None = None


class Strategist:
    """AI Strategist that reviews market data to form a decision."""

    def __init__(self, api_key: SecretStr, model_name: str) -> None:
        """Initialize the Gemini client."""
        self._client = genai.Client(api_key=api_key.get_secret_value())
        self._model_name = model_name
        self._fallbacks = [model_name, "gemini-3-flash-preview", "gemini-2.5-flash"]

    @retry(
        retry=retry_if_exception_type((ClientError, ServerError)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_gemini(self, model: str, prompt: str) -> str:
        """Call Gemini with retries."""
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        return response.text

    def generate_signal(
        self,
        ticker: str,
        current_price: float,
        sentiment: str,
        consensus: str,
        fear_greed: str,
        history: str,
    ) -> TradeSignal:
        """Review data and issue a signal with fallback."""
        prompt = f"""
        You are a conservative algorithmic crypto trader.
        Review the following data for {ticker}:

        - Current Price: ${current_price}
        - Market Sentiment (Grok): "{sentiment}"
        - Consensus/Odds (Grok Search): "{consensus}"
        - Fear & Greed Index: "{fear_greed}"

        Recent Trade History (Context):
        {history}

        Your Goal: Decide whether to BUY, SELL, or HOLD.

        Constraints:
        1. Be conservative. If sentiment is mixed or odds are low, HOLD.
        2. Only BUY if sentiment is clearly BULLISH and/or odds > 0.60.
        3. Provide a JSON response matching this schema:
        {{
            "action": "BUY" | "SELL" | "HOLD",
            "confidence": 0.0 to 1.0,
            "reasoning": "One sentence explanation.",
            "target_price": <float or null>
        }}
        """

        last_error = None
        for model in self._fallbacks:
            try:
                logger.info("Strategy: Attempting %s for %s...", model, ticker)
                text = self._call_gemini(model, prompt)
                data = json.loads(text)
                return TradeSignal(**data)
            except (ClientError, ServerError) as e:
                last_error = e
                logger.warning(
                    "Strategy: %s failed for %s: %s. Trying fallback...",
                    model, ticker, e,
                )
                continue
            except Exception:
                logger.exception("Strategy: Unexpected error with %s", model)
                continue

        # If all fail
        logger.error(
            "Strategy: All Gemini models failed. Last error: %s",
            last_error,
        )
        return TradeSignal(
            action="HOLD",
            confidence=0.0,
            reasoning=f"Strategy layer exhausted all fallbacks. Last error: {last_error}",
        )
