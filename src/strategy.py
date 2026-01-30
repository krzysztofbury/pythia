"""Strategy layer powered by Google Gemini."""
import json
import logging
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, SecretStr

logger = logging.getLogger(__name__)


class TradeSignal(BaseModel):
    """Structured output for the strategy decision."""

    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float  # 0.0 to 1.0
    reasoning: str
    target_price: float | None = None


class Strategist:
    """AI Strategist that reviews market data to form a decision."""

    def __init__(self, api_key: SecretStr, model_name: str = "gemini-2.0-flash") -> None:
        """Initialize the Gemini client."""
        self._client = genai.Client(api_key=api_key.get_secret_value())
        self._model_name = model_name

    def generate_signal(
        self,
        ticker: str,
        current_price: float,
        sentiment: str,
        prediction_odds: float | None,
    ) -> TradeSignal:
        """Ask Gemini to review the data and issue a trade signal."""
        prompt = f"""
        You are a conservative algorithmic crypto trader.
        Review the following data for {ticker}:

        - Current Price: ${current_price}
        - Market Sentiment (Grok/Twitter): "{sentiment}"
        - Prediction Market Odds (Polymarket): {prediction_odds if prediction_odds else 'N/A'}

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

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )

            # Parse JSON
            # response.text is directly available in new SDK usually.
            # or response.candidates[0].content.parts[0].text
            # The new SDK helper often exposes `.text` property on the response object.
            # Let's verify or assume it behaves like the old one in terms of .text accessor.
            # Actually, `google-genai` response object has `.text` property.
            data = json.loads(response.text)
            return TradeSignal(**data)

        except Exception:
            logger.exception("Strategy generation failed")
            return TradeSignal(
                action="HOLD",
                confidence=0.0,
                reasoning="Error in strategy layer.",
            )
