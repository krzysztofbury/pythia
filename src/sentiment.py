"""Sentiment analysis using Grok and Polymarket."""

import json
import logging
import re
import threading

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

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

# Circuit breaker: skip Grok for N cycles after consecutive failures
_MAX_CONSECUTIVE_FAILURES = 3


class SentimentOracle:
    """Oracle for fetching market sentiment and predictions."""

    def __init__(
        self,
        xai_key: SecretStr,
        model_name: str,
        cmc_key: SecretStr | None = None,
        warp_proxy: str | None = None,
    ) -> None:
        """Initialize the Grok, CMC, and Polymarket clients."""
        self._grok = Client(api_key=xai_key.get_secret_value())
        self._model_name = model_name
        self._cmc_key = cmc_key
        self._warp_proxy = warp_proxy
        self._consecutive_failures = 0
        self._cb_lock = threading.Lock()

    def _is_circuit_open(self) -> bool:
        """Check if the Grok circuit breaker is open (too many failures)."""
        with self._cb_lock:
            return self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES

    def _record_success(self) -> None:
        """Record a successful Grok call, resetting the circuit breaker."""
        with self._cb_lock:
            if self._consecutive_failures > 0:
                logger.info("Grok circuit breaker reset after success.")
            self._consecutive_failures = 0

    def _record_failure(self) -> None:
        """Record a failed Grok call, potentially opening the circuit breaker."""
        with self._cb_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    "Grok circuit breaker OPEN: %d consecutive failures. "
                    "Skipping Grok calls until next successful attempt.",
                    self._consecutive_failures,
                )

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

            item = data.get("data", {})
            value = item.get("value")
            label = item.get("value_classification")

            return f"{value} ({label})"
        except Exception:
            logger.exception("CMC Fear & Greed fetch failed")
            return "Error fetching index"

    def analyze_ticker_sentiment(self, ticker: str) -> str:
        """Query Grok for a sentiment analysis of the given ticker.

        Returns a structured response with direction and numeric score.
        Format: "BULLISH (Score: 8/10): reasoning..."
        """
        if self._is_circuit_open():
            logger.info("Grok circuit breaker open. Returning NEUTRAL for %s.", ticker)
            return "NEUTRAL (Score: 5/10): Grok temporarily unavailable."

        try:
            prompt = (
                f"Analyze the current market sentiment for {ticker} (Crypto) based on recent "
                "news and Twitter/X discussions. "
                "Format your response EXACTLY as: "
                "'DIRECTION (Score: X/10): reasoning' "
                "where DIRECTION is BULLISH, BEARISH, or NEUTRAL, "
                "and X is a score from 1 to 10 (1=extremely bearish, 5=neutral, "
                "10=extremely bullish). "
                "Provide a one-sentence reasoning. "
                "IMPORTANT: Do NOT mention general regulatory developments "
                "or news unless there is a specific active ban, delisting, "
                "or enforcement action affecting this asset."
            )

            conversation = self._grok.chat.create(
                model=self._model_name,
                messages=[
                    system("You are a conservative crypto market analyst."),
                    user(prompt),
                ],
            )
            response = conversation.sample()

            if hasattr(response, "message"):
                text = response.message.content.strip()
            elif hasattr(response, "content"):
                text = response.content.strip()
            else:
                msg = f"Unknown response structure from Grok: {type(response)}"
                raise TypeError(msg)

            self._record_success()
            return text
        except Exception:
            logger.exception("Grok Analysis failed")
            self._record_failure()
            return "NEUTRAL (Score: 5/10): Analysis failed."

    def _get_proxied_session(self) -> requests.Session:
        """Return a requests.Session routed through the WARP SOCKS5 proxy."""
        session = requests.Session()
        if self._warp_proxy:
            session.proxies = {
                "http": self._warp_proxy,
                "https": self._warp_proxy,
            }
        return session

    @retry(
        retry=retry_if_exception_type(
            (requests.exceptions.ConnectionError, requests.exceptions.Timeout),
        ),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _fetch_polymarket(self, url: str, params: dict) -> list:
        """Fetch data from Polymarket Gamma API via WARP proxy with retries."""
        session = self._get_proxied_session()
        resp = session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_polymarket_odds(self, ticker: str) -> str:
        """Fetch Polymarket prediction market odds for a crypto ticker."""
        base_asset = ticker.split("/")[0].upper()
        _ASSET_NAMES: dict[str, list[str]] = {
            "BTC": ["BTC", "BITCOIN"],
            "ETH": ["ETH", "ETHEREUM", "ETHER"],
            "SOL": ["SOL", "SOLANA"],
            "XRP": ["XRP", "RIPPLE"],
            "DOGE": ["DOGE", "DOGECOIN"],
            "SHIB": ["SHIB", "SHIBA"],
            "LINK": ["LINK", "CHAINLINK"],
        }
        aliases = _ASSET_NAMES.get(base_asset, [base_asset])
        asset_pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b",
            re.IGNORECASE,
        )

        try:
            markets = self._fetch_polymarket(
                f"{GAMMA_API_BASE}/markets",
                params={
                    "closed": "false",
                    "limit": "200",
                    "order": "volume24hr",
                    "ascending": "false",
                },
            )

            if not isinstance(markets, list):
                return "Polymarket: No data available."

            matches = []
            for market in markets:
                question = market.get("question", "")
                if not asset_pattern.search(question):
                    continue

                raw_prices = market.get("outcomePrices", "")
                if not raw_prices:
                    continue

                try:
                    prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
                    yes_price = float(prices[0])
                    yes_pct = round(yes_price * 100)
                except (json.JSONDecodeError, IndexError, ValueError, TypeError):
                    continue

                if yes_pct <= 0 or yes_pct >= 100:
                    continue

                matches.append(f'"{question}" -> YES: {yes_pct}%')

            if matches:
                return f"Polymarket: {' | '.join(matches[:3])}"

            return "Polymarket: No data available."

        except Exception:
            logger.warning("Polymarket fetch failed for %s.", ticker, exc_info=True)
            return "Polymarket: No data available."

    def fetch_polymarket_markets(self) -> list[dict]:
        """Fetch all top Polymarket markets once per cycle (raw data)."""
        try:
            markets = self._fetch_polymarket(
                f"{GAMMA_API_BASE}/markets",
                params={
                    "closed": "false",
                    "limit": "200",
                    "order": "volume24hr",
                    "ascending": "false",
                },
            )
            return markets if isinstance(markets, list) else []
        except Exception:
            logger.warning("Polymarket bulk fetch failed.", exc_info=True)
            return []

    def get_polymarket_odds_from_cache(self, ticker: str, markets: list[dict]) -> str:
        """Filter pre-fetched Polymarket markets for a specific ticker."""
        base_asset = ticker.split("/")[0].upper()
        asset_names: dict[str, list[str]] = {
            "BTC": ["BTC", "BITCOIN"],
            "ETH": ["ETH", "ETHEREUM", "ETHER"],
            "SOL": ["SOL", "SOLANA"],
            "XRP": ["XRP", "RIPPLE"],
            "DOGE": ["DOGE", "DOGECOIN"],
            "SHIB": ["SHIB", "SHIBA"],
            "LINK": ["LINK", "CHAINLINK"],
        }
        aliases = asset_names.get(base_asset, [base_asset])
        asset_pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b",
            re.IGNORECASE,
        )

        found = []
        for market in markets:
            question = market.get("question", "")
            if not asset_pattern.search(question):
                continue

            raw_prices = market.get("outcomePrices", "")
            if not raw_prices:
                continue

            try:
                prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
                yes_price = float(prices[0])
                yes_pct = round(yes_price * 100)
            except (json.JSONDecodeError, IndexError, ValueError, TypeError):
                continue

            if yes_pct <= 0 or yes_pct >= 100:  # noqa: PLR2004
                continue

            found.append(f'"{question}" -> YES: {yes_pct}%')

        if found:
            return f"Polymarket: {' | '.join(found[:3])}"

        return "Polymarket: No data available."

    def get_consensus_odds(self, ticker: str) -> str:
        """Fetch market consensus from Grok."""
        if self._is_circuit_open():
            logger.info("Grok circuit breaker open. Returning unavailable for %s.", ticker)
            return "Consensus unavailable."

        try:
            prompt = (
                f"Search for recent prediction market odds (like Polymarket) or "
                f"expert consensus on X/Twitter regarding {ticker}. "
                f"Is the consensus that it will go UP or DOWN in the next 24-48h? "
                "Only include data from the last 7 days. Ignore any results older than that. "
                "Provide the consensus direction (UP/DOWN), the approximate probability "
                "(e.g., 65%), and a one-sentence summary. Do NOT include dates or timestamps."
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
                text = response.message.content.strip()
            elif hasattr(response, "content"):
                text = response.content.strip()
            else:
                text = str(response).strip()

            self._record_success()
            return text
        except Exception:
            logger.exception("Grok Consensus fetch failed")
            self._record_failure()
            return "Consensus unavailable."
