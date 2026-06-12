"""Sentiment analysis using LiteLLM-backed chat models and Polymarket."""

import json
import logging
import re
import threading
import time

import requests
from pydantic import SecretStr
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from llm_client import LLMClient

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"


class SentimentOracle:
    """Oracle for fetching market sentiment and predictions.

    LLM calls route through :class:`LLMClient` (LiteLLM-backed); any provider
    LiteLLM supports is available by switching ``model_name``. Legacy bare
    model strings (e.g. ``grok-4-1-fast-reasoning``) are auto-prefixed with
    ``xai/`` to keep the old ``.env`` working without edits.
    """

    def __init__(
        self,
        xai_key: SecretStr | None,
        model_name: str,
        cmc_key: SecretStr | None = None,
        warp_proxy: str | None = None,
        *,
        fallback_model: str | None = None,
        extra_api_keys: dict[str, str] | None = None,
        cache_ttl_seconds: float = 0.0,
    ) -> None:
        """Build the LLM client + HTTP session state.

        ``cache_ttl_seconds`` > 0 enables a per-ticker TTL cache for the
        sentiment and consensus LLM calls. Failure fallbacks are never
        cached, so a degraded provider is retried on the next cycle.
        """
        self._cmc_key = cmc_key
        self._warp_proxy = warp_proxy

        resolved_model = model_name if "/" in model_name else f"xai/{model_name}"

        api_keys: dict[str, str] = {}
        if xai_key is not None:
            api_keys["xai"] = xai_key.get_secret_value()
        if extra_api_keys:
            api_keys.update(extra_api_keys)

        self._llm = LLMClient(
            primary_model=resolved_model,
            fallback_model=fallback_model,
            api_keys=api_keys,
            max_tokens=512,
            timeout=30.0,
        )

        assert cache_ttl_seconds >= 0, f"cache_ttl_seconds must be >= 0: {cache_ttl_seconds}"  # noqa: S101
        self._cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[float, str]] = {}
        self._cache_lock = threading.Lock()

    def _cache_get(self, key: str) -> str | None:
        """Return a cached LLM response if present and fresh."""
        if self._cache_ttl <= 0:
            return None
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is not None and (time.monotonic() - entry[0]) < self._cache_ttl:
                return entry[1]
        return None

    def _cache_put(self, key: str, value: str) -> None:
        """Store a successful LLM response."""
        if self._cache_ttl <= 0:
            return
        with self._cache_lock:
            self._cache[key] = (time.monotonic(), value)

    def _is_circuit_open(self) -> bool:
        """Expose the underlying LLM circuit-breaker state for callers."""
        return self._llm.is_circuit_open()

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
        """Ask the LLM for a sentiment analysis of the given ticker.

        Returns a structured response in the format expected by
        ``strategy._parse_sentiment``: ``"DIRECTION (Score: X/10): reasoning"``.
        """
        cached = self._cache_get(f"sent:{ticker}")
        if cached is not None:
            return cached

        if self._is_circuit_open():
            logger.info(
                "LLM circuit breaker open. Returning NEUTRAL for %s.",
                ticker,
            )
            return "NEUTRAL (Score: 5/10): Sentiment oracle temporarily unavailable."

        system_prompt = "You are a conservative crypto market analyst."
        user_prompt = (
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

        try:
            response = self._llm.chat(system_prompt, user_prompt)
        except Exception:
            logger.exception("Sentiment LLM call failed")
            return "NEUTRAL (Score: 5/10): Analysis failed."
        self._cache_put(f"sent:{ticker}", response)
        return response

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
        aliases = _POLYMARKET_ASSET_NAMES.get(base_asset, [base_asset])
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

            matches = _filter_polymarket_matches(markets, asset_pattern)
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
        aliases = _POLYMARKET_ASSET_NAMES.get(base_asset, [base_asset])
        asset_pattern = re.compile(
            r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b",
            re.IGNORECASE,
        )

        found = _filter_polymarket_matches(markets, asset_pattern)
        if found:
            return f"Polymarket: {' | '.join(found[:3])}"
        return "Polymarket: No data available."

    def get_consensus_odds(self, ticker: str) -> str:
        """Ask the LLM for market consensus on a ticker's near-term direction."""
        cached = self._cache_get(f"cons:{ticker}")
        if cached is not None:
            return cached

        if self._is_circuit_open():
            logger.info(
                "LLM circuit breaker open. Returning unavailable for %s.",
                ticker,
            )
            return "Consensus unavailable."

        system_prompt = "You are a market data aggregator."
        user_prompt = (
            f"Search for recent prediction market odds (like Polymarket) or "
            f"expert consensus on X/Twitter regarding {ticker}. "
            f"Is the consensus that it will go UP or DOWN in the next 24-48h? "
            "Only include data from the last 7 days. Ignore any results older than that. "
            "Provide the consensus direction (UP/DOWN), the approximate probability "
            "(e.g., 65%), and a one-sentence summary. Do NOT include dates or timestamps."
        )

        try:
            response = self._llm.chat(system_prompt, user_prompt)
        except Exception:
            logger.exception("Consensus LLM call failed")
            return "Consensus unavailable."
        self._cache_put(f"cons:{ticker}", response)
        return response


_POLYMARKET_ASSET_NAMES: dict[str, list[str]] = {
    "BTC": ["BTC", "BITCOIN"],
    "ETH": ["ETH", "ETHEREUM", "ETHER"],
    "SOL": ["SOL", "SOLANA"],
    "XRP": ["XRP", "RIPPLE"],
    "DOGE": ["DOGE", "DOGECOIN"],
    "SHIB": ["SHIB", "SHIBA"],
    "LINK": ["LINK", "CHAINLINK"],
}


def _filter_polymarket_matches(markets: list[dict], asset_pattern: re.Pattern) -> list[str]:
    """Produce formatted market strings for Polymarket questions matching a ticker."""
    matches: list[str] = []
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

        matches.append(f'"{question}" -> YES: {yes_pct}%')
    return matches
