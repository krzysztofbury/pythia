"""LLM Challenger — second-opinion commentary on algorithmic trade decisions.

Design:
- The algorithm decides (BUY / SELL / HOLD) autonomously.
- The challenger reads that decision + the market snapshot and produces a
  one-paragraph thesis plus a list of concerns. It NEVER vetoes and is never
  on the execution path; its output is logged and optionally notified when
  concern_level == "HIGH".
- Failures degrade gracefully to Critique(concern_level="NONE", …) so a
  challenger outage cannot block trading.

This separation keeps the bot ToS-safe across LLM providers: the LLM provides
commentary on an already-made decision rather than issuing trading instructions.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from llm_client import LLMClient
from models import MarketData, TradeSignal

logger = logging.getLogger(__name__)

ConcernLevel = Literal["NONE", "LOW", "MEDIUM", "HIGH"]

_VALID_LEVELS: frozenset[str] = frozenset({"NONE", "LOW", "MEDIUM", "HIGH"})


@dataclass(frozen=True)
class Critique:
    """LLM's commentary on a bot decision. Never changes execution."""

    concern_level: ConcernLevel
    thesis: str
    concerns: list[str] = field(default_factory=list)
    model: str = ""

    def format_for_journal(self) -> str:
        """Compact one-line representation for logging to signals.llm_critique."""
        concerns_str = " | ".join(self.concerns[:5]) if self.concerns else "none"
        return f"[{self.concern_level}] {self.thesis[:200]} (concerns: {concerns_str[:200]})"


class LLMChallenger:
    """Reviews algorithmic decisions and produces commentary.

    Use via ``review(signal, data)``. When ``enabled=False`` returns ``None``
    without calling the LLM. When the LLM fails, returns a Critique with
    level="NONE" so callers don't need exception handling.
    """

    def __init__(self, client: LLMClient | None, *, enabled: bool) -> None:
        """Store the LLM client and the enable flag."""
        self._client = client
        self._enabled = enabled and client is not None

    @property
    def enabled(self) -> bool:
        """True when reviews are actually executed."""
        return self._enabled

    def review(self, signal: TradeSignal, data: MarketData) -> Critique | None:
        """Return a Critique for BUY/SELL signals, None otherwise.

        - Disabled or HOLD signals return None (cheap no-op).
        - LLM or parse failures return a Critique(level="NONE") — never raise.
        """
        if not self._enabled:
            return None
        if signal.action not in ("BUY", "SELL"):
            return None

        assert self._client is not None  # noqa: S101

        system_prompt = (
            "You are a senior crypto risk reviewer. The trading bot has ALREADY "
            "made an autonomous decision; your job is to describe what the "
            "decision implies and list any risks the bot may have overlooked. "
            "DO NOT recommend buying or selling. DO NOT instruct the bot to "
            "execute or cancel the trade. Your output is commentary only."
        )
        user_prompt = _build_user_prompt(signal, data)

        try:
            raw = self._client.chat(system_prompt, user_prompt)
        except Exception:
            logger.exception("LLM challenger chat failed")
            return Critique(
                concern_level="NONE",
                thesis="Challenger unavailable.",
                concerns=[],
                model="(error)",
            )

        critique = _parse_critique(raw, model=_client_model(self._client))
        if critique is None:
            logger.warning(
                "LLM challenger returned unparseable response: %s",
                raw[:200],
            )
            return Critique(
                concern_level="NONE",
                thesis="Unparseable challenger response.",
                concerns=[],
                model=_client_model(self._client),
            )
        return critique


def _client_model(client: LLMClient) -> str:
    """Best-effort extraction of the model name from the wrapped client."""
    return getattr(client, "_primary", "unknown")


def _build_user_prompt(signal: TradeSignal, data: MarketData) -> str:
    """Assemble the context + decision payload for the LLM."""
    indicators_summary = "n/a"
    if data.indicators is not None:
        ind = data.indicators
        indicators_summary = (
            f"RSI={getattr(ind, 'rsi', '?')}, "
            f"MACD_hist={getattr(ind, 'macd_histogram', '?')}, "
            f"BB%B={getattr(ind, 'bb_pct_b', '?')}, "
            f"ADX={getattr(ind, 'adx', '?')}, "
            f"ROC={getattr(ind, 'roc', '?')}"
        )
    regime = data.regime.value if data.regime else "UNKNOWN"
    mtf = data.multi_tf.trend_bias if data.multi_tf else "UNKNOWN"

    return (
        f"Ticker: {data.ticker}\n"
        f"Bot decision: {signal.action} (composite={signal.composite:+.3f}, "
        f"confidence={signal.confidence:.3f})\n"
        f"Reasoning from bot: {signal.reasoning[:400]}\n"
        f"Price: {data.current_price}\n"
        f"Indicators: {indicators_summary}\n"
        f"Regime: {regime} | Multi-timeframe: {mtf} | F&G: {data.fear_greed}\n"
        f"Sentiment (Grok): {data.sentiment[:200]}\n"
        f"Consensus (Grok): {data.consensus[:200]}\n"
        f"Polymarket: {data.polymarket_odds[:200]}\n"
        f"\n"
        f"Respond with a JSON object on a single line, no surrounding prose:\n"
        f'{{"concern_level": "NONE|LOW|MEDIUM|HIGH", '
        f'"thesis": "one-paragraph thesis describing what this trade needs '
        f'to succeed", "concerns": ["risk1", "risk2", "..."]}}\n'
        f"Limit concerns to at most 5. Do NOT instruct the bot to act."
    )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_critique(raw: str, *, model: str) -> Critique | None:
    """Parse the LLM response into a Critique; return None on malformed output."""
    if not raw:
        return None

    match = _JSON_OBJECT_RE.search(raw)
    if not match:
        return None

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    level = str(payload.get("concern_level", "NONE")).upper()
    if level not in _VALID_LEVELS:
        return None

    thesis = str(payload.get("thesis", "")).strip()
    if not thesis:
        return None

    raw_concerns = payload.get("concerns", [])
    concerns = [str(c).strip() for c in raw_concerns if isinstance(c, str) and str(c).strip()]

    return Critique(
        concern_level=level,  # type: ignore[arg-type]
        thesis=thesis,
        concerns=concerns[:5],
        model=model,
    )


def build_challenger_from_settings(settings: object) -> LLMChallenger:
    """Factory: build an LLMChallenger from a Settings-like object.

    Duck-typed so tests can pass mocks. Required fields: ``challenger_enabled``,
    ``challenger_model``, ``challenger_fallback_model``, and the relevant
    ``*_api_key`` SecretStr fields for the selected models.
    """
    required = ("challenger_enabled", "challenger_model", "challenger_fallback_model")
    missing = [f for f in required if not hasattr(settings, f)]
    assert not missing, f"build_challenger_from_settings: missing fields {missing}"  # noqa: S101

    if not settings.challenger_enabled:
        return LLMChallenger(client=None, enabled=False)

    api_keys: dict[str, str] = {}
    for provider, attr in (
        ("anthropic", "anthropic_api_key"),
        ("gemini", "gemini_api_key"),
        ("openai", "openai_api_key"),
        ("xai", "xai_api_key"),
    ):
        secret = getattr(settings, attr, None)
        if secret is not None and hasattr(secret, "get_secret_value"):
            api_keys[provider] = secret.get_secret_value()

    client = LLMClient(
        primary_model=settings.challenger_model,
        fallback_model=settings.challenger_fallback_model,
        api_keys=api_keys,
        max_tokens=512,
        timeout=30.0,
    )
    return LLMChallenger(client=client, enabled=True)
