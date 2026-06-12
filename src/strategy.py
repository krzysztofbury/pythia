"""Algorithmic strategy engine — pure rule-based scoring, no LLM dependency.

Replaces the former Gemini-powered strategy to avoid Google API policy
violations around financial advice.  Every buy/sell/hold decision is now
made by a weighted composite score derived from:

  * Technical indicators  (RSI, MACD, Bollinger Bands, ADX, volume)
  * Momentum             (ROC — rate of change)
  * ARIMA price forecast
  * Grok sentiment direction + continuous score
  * Grok consensus direction + probability
  * Polymarket prediction-market odds
  * Fear & Greed Index    (contrarian — fear = bullish)
  * GARCH volatility regime (as a confidence gate)

v8.0 — Rebalanced weights, added Fear & Greed contrarian signal,
mean-reversion bonus, active-weight normalization for missing data.

v8.1 — Post-deployment tuning:
  - Entry threshold 0.10 → 0.15 (too much noise at 0.10)
  - Fear & Greed weight 0.10 → 0.05 (permanently bullish in fear markets)
  - Stale position timeout 72h → 120h (stale exits were #1 loss source)
  - Gate entry on negative technicals (prevents buying overbought)
  - Gate entry on MTF bearish conflict (stops buying into downtrends)

v8.2 — Post-analysis tuning (April 2026):
  - Hard-block all entries in TRENDING_DOWN regime (no buying into downtrends)
  - Entry threshold 0.15 → 0.25 (kill low-conviction entries that become stale exits)
  - Mean reversion bonus gated to RANGING regime only (was firing in early downtrends)
  - TRENDING_DOWN regime modifier now dampens positive composites (x0.80)
  - Fear & Greed weight 0.05 → 0.05 (unchanged, but technical weight absorbs more)
"""

import logging
import math
import re

from models import MarketData, TradeSignal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Composite-score weights — sum to 1.0.
#
# Changes from v7:
#   Technical  0.25 → 0.30  (oversold RSI/BB deserve more influence)
#   Momentum   0.15 → 0.10  (ROC over-punished in sideways markets)
#   Forecast   0.10 → 0.05  (ARIMA weak for 1h crypto, often FLAT)
#   Sentiment  0.25 → 0.20  (Grok can hallucinate, reduce reliance)
#   Consensus  0.10 → 0.10  (keep)
#   Polymarket 0.15 → 0.15  (keep — real market data)
#   Fear&Greed  NEW → 0.10  (contrarian: extreme fear = buying opportunity)
#
# v8.1 changes:
#   Fear&Greed 0.10 → 0.05  (too bullish in fear markets without counterweight)
#   Technical  0.30 → 0.35  (absorb F&G reduction — technicals should gate)
# ---------------------------------------------------------------------------

# NOTE: the numeric defaults below are mirrored in ``Settings`` in config.py.
# A drift check at import-time (bottom of this file) asserts they stay in sync,
# so tuning via .env cannot silently diverge from the module fallbacks used by
# ``Strategist()`` without a ``settings`` argument.

W_TECHNICAL = 0.34
W_MOMENTUM = 0.10
W_FORECAST = 0.05
W_SENTIMENT = 0.19
W_CONSENSUS = 0.09
W_POLYMARKET = 0.15
W_FEAR_GREED = 0.05
# v9.7: perpetual funding as a contrarian component (negative rate = shorts
# paying longs = bullish). Active only when FUNDING_RATES_ENABLED supplies
# data; otherwise the weight redistributes like missing Polymarket.
W_FUNDING = 0.03

_WEIGHT_SUM = (
    W_TECHNICAL
    + W_MOMENTUM
    + W_FORECAST
    + W_SENTIMENT
    + W_CONSENSUS
    + W_POLYMARKET
    + W_FEAR_GREED
    + W_FUNDING
)
assert abs(_WEIGHT_SUM - 1.0) < 1e-9, f"Weights must sum to 1.0, got {_WEIGHT_SUM}"  # noqa: S101, PLR2004

# Entry / exit thresholds (asymmetric — drops are faster than rallies)
_ENTRY_THRESHOLD = 0.25  # Was 0.15 — still too many low-conviction entries becoming stale exits
_EXIT_THRESHOLD = -0.05
_STRONG_SELL_THRESHOLD = -0.25
_DCA_THRESHOLD = 0.30  # Was 0.25 — DCA should require stronger signals than initial entry
_PROFIT_TAKE_PCT = 5.0
_MOMENTUM_WEAK_THRESHOLD = 0.1
_SMA_OVEREXTENDED_PCT = 15.0  # Block entry when price is >15% above SMA-7

# Confidence gates per volatility regime (crypto-adjusted)
_VOL_CONFIDENCE_GATES = {
    "NORMAL": 0.0,  # No gate
    "HIGH": 0.45,  # Require moderate confidence
    "EXTREME": 0.65,  # Require strong confidence
}

# 15% boost when forecast and sentiment align
_AGREEMENT_BOOST = 1.15
# 25% penalty when they conflict
_CONFLICT_PENALTY = 0.75

# Mean reversion bonus: oversold + weak trend = high-probability dip buy
_MEAN_REVERSION_RSI = 28  # RSI below this = deeply oversold
_MEAN_REVERSION_ADX = 30  # ADX below this = no strong trend (range-bound)
_MEAN_REVERSION_BONUS = 0.10  # Added to composite when conditions met

# Position aging: signal review when position exceeds max age
_STALE_POSITION_HOURS = 120  # Was 72 — stale exits were #1 loss source in v8.0
# Positions with |P&L| above this are actively moving — don't force-exit.
_STALE_EXIT_PNL_THRESHOLD_PCT = 2.0


# ---------------------------------------------------------------------------
# Input parsers
# ---------------------------------------------------------------------------


def _parse_sentiment(sentiment: str) -> tuple[str, float]:
    """Parse Grok sentiment -> (direction, score in [-1.0, +1.0]).

    Attempts to extract a numeric score first (e.g., "Score: 8/10"),
    then falls back to keyword matching.
    """
    text = sentiment.strip()

    # Try to extract numeric score: "Score: 8/10", "(8/10)", or bare "8/10"
    score_match = re.search(r"(?:Score:\s*)?(\d{1,2})/10", text, re.IGNORECASE)
    if score_match:
        raw_score = int(score_match.group(1))
        # Map 1-10 to -1.0 to +1.0
        normalized = (raw_score - 5.5) / 4.5
        normalized = max(-1.0, min(1.0, normalized))

        # Determine direction from score
        if raw_score >= 7:  # noqa: PLR2004
            return "BULLISH", normalized
        if raw_score <= 3:  # noqa: PLR2004
            return "BEARISH", normalized
        # v9.6: the neutral band scores exactly 0. The old (raw-5.5)/4.5
        # mapping made "NEUTRAL (Score: 5/10)" — also the LLM-failure
        # fallback — a permanent -0.111 drag on the composite.
        return "NEUTRAL", 0.0

    # Fallback: keyword matching (capped at +/-0.6 since it's less precise)
    upper = text.upper()
    if upper.startswith("BULLISH"):
        return "BULLISH", 0.6
    if upper.startswith("BEARISH"):
        return "BEARISH", -0.6
    return "NEUTRAL", 0.0


# Marker substrings emitted by sentiment.py / data_pipeline.py fallback paths.
# Matching responses carry no market information and must be excluded from the
# composite's effective weight (same treatment as missing Polymarket data).
_UNAVAILABLE_MARKERS = ("unavailable", "failed", "timeout", "insufficient")


def _sentiment_available(text: str) -> bool:
    """True when an LLM sentiment/consensus response carries real signal.

    Failure fallbacks ("Analysis failed.", "Sentiment oracle temporarily
    unavailable.", "Consensus unavailable.", "Timeout.") and empty strings
    are dead weight, not neutral opinions.
    """
    stripped = text.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    return not any(marker in lowered for marker in _UNAVAILABLE_MARKERS)


def _parse_consensus(consensus: str) -> tuple[str, float, float]:
    """Parse LLM consensus -> (direction, probability, score in [-1, +1])."""
    text = consensus.strip().upper()

    prob_match = re.search(r"(\d{1,3})%", text)
    probability = float(prob_match.group(1)) if prob_match else 50.0

    # v9.7: word boundaries — bare substring matching turned "UPDATE" into UP
    # and "DOWNTREND" into DOWN, letting prose false-positive the direction.
    if re.search(r"\bUP\b", text[:30]):
        return "UP", probability, (probability - 50) / 50
    if re.search(r"\bDOWN\b", text[:30]):
        return "DOWN", probability, -(probability - 50) / 50
    return "NEUTRAL", 50.0, 0.0


# Funding-rate normalization cap: +/-0.1% per 8h funding period is an extreme
# reading on major perps; anything beyond saturates the score.
_FUNDING_EXTREME_RATE = 0.001


def _parse_funding(funding_rate: float | None) -> tuple[float, bool]:
    """Normalize a perp funding rate -> (score in [-1, +1], has_data).

    Contrarian sign convention: negative funding (shorts paying longs,
    crowded short positioning) scores bullish; positive funding bearish.
    """
    if funding_rate is None or not math.isfinite(funding_rate):
        # NaN poisons min/max into returning +1.0 — a max-conviction signal
        # fabricated from garbage. Non-finite means no data.
        return 0.0, False
    score = -funding_rate / _FUNDING_EXTREME_RATE
    return max(-1.0, min(1.0, score)), True


def _parse_polymarket(polymarket_odds: str) -> tuple[float, bool]:
    """Parse Polymarket odds -> (score in [-1, +1], has_data).

    Returns has_data=False when no data is available, so the weight
    can be redistributed to components that have actual signal.
    """
    if not polymarket_odds or "No data" in polymarket_odds:
        return 0.0, False

    matches = re.findall(r"YES:\s*(\d{1,3})%", polymarket_odds)
    if not matches:
        return 0.0, False

    avg_yes = sum(float(m) for m in matches) / len(matches)
    return (avg_yes - 50) / 50, True


def _parse_fear_greed(fear_greed: str) -> tuple[float, bool]:
    """Parse Fear & Greed Index -> contrarian score and data availability.

    Contrarian logic (Financial Advisor perspective):
      Extreme Fear (0-20)  →  +0.8 to +1.0 (strong buy signal)
      Fear (21-40)         →  +0.2 to +0.6
      Neutral (41-60)      →  -0.2 to +0.2
      Greed (61-80)        →  -0.6 to -0.2
      Extreme Greed (81-100) → -1.0 to -0.6 (strong sell signal)
    """
    if not fear_greed:
        return 0.0, False

    match = re.search(r"^(\d{1,3})", fear_greed)
    if not match:
        return 0.0, False

    value = int(match.group(1))
    # Fear & Greed Index is 0-100; reject out-of-range values.
    if value > 100:  # noqa: PLR2004
        logger.warning("Fear & Greed value out of range: %d, ignoring", value)
        return 0.0, False

    # 0 → +1.0, 50 → 0.0, 100 → -1.0
    return (50 - value) / 50, True


# ---------------------------------------------------------------------------
# Individual indicator scorers
# ---------------------------------------------------------------------------


def _score_rsi(rsi: float | None) -> float:
    """RSI -> score in [-0.8, +0.8]. Oversold = bullish, overbought = bearish."""
    if rsi is None:
        return 0.0
    if rsi <= 30:  # noqa: PLR2004
        return 0.8
    if rsi <= 40:  # noqa: PLR2004
        return 0.3
    if rsi >= 70:  # noqa: PLR2004
        return -0.8
    if rsi >= 60:  # noqa: PLR2004
        return -0.3
    return 0.0


def _score_macd(histogram: float | None) -> float:
    """MACD histogram -> score in [-0.5, +0.5]. Positive = bullish momentum."""
    if histogram is None:
        return 0.0
    return 0.5 if histogram > 0 else (-0.5 if histogram < 0 else 0.0)


def _score_bollinger(percent_b: float | None) -> float:
    """Bollinger %B -> score in [-0.6, +0.6]. Below 0 = oversold, above 1 = overbought."""
    if percent_b is None:
        return 0.0
    if percent_b < 0:
        return 0.6
    if percent_b < 0.2:  # noqa: PLR2004
        return 0.3
    if percent_b > 1.0:
        return -0.6
    if percent_b > 0.8:  # noqa: PLR2004
        return -0.3
    return 0.0


def _score_volume(obv_slope: float | None) -> float:
    """OBV slope -> score in [-0.4, +0.4]. Accumulation = bullish confirmation."""
    if obv_slope is None:
        return 0.0
    return max(-0.4, min(0.4, obv_slope * 0.6))


# ---------------------------------------------------------------------------
# Composite sub-score helpers
# ---------------------------------------------------------------------------


def _compute_technical_score(
    data: MarketData,
) -> tuple[float, float | None, float | None, float | None, list[str]]:
    """Score technical indicators from the IndicatorSnapshot.

    Returns (technical_score, rsi, atr_pct, sma_7_deviation_pct, reasons).
    """
    rsi = None
    macd_histogram = None
    bollinger_pct_b = None
    atr_pct = None
    sma_7_deviation_pct = None
    obv_slope = None

    if data.indicators:
        rsi = data.indicators.rsi
        macd_histogram = data.indicators.macd_histogram
        bollinger_pct_b = data.indicators.bb_pct_b
        atr_pct = data.indicators.atr_pct
        sma_7_deviation_pct = data.indicators.sma_7_deviation_pct
        obv_slope = data.indicators.obv_slope

    rsi_score = _score_rsi(rsi)
    macd_score = _score_macd(macd_histogram)
    bollinger_score = _score_bollinger(bollinger_pct_b)
    volume_score = _score_volume(obv_slope)
    # Average 4 sub-scores instead of 3
    technical_score = (rsi_score + macd_score + bollinger_score + volume_score) / 4.0

    reasons: list[str] = []
    if rsi is not None:
        reasons.append(f"RSI={rsi:.0f}")
    if macd_histogram is not None:
        sign = "+" if macd_histogram > 0 else ""
        reasons.append(f"MACD={sign}{macd_histogram:.4f}")
    if bollinger_pct_b is not None:
        reasons.append(f"BB%B={bollinger_pct_b:.2f}")

    return technical_score, rsi, atr_pct, sma_7_deviation_pct, reasons


def _compute_momentum_score(data: MarketData) -> tuple[float, float | None, list[str]]:
    """Score momentum from ROC and ADX trend strength.

    Returns (momentum_score, adx, reasons).
    """
    roc = None
    adx = None

    if data.indicators:
        roc = data.indicators.roc
        adx = data.indicators.adx

    # ROC score: normalize to [-1, +1], capped
    roc_score = 0.0
    if roc is not None:
        # ROC of 5% maps to score ~1.0
        roc_score = max(-1.0, min(1.0, roc / 5.0))

    # ADX as a trend strength multiplier
    # ADX < 20 = no trend (reduce signal), ADX > 40 = strong trend (amplify)
    adx_multiplier = 0.5  # default: weak trend
    if adx is not None:
        if adx >= 40:  # noqa: PLR2004
            adx_multiplier = 1.2
        elif adx >= 25:  # noqa: PLR2004
            adx_multiplier = 1.0
        elif adx >= 15:  # noqa: PLR2004
            adx_multiplier = 0.7

    momentum_score = roc_score * adx_multiplier
    momentum_score = max(-1.0, min(1.0, momentum_score))

    reasons: list[str] = []
    if roc is not None:
        reasons.append(f"ROC={roc:+.2f}%")
    if adx is not None:
        label = "Strong" if adx >= 25 else "Weak"  # noqa: PLR2004
        reasons.append(f"ADX={adx:.0f}({label})")

    return momentum_score, adx, reasons


def _compute_forecast_score(
    data: MarketData,
    atr_pct: float | None,
) -> tuple[float, str, float, bool, str]:
    """Score the ARIMA forecast, normalized against ATR.

    Returns (forecast_score, forecast_direction, forecast_change_pct,
             has_forecast_data, reason).
    """
    forecast_score = 0.0
    forecast_direction = "FLAT"
    forecast_change_pct = 0.0
    has_data = False

    if data.forecast:
        forecast_direction = data.forecast.direction
        forecast_change_pct = data.forecast.change_pct

        if forecast_direction != "FLAT":
            has_data = True
            if atr_pct is not None and atr_pct > 0:
                forecast_score = min(abs(forecast_change_pct) / atr_pct, 1.0)
            else:
                forecast_score = min(abs(forecast_change_pct), 1.0)

            if forecast_direction == "DOWN":
                forecast_score = -forecast_score

    reason = f"ARIMA={forecast_direction}({forecast_change_pct:+.2f}%)"
    return forecast_score, forecast_direction, forecast_change_pct, has_data, reason


def _apply_agreement_modifier(
    composite: float,
    forecast_direction: str,
    sentiment_direction: str,
) -> tuple[float, str | None]:
    """Adjust composite when forecast and sentiment agree or conflict.

    Returns (adjusted_composite, reason_string_or_None).
    """
    assert forecast_direction in ("UP", "DOWN", "FLAT"), (  # noqa: S101
        f"Unexpected forecast: {forecast_direction}"
    )
    assert sentiment_direction in ("BULLISH", "BEARISH", "NEUTRAL"), (  # noqa: S101
        f"Unexpected sentiment: {sentiment_direction}"
    )

    forecast_bullish = forecast_direction == "UP"
    forecast_bearish = forecast_direction == "DOWN"
    sentiment_bullish = sentiment_direction == "BULLISH"
    sentiment_bearish = sentiment_direction == "BEARISH"

    if (forecast_bullish and sentiment_bullish) or (forecast_bearish and sentiment_bearish):
        return composite * _AGREEMENT_BOOST, "forecast+sentiment agree"
    if (forecast_bullish and sentiment_bearish) or (forecast_bearish and sentiment_bullish):
        return composite * _CONFLICT_PENALTY, "forecast/sentiment conflict"
    return composite, None


def _compute_volatility_penalty(volatility_regime: str) -> tuple[float, str | None]:
    """Compute confidence penalty for volatile regimes.

    Returns (confidence_gate, reason_string_or_None).
    """
    gate = _VOL_CONFIDENCE_GATES.get(volatility_regime, 0.0)
    if gate > 0:
        return gate, f"Vol={volatility_regime}"
    return 0.0, None


def _apply_regime_modifier(
    composite: float,
    regime: "MarketRegime | None",
) -> tuple[float, str | None]:
    """Adjust composite based on market regime.

    Only applies when composite sign matches regime expectation:
    - TRENDING_UP boosts positive composites (buy signals in uptrends)
    - TRENDING_DOWN amplifies negative composites (sell signals in downtrends)
    - RANGING boosts positive composites (mean-reversion buys)
    - BREAKOUT penalizes positive composites (wait for confirmation)
    """
    if regime is None:
        return composite, None

    from models import MarketRegime

    # Bullish regimes boost positive composites only
    if regime == MarketRegime.TRENDING_UP and composite > 0:
        return composite * 1.10, "regime=TRENDING_UP(x1.10)"
    # Bearish regime: dampen positive composites, amplify negative ones
    if regime == MarketRegime.TRENDING_DOWN:
        if composite > 0:
            return composite * 0.80, "regime=TRENDING_DOWN(x0.80)"
        if composite < 0:
            return composite * 1.20, "regime=TRENDING_DOWN(x1.20)"
        return composite, "regime=TRENDING_DOWN(no-op)"
    # Range regime boosts positive composites (mean reversion)
    if regime == MarketRegime.RANGING and composite > 0:
        return composite * 1.05, "regime=RANGING(x1.05)"
    # Breakout penalizes positive composites (wait for confirmation)
    if regime == MarketRegime.BREAKOUT and composite > 0:
        return composite * 0.90, "regime=BREAKOUT(x0.90)"

    return composite, f"regime={regime.value}(no-op)"


_MTF_ALIGNMENT_BOOST = 1.15  # 15% boost when 4h trend aligns with 1h signal
_MTF_CONFLICT_PENALTY = 0.75  # 25% penalty when they conflict


def _apply_mtf_alignment(
    composite: float,
    multi_tf: "MultiTimeframeContext | None",
) -> tuple[float, str | None]:
    """Adjust composite when multi-timeframe trend aligns or conflicts with signal direction.

    If 4h trend is BULLISH and composite is positive → boost
    If 4h trend is BEARISH and composite is positive → penalty
    Vice versa for negative composites.
    """
    if multi_tf is None or multi_tf.trend_bias == "NEUTRAL":
        return composite, None

    signal_bullish = composite > 0
    trend_bullish = multi_tf.trend_bias == "BULLISH"

    if signal_bullish == trend_bullish:
        return composite * _MTF_ALIGNMENT_BOOST, f"mtf={multi_tf.trend_bias}(aligned)"
    return composite * _MTF_CONFLICT_PENALTY, f"mtf={multi_tf.trend_bias}(conflict)"


# ---------------------------------------------------------------------------
# Strategist
# ---------------------------------------------------------------------------


class Strategist:
    """Algorithmic Strategist — pure rule-based scoring engine.

    Accepts a ``MarketData`` dataclass so that ``main.py`` builds all
    context before calling into the strategy.

    Thresholds load from ``Settings`` at construction; pass ``settings=None``
    (the default) to use the historical module-level constants.
    """

    _REQUIRED_SETTINGS_FIELDS = (
        "entry_threshold",
        "exit_threshold",
        "strong_sell_threshold",
        "dca_threshold",
        "profit_take_pct",
        "momentum_weak_threshold",
        "sma_overextended_pct",
        "mean_reversion_rsi",
        "mean_reversion_adx",
        "mean_reversion_bonus",
        "stale_position_hours",
        "stale_exit_pnl_threshold_pct",
    )

    DEFAULT_WEIGHTS: "dict[str, float]" = {
        "technical": W_TECHNICAL,
        "momentum": W_MOMENTUM,
        "forecast": W_FORECAST,
        "sentiment": W_SENTIMENT,
        "consensus": W_CONSENSUS,
        "polymarket": W_POLYMARKET,
        "fear_greed": W_FEAR_GREED,
        "funding": W_FUNDING,
    }

    def __init__(
        self,
        settings: object | None = None,
        *,
        weights_override: "dict[str, float] | None" = None,
    ) -> None:
        """Bind strategy thresholds from Settings (or module defaults).

        ``settings`` is duck-typed: any object exposing the required fields
        works, which keeps this module free of a top-level config import
        (avoids a circular-import risk with tests and the backtest harness).
        Required fields are asserted up front so a misconfigured caller
        fails loudly at construction rather than deep inside ``generate_signal``.

        ``weights_override`` lets the optimizer run trials with custom
        composite weights. When provided, must contain the seven keys in
        :attr:`DEFAULT_WEIGHTS` and sum to 1.0 (±1e-9).
        """
        if settings is None:
            self.entry_threshold: float = _ENTRY_THRESHOLD
            self.exit_threshold: float = _EXIT_THRESHOLD
            self.strong_sell_threshold: float = _STRONG_SELL_THRESHOLD
            self.dca_threshold: float = _DCA_THRESHOLD
            self.profit_take_pct: float = _PROFIT_TAKE_PCT
            self.momentum_weak_threshold: float = _MOMENTUM_WEAK_THRESHOLD
            self.sma_overextended_pct: float = _SMA_OVEREXTENDED_PCT
            self.mean_reversion_rsi: float = _MEAN_REVERSION_RSI
            self.mean_reversion_adx: float = _MEAN_REVERSION_ADX
            self.mean_reversion_bonus: float = _MEAN_REVERSION_BONUS
            self.stale_position_hours: float = _STALE_POSITION_HOURS
            self.stale_exit_pnl_threshold_pct: float = _STALE_EXIT_PNL_THRESHOLD_PCT
            self.require_bullish_sentiment: bool = False
        else:
            missing = [f for f in self._REQUIRED_SETTINGS_FIELDS if not hasattr(settings, f)]
            assert not missing, f"Strategist settings missing fields: {missing}"  # noqa: S101

            self.entry_threshold = settings.entry_threshold
            self.exit_threshold = settings.exit_threshold
            self.strong_sell_threshold = settings.strong_sell_threshold
            self.dca_threshold = settings.dca_threshold
            self.profit_take_pct = settings.profit_take_pct
            self.momentum_weak_threshold = settings.momentum_weak_threshold
            self.sma_overextended_pct = settings.sma_overextended_pct
            self.mean_reversion_rsi = settings.mean_reversion_rsi
            self.mean_reversion_adx = settings.mean_reversion_adx
            self.mean_reversion_bonus = settings.mean_reversion_bonus
            self.stale_position_hours = settings.stale_position_hours
            self.stale_exit_pnl_threshold_pct = settings.stale_exit_pnl_threshold_pct
            # getattr: duck-typed settings from the optimizer/backtest harness
            # predate v9.6 and may not carry the gate-mode flag.
            self.require_bullish_sentiment = bool(
                getattr(settings, "require_bullish_sentiment", False),
            )

        if weights_override is None:
            self._weights = dict(self.DEFAULT_WEIGHTS)
        else:
            missing_w = set(self.DEFAULT_WEIGHTS) - set(weights_override)
            assert not missing_w, f"weights_override missing keys: {sorted(missing_w)}"  # noqa: S101
            total = sum(weights_override.values())
            assert abs(total - 1.0) < 1e-9, (  # noqa: S101
                f"weights_override must sum to 1.0, got {total}"
            )
            self._weights = dict(weights_override)

    def generate_signal(self, data: MarketData) -> TradeSignal:
        """Score all data algorithmically and issue a BUY / SELL / HOLD signal."""
        assert data.current_price > 0, f"Price must be positive, got {data.current_price}"  # noqa: S101

        reasons: list[str] = []

        # Parse string-only inputs
        sentiment_direction, sentiment_score = _parse_sentiment(data.sentiment)
        consensus_direction, consensus_probability, consensus_score = _parse_consensus(
            data.consensus,
        )
        polymarket_score, has_polymarket = _parse_polymarket(data.polymarket_odds)
        fear_greed_score, has_fear_greed = _parse_fear_greed(data.fear_greed)
        funding_score, has_funding = _parse_funding(data.funding_rate)

        # Technical sub-score
        technical_score, rsi, atr_pct, sma_7_deviation_pct, tech_reasons = _compute_technical_score(
            data,
        )
        reasons.extend(tech_reasons)

        # Momentum sub-score (ROC + ADX)
        momentum_score, adx, momentum_reasons = _compute_momentum_score(data)
        reasons.extend(momentum_reasons)

        # Forecast sub-score
        forecast_score, forecast_direction, _fc_change_pct, has_forecast, forecast_reason = (
            _compute_forecast_score(data, atr_pct)
        )
        reasons.append(forecast_reason)

        # Volatility regime
        volatility_regime = "NORMAL"
        if data.volatility:
            volatility_regime = data.volatility.regime.value

        # ---------------------------------------------------------------
        # Weighted composite with active-weight normalization.
        #
        # When a component has no data (Polymarket unavailable, ARIMA FLAT,
        # Fear & Greed missing), its weight is excluded from the denominator
        # so the remaining signals aren't diluted by dead weight.
        # ---------------------------------------------------------------
        w = self._weights
        raw_sum = w["technical"] * technical_score + w["momentum"] * momentum_score
        effective_weight = w["technical"] + w["momentum"]

        # v9.6: LLM failure fallbacks are dead weight, not neutral opinions —
        # exclude them from the denominator like missing Polymarket data.
        if _sentiment_available(data.sentiment):
            raw_sum += w["sentiment"] * sentiment_score
            effective_weight += w["sentiment"]

        if _sentiment_available(data.consensus):
            raw_sum += w["consensus"] * consensus_score
            effective_weight += w["consensus"]

        if has_forecast:
            raw_sum += w["forecast"] * forecast_score
            effective_weight += w["forecast"]

        if has_polymarket:
            raw_sum += w["polymarket"] * polymarket_score
            effective_weight += w["polymarket"]

        if has_fear_greed:
            raw_sum += w["fear_greed"] * fear_greed_score
            effective_weight += w["fear_greed"]

        if has_funding:
            raw_sum += w["funding"] * funding_score
            effective_weight += w["funding"]

        assert effective_weight > 0, f"Effective weight must be positive, got {effective_weight}"  # noqa: S101
        composite = raw_sum / effective_weight

        reasons.append(f"Sent={sentiment_direction}")
        if consensus_direction != "NEUTRAL":
            reasons.append(f"Cons={consensus_direction}({consensus_probability:.0f}%)")

        # Agreement/conflict modifier (applied before additive bonuses
        # so the multiplicative adjustment only scales the weighted composite,
        # not the fixed mean-reversion bonus).
        composite, agreement_reason = _apply_agreement_modifier(
            composite,
            forecast_direction,
            sentiment_direction,
        )
        if agreement_reason:
            reasons.append(agreement_reason)

        # ---------------------------------------------------------------
        # Mean reversion bonus (Day Trader's edge).
        #
        # Deeply oversold RSI + weak trend (ADX < 30) = range-bound market,
        # not a crash.  This is a high-probability dip-buy setup.
        # Applied AFTER agreement modifier so the flat +0.10 is independent
        # of forecast/sentiment alignment.
        # Disabled during EXTREME volatility for safety.
        # ---------------------------------------------------------------
        from models import MarketRegime as _MR

        if (
            data.regime == _MR.RANGING
            and rsi is not None
            and rsi < self.mean_reversion_rsi
            and adx is not None
            and adx < self.mean_reversion_adx
            and volatility_regime != "EXTREME"
        ):
            composite += self.mean_reversion_bonus
            reasons.append("mean-reversion(oversold+weak-trend+RANGING)")

        # Regime modifier
        composite, regime_reason = _apply_regime_modifier(composite, data.regime)
        if regime_reason:
            reasons.append(regime_reason)

        # Multi-timeframe alignment
        composite, mtf_reason = _apply_mtf_alignment(composite, data.multi_tf)
        if mtf_reason:
            reasons.append(mtf_reason)

        # Volatility confidence gate
        vol_gate, vol_reason = _compute_volatility_penalty(volatility_regime)
        if vol_reason:
            reasons.append(vol_reason)

        # Machine-readable component-score token — consumed by
        # ``signal_analytics.factor_attribution`` via regex so the per-factor
        # correlation with trade P&L can be computed post-hoc. Keep the exact
        # ``key=+0.000`` format; the parser's regex is anchored to it.
        component_token = (
            f"tech={technical_score:+.3f} mom={momentum_score:+.3f} "
            f"fc={forecast_score:+.3f} sent={sentiment_score:+.3f} "
            f"cons={consensus_score:+.3f} poly={polymarket_score:+.3f} "
            f"fg={fear_greed_score:+.3f}"
        )
        if has_funding:
            component_token += f" fund={funding_score:+.3f}"
        reasons.append(component_token)

        raw_confidence = min(abs(composite), 1.0)

        logger.info(
            "Algo [%s]: comp=%+.3f tech=%+.3f mom=%+.3f fc=%+.3f "
            "sent=%+.3f cons=%+.3f poly=%+.3f fg=%+.3f "
            "eff_w=%.2f vol=%s regime=%s",
            data.ticker,
            composite,
            technical_score,
            momentum_score,
            forecast_score,
            sentiment_score,
            consensus_score,
            polymarket_score,
            fear_greed_score,
            effective_weight,
            volatility_regime,
            data.regime.value if data.regime else "N/A",
        )

        # Compute regime-adjusted entry threshold
        effective_entry_threshold = self.entry_threshold
        if data.regime is not None:
            from regime import RegimeClassifier

            _rc = RegimeClassifier()
            effective_entry_threshold += _rc.get_entry_threshold_adjustment(data.regime)

        # v8.2: Hard-block all entries in TRENDING_DOWN
        from models import MarketRegime as _MR

        if data.regime == _MR.TRENDING_DOWN and not data.has_position:
            return TradeSignal(
                action="HOLD",
                confidence=raw_confidence,
                composite=composite,
                reasoning=f"Blocked: TRENDING_DOWN regime — no new entries. {', '.join(reasons)}",
            )

        if data.has_position:
            return self._exit_signal(
                composite,
                raw_confidence,
                vol_gate,
                volatility_regime,
                data,
                sentiment_direction,
                reasons,
                data.current_price,
            )
        return self._entry_signal(
            composite,
            raw_confidence,
            vol_gate,
            volatility_regime,
            sentiment_direction,
            consensus_probability,
            consensus_direction,
            rsi,
            sma_7_deviation_pct,
            reasons,
            data.current_price,
            effective_entry_threshold,
            technical_score=technical_score,
            multi_tf=data.multi_tf,
        )

    # ------------------------------------------------------------------
    # Private decision helpers
    # ------------------------------------------------------------------

    def _entry_signal(  # noqa: PLR0913
        self,
        composite: float,
        raw_confidence: float,
        vol_gate: float,
        volatility_regime: str,
        sentiment_direction: str,
        consensus_probability: float,
        consensus_direction: str,
        rsi: float | None,
        sma_7_deviation_pct: float | None,
        reasons: list[str],
        current_price: float,
        entry_threshold: float | None = None,
        *,
        technical_score: float = 0.0,
        multi_tf: "MultiTimeframeContext | None" = None,
    ) -> TradeSignal:
        """Entry logic: BUY or HOLD."""
        if entry_threshold is None:
            entry_threshold = self.entry_threshold
        assert -1.0 <= technical_score <= 1.0, (  # noqa: S101
            f"technical_score out of range: {technical_score}"
        )
        if multi_tf is not None:
            assert multi_tf.trend_bias in ("BULLISH", "BEARISH", "NEUTRAL"), (  # noqa: S101
                f"Unexpected trend_bias: {multi_tf.trend_bias}"
            )

        if self.require_bullish_sentiment:
            # Legacy hard gate: the LLM must actively confirm the entry.
            # Disables ALL entries whenever the sentiment provider is degraded
            # (the 2026-05/06 seven-week silence). Opt-in via Settings.
            sentiment_gate = (
                (sentiment_direction == "BULLISH")
                or (consensus_direction == "UP" and consensus_probability > 60)  # noqa: PLR2004
            )
        else:
            # v9.6 advisory mode: sentiment can veto, never authorize —
            # block only on BEARISH sentiment or a strong DOWN consensus.
            sentiment_gate = sentiment_direction != "BEARISH" and not (
                consensus_direction == "DOWN" and consensus_probability > 60  # noqa: PLR2004
            )

        # Block entry when RSI signals overbought.
        rsi_gate = rsi is None or rsi <= 70  # noqa: PLR2004

        # Block entry when price is too far above SMA (overextended rally).
        sma_concern = (
            sma_7_deviation_pct is not None and sma_7_deviation_pct > self.sma_overextended_pct
        )

        # v8.1: Block entry when technicals are negative (sentiment overriding bearish price action).
        technical_gate = technical_score >= 0.0

        # v8.1: Block entry when multi-timeframe trend conflicts (4h bearish vs 1h bullish).
        mtf_gate = multi_tf is None or multi_tf.trend_bias != "BEARISH"

        all_gates = (
            composite > entry_threshold
            and sentiment_gate
            and rsi_gate
            and not sma_concern
            and technical_gate
            and mtf_gate
        )

        if all_gates:
            confidence = raw_confidence

            # Apply volatility confidence gate
            if vol_gate > 0 and confidence <= vol_gate:
                return TradeSignal(
                    action="HOLD",
                    confidence=confidence,
                    composite=composite,
                    reasoning=(
                        f"Bullish but {volatility_regime} volatility requires "
                        f"confidence >{vol_gate:.2f} (got {confidence:.2f}). "
                        f"{', '.join(reasons)}"
                    ),
                )

            return TradeSignal(
                action="BUY",
                confidence=confidence,
                composite=composite,
                reasoning=(f"Algorithmic BUY: composite={composite:+.3f}. {', '.join(reasons)}"),
                target_price=current_price * 1.04,
            )

        # Explain why HOLD
        blockers: list[str] = []
        if not sentiment_gate:
            blockers.append("sentiment/consensus gate not met")
        if not rsi_gate:
            blockers.append("RSI overbought (>70)")
        if sma_concern:
            blockers.append(
                f"SMA deviation +{sma_7_deviation_pct:.1f}% > {self.sma_overextended_pct}%",
            )
        if composite <= entry_threshold:
            blockers.append(f"composite too low ({composite:+.3f})")
        if not technical_gate:
            blockers.append(f"technicals negative ({technical_score:+.3f})")
        if not mtf_gate:
            blockers.append("MTF bearish conflict")

        return TradeSignal(
            action="HOLD",
            confidence=raw_confidence,
            composite=composite,
            reasoning=f"HOLD: {', '.join(blockers)}. {', '.join(reasons)}",
        )

    def _exit_signal(  # noqa: PLR0913, PLR0911
        self,
        composite: float,
        raw_confidence: float,
        vol_gate: float,
        volatility_regime: str,
        data: MarketData,
        sentiment_direction: str,
        reasons: list[str],
        current_price: float,
    ) -> TradeSignal:
        """Exit logic: SELL, DCA-BUY, or HOLD."""
        pnl_pct = data.pnl_pct or 0.0

        # Position aging: if held too long with flat P&L, suggest exit review
        position_age = data.position_age_hours
        if (
            position_age is not None
            and position_age > self.stale_position_hours
            and abs(pnl_pct) < self.stale_exit_pnl_threshold_pct
        ):
            return TradeSignal(
                action="SELL",
                confidence=0.6,
                composite=composite,
                reasoning=(
                    f"Stale position: held {position_age:.0f}h with "
                    f"P&L {pnl_pct:+.1f}%. Freeing capital. "
                    f"{', '.join(reasons)}"
                ),
            )

        # SELL if sentiment turned BEARISH and composite confirms.
        if sentiment_direction == "BEARISH" and composite < self.exit_threshold:
            return TradeSignal(
                action="SELL",
                confidence=min(abs(composite) + 0.1, 1.0),
                composite=composite,
                reasoning=(
                    f"Bearish sentiment + negative signals "
                    f"(composite={composite:+.3f}). {', '.join(reasons)}"
                ),
            )

        # SELL if profit exceeds profit_take_pct with weakening momentum — lock in gains.
        if pnl_pct > self.profit_take_pct and composite < self.momentum_weak_threshold:
            return TradeSignal(
                action="SELL",
                confidence=0.7,
                composite=composite,
                reasoning=(
                    f"Profit {pnl_pct:+.1f}% with weakening momentum "
                    f"(composite={composite:+.3f}). {', '.join(reasons)}"
                ),
            )

        # Strong negative composite — cut losses.
        if composite < self.strong_sell_threshold:
            return TradeSignal(
                action="SELL",
                confidence=min(abs(composite), 1.0),
                composite=composite,
                reasoning=(f"Strong negative composite={composite:+.3f}. {', '.join(reasons)}"),
            )

        # DCA opportunity — strong bullish signal on existing position.
        if composite > self.dca_threshold:
            confidence = raw_confidence
            if vol_gate > 0 and confidence <= vol_gate:
                return TradeSignal(
                    action="HOLD",
                    confidence=confidence,
                    composite=composite,
                    reasoning=(
                        f"DCA potential but {volatility_regime} volatility. {', '.join(reasons)}"
                    ),
                )
            return TradeSignal(
                action="BUY",
                confidence=confidence,
                composite=composite,
                reasoning=(
                    f"DCA opportunity: strong signals "
                    f"(composite={composite:+.3f}). {', '.join(reasons)}"
                ),
                target_price=current_price * 1.04,
            )

        # No strong signal — maintain position.
        return TradeSignal(
            action="HOLD",
            confidence=raw_confidence,
            composite=composite,
            reasoning=(f"Position maintained (composite={composite:+.3f}). {', '.join(reasons)}"),
        )


def _assert_module_defaults_match_settings() -> None:
    """Fail at import time if strategy module constants drift from Settings defaults.

    Two sources of truth is bad; we allow it only because tests and the
    backtest harness instantiate ``Strategist()`` without a settings
    argument.  This check makes any drift loud instead of silent.
    """
    from config import Settings

    fields = Settings.model_fields
    expected = {
        "entry_threshold": _ENTRY_THRESHOLD,
        "exit_threshold": _EXIT_THRESHOLD,
        "strong_sell_threshold": _STRONG_SELL_THRESHOLD,
        "dca_threshold": _DCA_THRESHOLD,
        "profit_take_pct": _PROFIT_TAKE_PCT,
        "momentum_weak_threshold": _MOMENTUM_WEAK_THRESHOLD,
        "sma_overextended_pct": _SMA_OVEREXTENDED_PCT,
        "mean_reversion_rsi": float(_MEAN_REVERSION_RSI),
        "mean_reversion_adx": float(_MEAN_REVERSION_ADX),
        "mean_reversion_bonus": _MEAN_REVERSION_BONUS,
        "stale_position_hours": float(_STALE_POSITION_HOURS),
        "stale_exit_pnl_threshold_pct": _STALE_EXIT_PNL_THRESHOLD_PCT,
    }
    drifted = [
        name
        for name, module_val in expected.items()
        if abs(fields[name].default - module_val) > 1e-9
    ]
    assert not drifted, (  # noqa: S101
        f"strategy.py module constants drift from Settings defaults: {drifted}. "
        "Update one to match the other."
    )


_assert_module_defaults_match_settings()
