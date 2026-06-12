"""Shared types, formatters, and the MarketData container."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from forecast import PriceForecast
    from indicators import IndicatorSnapshot
    from volatility import VolatilityForecast


class MarketRegime(str, Enum):
    """Market regime classification."""

    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    BREAKOUT = "BREAKOUT"


class StopMode(str, Enum):
    """Stop-loss management mode."""

    FIXED = "FIXED"
    TRAILING = "TRAILING"
    TRAILING_BREAKEVEN = "TRAILING_BREAKEVEN"


@dataclass
class MultiTimeframeContext:
    """Aggregated signals from multiple timeframes."""

    trend_bias: str = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL"
    short_tf_rsi: float | None = None  # 15m RSI for entry timing
    short_tf_bb_pct_b: float | None = None  # 15m Bollinger %B
    ema_21: float | None = None  # 4h EMA(21)
    ema_50: float | None = None  # 4h EMA(50)
    ema_bullish_cross: bool = False  # EMA21 > EMA50


@dataclass
class PositionState:
    """Per-position tracking state for stop management."""

    ticker: str
    entry_price: float
    high_watermark: float
    stop_mode: StopMode = StopMode.FIXED
    current_sl: float = 0.0
    current_tp: float = 0.0
    partial_exits: int = 0
    regime_at_entry: str = "RANGING"


def fmt_price(price: float) -> str:
    """Format price with dynamic precision for sub-cent assets."""
    assert price >= 0, f"Price must be non-negative, got {price}"  # noqa: S101

    if price >= 100:  # noqa: PLR2004
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:.4f}"
    if price >= 0.01:  # noqa: PLR2004
        return f"{price:.6f}"
    return f"{price:.10f}"


class TradeSignal(BaseModel):
    """Structured output for the strategy decision."""

    action: Literal["BUY", "SELL", "HOLD"]
    confidence: float  # 0.0 to 1.0
    composite: float = 0.0  # Raw composite score for logging/debugging
    reasoning: str
    target_price: float | None = None


@dataclass
class MarketData:
    """All data collected for a single ticker in one cycle."""

    ticker: str
    current_price: float

    # Price context
    vwap: float | None = None
    sma_7d: float | None = None
    sma_deviation_pct: float | None = None
    vwap_deviation_pct: float | None = None

    # Quantitative modules (structured)
    indicators: IndicatorSnapshot | None = None
    volatility: VolatilityForecast | None = None
    forecast: PriceForecast | None = None

    # Grok / external signals (strings)
    sentiment: str = ""
    consensus: str = ""
    fear_greed: str = ""
    polymarket_odds: str = ""

    # Position context
    has_position: bool = False
    entry_price: float | None = None
    pnl_pct: float | None = None
    tp_distance_pct: float | None = None
    position_age_hours: float | None = None

    # Journal history (string for logging)
    history: str = ""

    # Multi-timeframe and regime context
    multi_tf: MultiTimeframeContext | None = None
    regime: MarketRegime | None = None

    # v9.5: perp funding rate from secondary exchange (informational; not yet
    # fed into composite). Negative = shorts paying longs (contrarian bullish).
    funding_rate: float | None = None
