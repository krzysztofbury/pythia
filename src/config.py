"""Configuration management using Pydantic."""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and secrets."""

    kraken_api_key: SecretStr
    kraken_private_key: SecretStr
    xai_api_key: SecretStr | None = None
    cmc_api_key: SecretStr | None = None
    polymarket_api_key: SecretStr | None = None

    # Optional LLM provider keys (used only if the corresponding model is
    # selected via challenger_model / challenger_fallback_model / sentiment_model).
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    xai_model: str = "grok-4-1-fast-reasoning"
    # Legacy (pre-v9.6) knob, no longer wired to anything. Kept so old .env
    # files load cleanly; to run Grok sentiment set SENTIMENT_MODEL=xai/<model>.

    # Sentiment model — LiteLLM provider/model format (e.g. "gemini/gemini-3.5-flash").
    # v9.6: Gemini Flash is the primary; bare legacy strings are auto-prefixed
    # with "xai/" for back-compat with old .env files.
    sentiment_model: str = "gemini/gemini-3.5-flash"
    sentiment_fallback_model: str | None = None

    # v9.6: per-ticker TTL cache for sentiment + consensus LLM calls.
    # The 2026-06 incident burned a full monthly xAI budget in ~7 days because
    # both calls ran for every ticker on every 15-min cycle (~1,344/day).
    # 0 disables caching.
    sentiment_cache_seconds: float = Field(default=3600.0, ge=0.0)

    # v9.6: sentiment gate mode. False (default) = sentiment is advisory —
    # entries are blocked only on BEARISH sentiment or strong DOWN consensus.
    # True restores the legacy hard gate (require BULLISH or consensus UP>60),
    # which fully disables the bot whenever the LLM is degraded.
    require_bullish_sentiment: bool = False

    # v9.6: alert when no entry executed for N hours while LIVE (0 disables).
    # The bot traded nothing for 7 weeks in 2026-04/06 without a single alert.
    no_entry_alert_hours: float = Field(default=72.0, ge=0.0)

    # LLM Challenger — commentary only, NEVER blocks execution (ToS-safe).
    # The algorithm always makes the final decision; the challenger just
    # logs a second-opinion thesis and may raise a HIGH concern notification.
    challenger_enabled: bool = False
    challenger_model: str = "anthropic/claude-sonnet-4-5"
    challenger_fallback_model: str | None = "gemini/gemini-3.5-flash"

    quote_currency: str = "USDC"
    # The currency you hold to buy other assets (e.g., USDC, USDT, EUR, USD)

    trading_mode: Literal["PAPER", "LIVE"] = "PAPER"

    # Safety: Hard cap at $500 to allow 2.0x conviction sizing on healthy accounts
    # while still preventing runaway allocation. Default remains conservative.
    max_position_size_usd: float = Field(default=50.0, le=500.0)

    # v9.2: conviction-tiered sizing — composite score → position multiplier.
    # Disabling this flag restores v9.1 flat behaviour.
    conviction_sizing_enabled: bool = True
    max_position_pct_of_balance: float = Field(default=0.40, ge=0.05, le=1.0)

    # Tier thresholds (composite score). Must be ascending.
    sizing_tier_1_threshold: float = 0.25
    sizing_tier_2_threshold: float = 0.35
    sizing_tier_3_threshold: float = 0.50
    sizing_tier_4_threshold: float = 0.70

    # Tier multipliers (applied to base size).
    sizing_tier_1_multiplier: float = 0.6
    sizing_tier_2_multiplier: float = 1.0
    sizing_tier_3_multiplier: float = 1.4
    sizing_tier_4_multiplier: float = 2.0

    # Risk Management
    stop_loss_pct: float = 0.02  # 2% (fallback when ATR unavailable)
    take_profit_pct: float = 0.04  # 4% (fallback when ATR unavailable)
    max_spread_pct: float = 0.005  # 0.5%
    limit_order_discount_pct: float = 0.002  # 0.2% below market for limit buys

    # ATR-based dynamic stops (replaces fixed % when available)
    use_atr_stops: bool = True
    atr_sl_multiplier: float = 2.0  # SL = entry - (ATR * multiplier)
    atr_tp_multiplier: float = 3.0  # TP = entry + (ATR * multiplier)

    # Portfolio risk
    max_correlated_positions: int = 3  # Max simultaneous correlated positions
    # v9.5: portfolio-wide cap across ALL groups. The v9.5 correlation regrouping
    # (btc_core / eth_core / l1_alt / defi / payments / meme) relaxed per-group
    # constraints; this backstop keeps total concurrent crypto exposure bounded
    # regardless of how operators slice up the groups.
    max_total_crypto_positions: int = 5
    daily_loss_limit_usd: float = 20.0  # Pause trading if daily realized losses exceed this

    # Position management
    position_max_age_hours: int = 120  # Signal exit review for stale positions (was 72)

    # Healthcheck
    healthcheck_file: str = "logs/healthcheck"

    # Trading Universe (Comma separated in .env).
    # v9.5 default: USDC quote (matches live .env), adds AVAX/DOT for L1
    # diversification, drops SHIB (meme volatility per v9.4 per-asset analysis).
    # All pairs verified live on Kraken spot in USDC.
    trading_universe: str = "BTC/USDC,ETH/USDC,SOL/USDC,AVAX/USDC,DOT/USDC,LINK/USDC,XRP/USDC"

    # v9.5: perpetual funding-rate signal (read-only, secondary exchange).
    # Disabled by default; the data is logged to market_snapshots when enabled
    # so v9.6+ can use the historical series for factor analysis.
    funding_rates_enabled: bool = False
    funding_rates_exchange: str = "binance"

    # Bot Loop settings
    timeout_seconds: int = 3600  # Default 1 hour

    # DCA (Dollar Cost Averaging)
    max_dca_per_position: int = 1  # Max additional buys per owned position

    # Entry pacing
    max_entries_per_cycle: int = 2  # Max new positions per 15-min cycle

    # Signal confidence gates (secondary floor above strategy thresholds).
    # 0.0 = defer to strategy's entry_threshold / exit_threshold (recommended default).
    #
    # Historical note: the legacy `MIN_CONFIDENCE=0.10` was Gemini-era. On the BUY
    # path it was always satisfied (entry_threshold=0.25 > 0.10). On the SELL path
    # it was also never triggered — every SELL branch in Strategist sets
    # confidence to 0.6/0.7 or `min(abs(composite)+0.1, 1.0)`, all ≥ 0.15 by
    # construction. Removing the gate is therefore safe for the current
    # strategy; if a new weaker SELL branch is added, set min_sell_confidence
    # explicitly rather than relying on this floor.
    min_buy_confidence: float = 0.0
    min_sell_confidence: float = 0.0

    # Strategy thresholds (composite-score based; previously hardcoded in strategy.py)
    entry_threshold: float = 0.25
    exit_threshold: float = -0.05
    strong_sell_threshold: float = -0.25
    dca_threshold: float = 0.30
    profit_take_pct: float = 5.0
    momentum_weak_threshold: float = 0.1
    sma_overextended_pct: float = 15.0
    mean_reversion_rsi: float = 28.0
    mean_reversion_adx: float = 30.0
    mean_reversion_bonus: float = 0.10
    stale_position_hours: float = 120.0
    stale_exit_pnl_threshold_pct: float = 2.0

    # Stale Order Reconciliation
    stale_order_max_age_seconds: int = 7200  # 2 hours

    # Technical Analysis
    ta_candle_timeframe: str = "1h"
    ta_candle_limit: int = 100

    # GARCH Volatility
    garch_enabled: bool = True
    garch_candle_timeframe: str = "4h"
    garch_candle_limit: int = 200

    # ARIMA Forecasting
    arima_enabled: bool = True
    arima_max_order: int = 5

    # Polymarket via WARP proxy
    warp_proxy: str | None = None
    polymarket_enabled: bool = True

    # Dynamic Risk (informational only by default)
    dynamic_risk_enabled: bool = False

    # Trailing Stop-Loss
    trailing_stop_enabled: bool = True
    trailing_sl_atr_multiplier: float = 1.5  # SL at high_watermark - 1.5*ATR
    breakeven_atr_threshold: float = 1.0  # Move SL to entry after +1*ATR profit

    # Time-based ROI decay (minutes:profit_pct pairs, comma-separated)
    roi_enabled: bool = True
    roi_schedule: str = "0:4.0,60:3.0,120:2.0,240:1.0"

    # Multi-timeframe analysis
    mtf_enabled: bool = True
    mtf_short_timeframe: str = "15m"
    mtf_long_timeframe: str = "4h"

    # Cycle timing
    entry_cycle_seconds: int = 900  # 15 min fast cycle for entries/stops
    full_cycle_seconds: int = 3600  # 1 hour full analysis cycle

    # Partial exits
    partial_exit_enabled: bool = True
    partial_exit_pct: float = 0.50  # Sell 50% of position
    partial_exit_atr_threshold: float = 2.0  # After 2*ATR profit

    # Optimizer
    optimized_params_file: str | None = None

    # Tax
    tax_rate: float = 0.19

    # Notifications (Discord/Slack webhook)
    webhook_url: str | None = None
    error_webhook_url: str | None = None

    # Log rotation (time-based, rotated at midnight UTC). v9.6: bumped 7 → 30 —
    # the 7-day window destroyed the evidence needed to diagnose the 2026-04/05
    # no-trades incident.
    log_retention_days: int = Field(default=30, ge=1, le=365)

    @property
    def universe_list(self) -> list[str]:
        """Parse comma-separated trading universe."""
        return [s.strip() for s in self.trading_universe.split(",") if s.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Global settings instance
# This will raise a ValidationError at import time if required env vars are missing.
# Ideally, this should be instantiated inside a main function, but global is common for config.
# We will instantiate it lazily or handle the error in main.
# For now, we expose the class.
