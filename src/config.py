"""Configuration management using Pydantic."""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and secrets."""

    kraken_api_key: SecretStr
    kraken_private_key: SecretStr
    xai_api_key: SecretStr
    cmc_api_key: SecretStr | None = None
    polymarket_api_key: SecretStr | None = None

    xai_model: str = "grok-4-1-fast-reasoning"
    # User allowed: grok-4-fast-reasoning, grok-4-fast-non-reasoning, grok-4-1

    quote_currency: str = "USDT"
    # The currency you hold to buy other assets (e.g., USDT, EUR, USD)

    trading_mode: Literal["PAPER", "LIVE"] = "PAPER"

    # Safety: Hard cap at $100 to prevent catastrophic tuition loss
    max_position_size_usd: float = Field(default=50.0, le=100.0)

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
    daily_loss_limit_usd: float = 20.0  # Pause trading if daily realized losses exceed this

    # Position management
    position_max_age_hours: int = 120  # Signal exit review for stale positions (was 72)

    # Healthcheck
    healthcheck_file: str = "logs/healthcheck"

    # Trading Universe (Comma separated in .env)
    trading_universe: str = "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,SHIB/USDT,LINK/USDT"

    # Bot Loop settings
    timeout_seconds: int = 3600  # Default 1 hour

    # DCA (Dollar Cost Averaging)
    max_dca_per_position: int = 1  # Max additional buys per owned position

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

    # Log rotation (time-based: 7-day retention, rotated at midnight UTC)
    log_retention_days: int = Field(default=7, ge=1, le=365)

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
