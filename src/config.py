"""Configuration management using Pydantic."""
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and secrets."""

    kraken_api_key: SecretStr
    kraken_private_key: SecretStr
    google_api_key: SecretStr
    xai_api_key: SecretStr
    cmc_api_key: SecretStr | None = None
    polymarket_api_key: SecretStr | None = None

    gemini_model: Literal["gemini-3-flash-preview", "gemini-3-pro-preview"] = (
        "gemini-3-pro-preview"
    )
    # Default to Gemini 3 Flash Preview.
    # Switch to gemini-3-pro-preview in .env for deeper analysis.

    xai_model: str = "grok-4-1-fast-reasoning"
    # User allowed: grok-4-fast-reasoning, grok-4-fast-non-reasoning, grok-4-1

    quote_currency: str = "USDT"
    # The currency you hold to buy other assets (e.g., USDT, EUR, USD)

    trading_mode: Literal["PAPER", "LIVE"] = "PAPER"

    # Safety: Hard cap at $100 to prevent catastrophic tuition loss
    max_position_size_usd: float = Field(default=50.0, le=100.0)

    # Risk Management
    stop_loss_pct: float = 0.02      # 2%
    take_profit_pct: float = 0.04    # 4%
    max_daily_drawdown_pct: float = 0.05  # 5%
    max_spread_pct: float = 0.005    # 0.5%

    # Trading Universe (Comma separated in .env)
    trading_universe: str = (
        "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,SHIB/USDT,LINK/USDT"
    )

    # Bot Loop settings
    timeout_seconds: int = 3600 # Default 1 hour

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
