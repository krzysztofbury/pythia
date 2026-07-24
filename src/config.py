"""Configuration management using Pydantic."""

import warnings
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar, Literal, Never, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

LIVE_TRADING_ACKNOWLEDGEMENT = "I_UNDERSTAND_LIVE_TRADING"
_MAX_UNIVERSE_SYMBOLS = 50
_MAX_QUOTE_CURRENCY_LENGTH = 12
_MAX_ROI_SCHEDULE_ENTRIES = 24
_MAX_ROI_SCHEDULE_MINUTES = 525600
_MAX_PERCENT = 100.0

_DEPRECATED_ENV_KEYS = frozenset(
    {
        "gemini_model",
        "google_api_key",
        "log_backup_count",
        "log_max_bytes",
        "max_daily_drawdown_pct",
        "min_confidence",
    },
)


def _invalid(message: str) -> Never:
    """Raise a configuration error without duplicating exception boilerplate."""
    raise ValueError(message)


class _CompatibleDotEnvSettingsSource(DotEnvSettingsSource):
    """Warn about unknown dotenv keys without rejecting legacy operator files."""

    def __call__(self) -> dict[str, Any]:
        values = super().__call__()
        known_keys = {name.casefold() for name in self.settings_cls.model_fields}
        for key in tuple(values):
            normalized = key.casefold()
            if normalized in known_keys or normalized in _DEPRECATED_ENV_KEYS:
                continue
            warnings.warn(
                f"Ignoring unknown dotenv key {key!r}; remove it or add a supported setting.",
                UserWarning,
                stacklevel=2,
            )
            values.pop(key)
        return values


class Settings(BaseSettings):
    """Application settings and secrets."""

    _deprecated_env_keys: ClassVar[frozenset[str]] = _DEPRECATED_ENV_KEYS

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

    # LLM features are opt-in. In LIVE, outputs remain shadow-only unless the
    # operator deliberately enables influence as part of the LIVE arming config.
    sentiment_enabled: bool = False
    llm_live_influence_enabled: bool = False
    llm_monthly_budget_usd: float = Field(default=1.0, gt=0.0, le=100.0)
    llm_daily_shadow_budget_usd: float = Field(default=0.05, gt=0.0, le=10.0)
    llm_max_tokens: int = Field(default=512, ge=64, le=4096)
    llm_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)

    # v9.6: per-ticker TTL cache for sentiment + consensus LLM calls.
    # The 2026-06 incident burned a full monthly xAI budget in ~7 days because
    # both calls ran for every ticker on every 15-min cycle (~1,344/day).
    # 0 disables caching.
    sentiment_cache_seconds: float = Field(default=3600.0, ge=0.0, le=86400.0)

    # v9.6: sentiment gate mode. False (default) = sentiment is advisory —
    # entries are blocked only on BEARISH sentiment or strong DOWN consensus.
    # True restores the legacy hard gate (require BULLISH or consensus UP>60),
    # which fully disables the bot whenever the LLM is degraded.
    require_bullish_sentiment: bool = False

    # v9.6: alert when no entry executed for N hours while LIVE (0 disables).
    # The bot traded nothing for 7 weeks in 2026-04/06 without a single alert.
    no_entry_alert_hours: float = Field(default=72.0, ge=0.0, le=720.0)

    # LLM Challenger — commentary only, NEVER blocks execution (ToS-safe).
    # The algorithm always makes the final decision; the challenger just
    # logs a second-opinion thesis and may raise a HIGH concern notification.
    challenger_enabled: bool = False
    challenger_model: str = "anthropic/claude-sonnet-4-5"
    challenger_fallback_model: str | None = "gemini/gemini-3.5-flash"

    quote_currency: str = "USDC"
    # The currency you hold to buy other assets (e.g., USDC, USDT, EUR, USD)

    trading_mode: Literal["PAPER", "LIVE"] = "PAPER"
    trading_account_id: SecretStr | None = None
    live_trading_ack: SecretStr | None = None
    live_entries_enabled: bool = False

    # Execution and analytics use fresh v10 files. Legacy trades.db is archived
    # read-only during cutover and is never opened by the v10 runtime.
    paper_state_db_path: str = "logs/paper-execution-v10.db"
    live_state_db_path: str = "logs/live-execution-v10.db"
    analytics_db_path: str = "logs/analytics-v10.db"
    llm_budget_db_path: str = "logs/llm-budget.db"
    paper_exchange_state_path: str = "logs/paper-exchange-state.json"
    singleton_lock_path: str = "logs/oracle-pythia.lock"
    paper_initial_balance_usd: float = Field(default=350.0, gt=0.0, le=1_000_000.0)
    reconciliation_balance_tolerance: float = Field(default=0.00000001, ge=0.0, le=1.0)
    shutdown_drain_timeout_seconds: float = Field(default=30.0, ge=0.0, le=300.0)

    # Safety: Hard cap at $500 to allow 2.0x conviction sizing on healthy accounts
    # while still preventing runaway allocation. Default remains conservative.
    max_position_size_usd: float = Field(default=50.0, gt=0.0, le=500.0)

    # v9.2: conviction-tiered sizing — composite score → position multiplier.
    # Disabling this flag restores v9.1 flat behaviour.
    conviction_sizing_enabled: bool = True
    max_position_pct_of_balance: float = Field(default=0.40, ge=0.05, le=1.0)

    # Tier thresholds (composite score). Must be ascending.
    sizing_tier_1_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    sizing_tier_2_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    sizing_tier_3_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    sizing_tier_4_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

    # Tier multipliers (applied to base size).
    sizing_tier_1_multiplier: float = Field(default=0.6, gt=0.0, le=10.0)
    sizing_tier_2_multiplier: float = Field(default=1.0, gt=0.0, le=10.0)
    sizing_tier_3_multiplier: float = Field(default=1.4, gt=0.0, le=10.0)
    sizing_tier_4_multiplier: float = Field(default=2.0, gt=0.0, le=10.0)

    # Risk Management
    stop_loss_pct: float = Field(default=0.02, gt=0.0, le=0.50)
    take_profit_pct: float = Field(default=0.04, gt=0.0, le=1.0)
    max_spread_pct: float = Field(default=0.005, ge=0.0, le=0.10)
    limit_order_discount_pct: float = Field(default=0.002, ge=0.0, le=0.10)

    # ATR-based dynamic stops (replaces fixed % when available)
    use_atr_stops: bool = True
    atr_sl_multiplier: float = Field(default=2.0, gt=0.0, le=20.0)
    atr_tp_multiplier: float = Field(default=3.0, gt=0.0, le=20.0)

    # Portfolio risk
    max_correlated_positions: int = Field(default=3, ge=1, le=100)
    # v9.5: portfolio-wide cap across ALL groups. The v9.5 correlation regrouping
    # (btc_core / eth_core / l1_alt / defi / payments / meme) relaxed per-group
    # constraints; this backstop keeps total concurrent crypto exposure bounded
    # regardless of how operators slice up the groups.
    max_total_crypto_positions: int = Field(default=5, ge=1, le=100)
    daily_loss_limit_usd: float = Field(default=20.0, gt=0.0, le=100000.0)

    # Position management
    position_max_age_hours: int = Field(default=120, ge=1, le=8760)

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
    timeout_seconds: int = Field(default=3600, ge=10, le=86400)

    # DCA (Dollar Cost Averaging)
    max_dca_per_position: int = Field(default=1, ge=0, le=20)

    # Entry pacing
    max_entries_per_cycle: int = Field(default=2, ge=0, le=100)

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
    min_buy_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    min_sell_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # Strategy thresholds (composite-score based; previously hardcoded in strategy.py)
    entry_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    exit_threshold: float = Field(default=-0.05, ge=-1.0, le=1.0)
    strong_sell_threshold: float = Field(default=-0.25, ge=-1.0, le=1.0)
    dca_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    profit_take_pct: float = Field(default=5.0, gt=0.0, le=100.0)
    momentum_weak_threshold: float = Field(default=0.1, ge=-1.0, le=1.0)
    sma_overextended_pct: float = Field(default=15.0, ge=0.0, le=100.0)
    mean_reversion_rsi: float = Field(default=28.0, ge=0.0, le=100.0)
    mean_reversion_adx: float = Field(default=30.0, ge=0.0, le=100.0)
    mean_reversion_bonus: float = Field(default=0.10, ge=0.0, le=1.0)
    stale_position_hours: float = Field(default=120.0, gt=0.0, le=8760.0)
    stale_exit_pnl_threshold_pct: float = Field(default=2.0, ge=0.0, le=100.0)

    # Stale Order Reconciliation
    stale_order_max_age_seconds: int = Field(default=7200, ge=1, le=604800)

    # Technical Analysis
    ta_candle_timeframe: str = "1h"
    ta_candle_limit: int = Field(default=100, ge=20, le=10000)

    # GARCH Volatility
    garch_enabled: bool = True
    garch_candle_timeframe: str = "4h"
    garch_candle_limit: int = Field(default=200, ge=20, le=10000)

    # ARIMA Forecasting
    arima_enabled: bool = True
    arima_max_order: int = Field(default=5, ge=1, le=20)

    # Polymarket via WARP proxy
    warp_proxy: str | None = None
    polymarket_enabled: bool = True
    polymarket_strategy_influence_enabled: bool = False

    # Dynamic Risk (informational only by default)
    dynamic_risk_enabled: bool = False

    # Trailing Stop-Loss
    trailing_stop_enabled: bool = True
    trailing_sl_atr_multiplier: float = Field(default=1.5, gt=0.0, le=20.0)
    breakeven_atr_threshold: float = Field(default=1.0, gt=0.0, le=20.0)

    # Time-based ROI decay (minutes:profit_pct pairs, comma-separated)
    roi_enabled: bool = True
    roi_schedule: str = "0:4.0,60:3.0,120:2.0,240:1.0"

    # Multi-timeframe analysis
    mtf_enabled: bool = True
    mtf_short_timeframe: str = "15m"
    mtf_long_timeframe: str = "4h"

    # Cycle timing
    entry_cycle_seconds: int = Field(default=900, ge=10, le=86400)
    full_cycle_seconds: int = Field(default=3600, ge=10, le=86400)

    # Partial exits
    partial_exit_enabled: bool = True
    partial_exit_pct: float = Field(default=0.50, gt=0.0, le=1.0)
    partial_exit_atr_threshold: float = Field(default=2.0, gt=0.0, le=20.0)

    # Optimizer
    optimized_params_file: str | None = None

    # Tax
    tax_rate: float = Field(default=0.19, ge=0.0, le=1.0)

    # Notifications (Discord/Slack webhook)
    webhook_url: str | None = None
    error_webhook_url: str | None = None

    # Log rotation (time-based, rotated at midnight UTC). v9.6: bumped 7 → 30 —
    # the 7-day window destroyed the evidence needed to diagnose the 2026-04/05
    # no-trades incident.
    log_retention_days: int = Field(default=30, ge=1, le=365)

    @field_validator(
        "xai_api_key",
        "cmc_api_key",
        "polymarket_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        "openai_api_key",
        "trading_account_id",
        "live_trading_ack",
        mode="before",
    )
    @classmethod
    def empty_optional_secret_is_unset(cls, value: object) -> object:
        """Treat empty optional secrets from dotenv as absent."""
        if value is None:
            return None
        raw_value = value.get_secret_value() if isinstance(value, SecretStr) else str(value)
        return None if not raw_value.strip() else value

    @field_validator("kraken_api_key", "kraken_private_key")
    @classmethod
    def required_secret_is_not_empty(cls, value: SecretStr) -> SecretStr:
        """Reject empty exchange credentials without exposing their values."""
        if not value.get_secret_value().strip():
            _invalid("required secret must not be empty")
        return value

    @field_validator("quote_currency")
    @classmethod
    def normalize_quote_currency(cls, value: str) -> str:
        """Normalize and validate the account quote currency."""
        normalized = value.strip().upper()
        if (
            not normalized
            or not normalized.isalnum()
            or len(normalized) > _MAX_QUOTE_CURRENCY_LENGTH
        ):
            _invalid("quote_currency must be 1-12 alphanumeric characters")
        return normalized

    @field_validator("sentiment_model", "challenger_model")
    @classmethod
    def required_model_is_not_empty(cls, value: str) -> str:
        """Reject empty primary model names."""
        if not value.strip():
            _invalid("model name must not be empty")
        return value.strip()

    @field_validator("sentiment_fallback_model", "challenger_fallback_model", mode="before")
    @classmethod
    def empty_optional_model_is_unset(cls, value: object) -> object:
        """Treat blank fallback model values as disabled."""
        if value is None:
            return None
        return None if not str(value).strip() else str(value).strip()

    @field_validator("roi_schedule")
    @classmethod
    def validate_roi_schedule(cls, value: str) -> str:
        """Require a bounded, ordered, non-increasing ROI decay schedule."""
        normalized = value.strip().strip('"').strip("'")
        entries = normalized.split(",") if normalized else []
        if not 1 <= len(entries) <= _MAX_ROI_SCHEDULE_ENTRIES:
            _invalid("roi_schedule must contain 1-24 entries")

        parsed: list[tuple[int, float]] = []
        for entry in entries:
            parts = entry.strip().split(":")
            if len(parts) != 2:  # noqa: PLR2004
                _invalid("roi_schedule entries must use minutes:profit_pct")
            try:
                minutes = int(parts[0])
                profit_pct = float(parts[1])
            except ValueError:
                _invalid("roi_schedule entries must be numeric")
            if not 0 <= minutes <= _MAX_ROI_SCHEDULE_MINUTES:
                _invalid("roi_schedule minutes must be between 0 and 525600")
            if not 0.0 <= profit_pct <= _MAX_PERCENT:
                _invalid("roi_schedule profit percentages must be between 0 and 100")
            parsed.append((minutes, profit_pct))

        if parsed[0][0] != 0:
            _invalid("roi_schedule must start at minute 0")
        if any(current[0] >= following[0] for current, following in pairwise(parsed)):
            _invalid("roi_schedule minutes must be strictly increasing")
        if any(current[1] < following[1] for current, following in pairwise(parsed)):
            _invalid("roi_schedule profit percentages must not increase")
        return normalized

    @model_validator(mode="before")
    @classmethod
    def migrate_deprecated_settings(cls, data: object) -> object:
        """Migrate known legacy dotenv names and explicitly deprecate removed knobs."""
        if not isinstance(data, Mapping):
            return data
        values = dict(data)
        keys_by_name = {str(key).casefold(): key for key in values}

        def pop_legacy(name: str) -> object | None:
            key = keys_by_name.get(name)
            if key is None:
                return None
            warnings.warn(
                f"{name.upper()} is deprecated and should be removed from the dotenv file.",
                DeprecationWarning,
                stacklevel=2,
            )
            return values.pop(key)

        google_key = pop_legacy("google_api_key")
        if google_key is not None and "gemini_api_key" not in keys_by_name:
            values["gemini_api_key"] = google_key

        gemini_model = pop_legacy("gemini_model")
        if gemini_model is not None and "sentiment_model" not in keys_by_name:
            model_name = str(gemini_model).strip()
            values["sentiment_model"] = model_name if "/" in model_name else f"gemini/{model_name}"

        for name in cls._deprecated_env_keys - {"google_api_key", "gemini_model"}:
            pop_legacy(name)

        if "xai_model" in keys_by_name:
            warnings.warn(
                "XAI_MODEL is deprecated; use SENTIMENT_MODEL=xai/<model> instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return values

    @model_validator(mode="after")
    def validate_universe_and_quote(self) -> Self:
        """Require a unique, bounded universe using the configured quote currency."""
        symbols = [symbol.strip() for symbol in self.trading_universe.split(",")]
        if not 1 <= len(symbols) <= _MAX_UNIVERSE_SYMBOLS or any(not symbol for symbol in symbols):
            _invalid("trading_universe must contain 1-50 non-empty symbols")
        if len({symbol.casefold() for symbol in symbols}) != len(symbols):
            _invalid("trading_universe symbols must be unique")
        for symbol in symbols:
            parts = symbol.split("/")
            if len(parts) != 2 or not all(part.isalnum() for part in parts):  # noqa: PLR2004
                _invalid("trading_universe symbols must use BASE/QUOTE format")
            if parts[1].upper() != self.quote_currency:
                _invalid("every trading_universe symbol must match quote_currency")
        self.trading_universe = ",".join(symbols)
        return self

    @model_validator(mode="after")
    def validate_risk_relationships(self) -> Self:
        """Validate ordering assumptions used by sizing, strategy, and stops."""
        tiers = (
            self.sizing_tier_1_threshold,
            self.sizing_tier_2_threshold,
            self.sizing_tier_3_threshold,
            self.sizing_tier_4_threshold,
        )
        if any(current >= following for current, following in pairwise(tiers)):
            _invalid("conviction sizing tier thresholds must be strictly ascending")
        multipliers = (
            self.sizing_tier_1_multiplier,
            self.sizing_tier_2_multiplier,
            self.sizing_tier_3_multiplier,
            self.sizing_tier_4_multiplier,
        )
        if any(current > following for current, following in pairwise(multipliers)):
            _invalid("conviction sizing tier multipliers must not decrease")
        if not self.strong_sell_threshold < self.exit_threshold < self.entry_threshold:
            _invalid("strategy thresholds must order strong sell < exit < entry")
        if self.dca_threshold < self.entry_threshold:
            _invalid("dca_threshold must be at least entry_threshold")
        if self.take_profit_pct <= self.stop_loss_pct:
            _invalid("take_profit_pct must be greater than stop_loss_pct")
        if self.atr_tp_multiplier <= self.atr_sl_multiplier:
            _invalid("atr_tp_multiplier must be greater than atr_sl_multiplier")
        if self.trailing_sl_atr_multiplier >= self.atr_tp_multiplier:
            _invalid("trailing stop distance must be below the ATR take-profit distance")
        return self

    @model_validator(mode="after")
    def validate_counts_and_timing(self) -> Self:
        """Validate portfolio caps and nested cycle timing."""
        if self.max_correlated_positions > self.max_total_crypto_positions:
            _invalid("max_correlated_positions cannot exceed the total position cap")
        if self.max_entries_per_cycle > self.max_total_crypto_positions:
            _invalid("max_entries_per_cycle cannot exceed the total position cap")
        if self.entry_cycle_seconds > self.full_cycle_seconds:
            _invalid("entry_cycle_seconds cannot exceed full_cycle_seconds")
        if self.full_cycle_seconds % self.entry_cycle_seconds != 0:
            _invalid("full_cycle_seconds must be a multiple of entry_cycle_seconds")
        if self.full_cycle_seconds > self.timeout_seconds:
            _invalid("full_cycle_seconds cannot exceed timeout_seconds")
        return self

    @model_validator(mode="after")
    def validate_state_paths_and_live_arming(self) -> Self:  # noqa: C901
        """Keep PAPER/LIVE state isolated and require three-part LIVE arming."""
        path_values = (
            self.paper_state_db_path,
            self.live_state_db_path,
            self.analytics_db_path,
            self.llm_budget_db_path,
            self.paper_exchange_state_path,
            self.singleton_lock_path,
            self.healthcheck_file,
        )
        if any(not value.strip() for value in path_values):
            _invalid("runtime state paths must not be empty")
        paper_path = Path(self.paper_state_db_path).expanduser().resolve()
        live_path = Path(self.live_state_db_path).expanduser().resolve()
        if paper_path == live_path:
            _invalid("PAPER and LIVE state database paths must be different")
        analytics_path = Path(self.analytics_db_path).expanduser().resolve()
        budget_path = Path(self.llm_budget_db_path).expanduser().resolve()
        legacy_path = Path("logs/trades.db").resolve()
        if legacy_path in (paper_path, live_path):
            _invalid("legacy logs/trades.db cannot be used as execution state")
        if paper_path in {analytics_path, budget_path} or live_path in {
            analytics_path,
            budget_path,
        }:
            _invalid("execution, analytics, and LLM budget databases must be different")
        if analytics_path == legacy_path:
            _invalid("legacy logs/trades.db must remain a read-only archive")
        resolved_paths = [Path(value).expanduser().resolve() for value in path_values]
        if len(resolved_paths) != len(set(resolved_paths)):
            _invalid("all runtime state paths must be pairwise distinct")

        if self.trading_mode == "PAPER":
            if self.live_entries_enabled:
                _invalid("LIVE_ENTRIES_ENABLED must be false in PAPER mode")
            return self
        if self.trading_account_id is None:
            _invalid("TRADING_ACCOUNT_ID is required in LIVE mode")
        if self.live_entries_enabled:
            if self.live_trading_ack is None:
                _invalid("LIVE_TRADING_ACK is required when LIVE entries are enabled")
            if self.live_trading_ack.get_secret_value() != LIVE_TRADING_ACKNOWLEDGEMENT:
                _invalid("LIVE_TRADING_ACK does not match the required acknowledgement")
        return self

    @model_validator(mode="after")
    def validate_llm_features(self) -> Self:
        """Require provider credentials only for explicitly enabled LLM features."""
        if self.challenger_enabled and not self.sentiment_enabled:
            _invalid("CHALLENGER_ENABLED requires SENTIMENT_ENABLED so disabled sentiment is free")
        if self.require_bullish_sentiment and not self.sentiment_enabled:
            _invalid("REQUIRE_BULLISH_SENTIMENT requires SENTIMENT_ENABLED")
        if self.llm_live_influence_enabled and not self.sentiment_enabled:
            _invalid("LLM live influence requires SENTIMENT_ENABLED")
        if self.require_bullish_sentiment and not self.llm_live_influence_enabled:
            _invalid("REQUIRE_BULLISH_SENTIMENT requires explicit LLM live influence")
        if self.polymarket_strategy_influence_enabled and not self.polymarket_enabled:
            _invalid("Polymarket strategy influence requires POLYMARKET_ENABLED")

        models: list[str] = []
        if self.sentiment_enabled:
            models.extend(
                model for model in (self.sentiment_model, self.sentiment_fallback_model) if model
            )
        if self.challenger_enabled:
            models.extend(
                model for model in (self.challenger_model, self.challenger_fallback_model) if model
            )
        for model in models:
            provider = model.split("/", maxsplit=1)[0].casefold() if "/" in model else "xai"
            key = getattr(self, f"{provider}_api_key", None)
            if not isinstance(key, SecretStr) or not key.get_secret_value().strip():
                _invalid(f"enabled LLM model provider {provider!r} requires its API key")
        return self

    @property
    def universe_list(self) -> list[str]:
        """Parse comma-separated trading universe."""
        return [s.strip() for s in self.trading_universe.split(",") if s.strip()]

    @property
    def state_db_path(self) -> str:
        """Return the state database selected for the active trading mode."""
        if self.trading_mode == "LIVE":
            return self.live_state_db_path
        return self.paper_state_db_path

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Use a compatibility-aware dotenv source; keep direct inputs strict."""
        if not isinstance(dotenv_settings, DotEnvSettingsSource):
            return init_settings, env_settings, dotenv_settings, file_secret_settings
        compatible_dotenv = _CompatibleDotEnvSettingsSource(
            settings_cls,
            env_file=dotenv_settings.env_file,
            env_file_encoding=dotenv_settings.env_file_encoding,
            case_sensitive=dotenv_settings.case_sensitive,
            env_prefix=dotenv_settings.env_prefix,
            env_nested_delimiter=dotenv_settings.env_nested_delimiter,
            env_nested_max_split=dotenv_settings.env_nested_max_split,
            env_ignore_empty=dotenv_settings.env_ignore_empty,
            env_parse_none_str=dotenv_settings.env_parse_none_str,
            env_parse_enums=dotenv_settings.env_parse_enums,
        )
        return init_settings, env_settings, compatible_dotenv, file_secret_settings

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


# Global settings instance
# This will raise a ValidationError at import time if required env vars are missing.
# Ideally, this should be instantiated inside a main function, but global is common for config.
# We will instantiate it lazily or handle the error in main.
# For now, we expose the class.
