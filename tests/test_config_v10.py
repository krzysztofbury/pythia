"""Strict configuration safety tests for v10."""
# ruff: noqa: ANN201, D101, D102, D103

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from config import LIVE_TRADING_ACKNOWLEDGEMENT, Settings


def _settings(**overrides: object) -> Settings:
    values = {
        "kraken_api_key": "kraken-key",
        "kraken_private_key": "kraken-private-key",
        **overrides,
    }
    with patch.dict(os.environ, {}, clear=True):
        return Settings(_env_file=None, **values)


class TestSafeDefaults:
    def test_paper_defaults_remain_valid(self):
        settings = _settings()

        assert settings.trading_mode == "PAPER"
        assert settings.max_position_size_usd == 50.0  # noqa: PLR2004
        assert settings.quote_currency == "USDC"
        assert settings.state_db_path == "logs/paper-execution-v10.db"
        assert settings.analytics_db_path == "logs/analytics-v10.db"

    def test_llm_defaults_are_disabled_shadow_budget(self):
        settings = _settings()

        assert settings.sentiment_enabled is False
        assert settings.llm_live_influence_enabled is False
        assert settings.llm_monthly_budget_usd == 1.0
        assert settings.llm_max_tokens == 512  # noqa: PLR2004
        assert settings.sentiment_cache_seconds == 3600.0  # noqa: PLR2004


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stop_loss_pct", 0.0),
        ("take_profit_pct", 1.1),
        ("max_spread_pct", 0.11),
        ("partial_exit_pct", 1.1),
        ("tax_rate", float("inf")),
        ("max_total_crypto_positions", 0),
        ("max_dca_per_position", 21),
        ("max_entries_per_cycle", 101),
        ("timeout_seconds", 9),
        ("entry_cycle_seconds", 0),
        ("stale_order_max_age_seconds", 604801),
        ("llm_max_tokens", 63),
        ("llm_timeout_seconds", 121.0),
        ("sentiment_cache_seconds", 86401.0),
        ("llm_monthly_budget_usd", float("nan")),
        ("ta_candle_limit", 19),
        ("arima_max_order", 21),
    ],
)
def test_numeric_safety_bounds(field: str, value: object):
    with pytest.raises(ValidationError):
        _settings(**{field: value})


class TestCrossFieldSafety:
    def test_conviction_thresholds_must_be_strictly_ascending(self):
        with pytest.raises(ValidationError, match="strictly ascending"):
            _settings(sizing_tier_2_threshold=0.50, sizing_tier_3_threshold=0.50)

    def test_conviction_multipliers_must_not_decrease(self):
        with pytest.raises(ValidationError, match="must not decrease"):
            _settings(sizing_tier_2_multiplier=1.5, sizing_tier_3_multiplier=1.4)

    @pytest.mark.parametrize(
        "schedule",
        [
            "60:3.0,120:2.0",
            "0:4.0,60:3.0,60:2.0",
            "0:3.0,60:4.0",
            "0:nan",
            "0:4.0,bad",
        ],
    )
    def test_roi_schedule_must_be_bounded_ordered_decay(self, schedule: str):
        with pytest.raises(ValidationError):
            _settings(roi_schedule=schedule)

    def test_quoted_roi_schedule_is_normalized(self):
        settings = _settings(roi_schedule='"0:4.0,60:3.0"')

        assert settings.roi_schedule == "0:4.0,60:3.0"

    def test_universe_must_be_unique(self):
        with pytest.raises(ValidationError, match="unique"):
            _settings(trading_universe="BTC/USDC,btc/USDC")

    def test_universe_quote_must_match_account_quote(self):
        with pytest.raises(ValidationError, match="quote_currency"):
            _settings(trading_universe="BTC/USD,ETH/USDC")

    def test_take_profit_must_exceed_stop_loss(self):
        with pytest.raises(ValidationError, match="take_profit_pct"):
            _settings(stop_loss_pct=0.04, take_profit_pct=0.04)

    def test_atr_take_profit_must_exceed_atr_stop(self):
        with pytest.raises(ValidationError, match="atr_tp_multiplier"):
            _settings(atr_sl_multiplier=3.0, atr_tp_multiplier=3.0)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"entry_cycle_seconds": 900, "full_cycle_seconds": 600},
            {"entry_cycle_seconds": 1000, "full_cycle_seconds": 3600},
            {"full_cycle_seconds": 7200, "timeout_seconds": 3600},
        ],
    )
    def test_cycle_timing_must_be_nested(self, overrides: dict[str, object]):
        with pytest.raises(ValidationError):
            _settings(**overrides)

    def test_portfolio_sub_caps_cannot_exceed_total(self):
        with pytest.raises(ValidationError, match="total position cap"):
            _settings(max_correlated_positions=6, max_total_crypto_positions=5)


class TestLiveArmingAndState:
    @pytest.mark.parametrize(
        "overrides",
        [
            {"trading_mode": "LIVE"},
            {
                "trading_mode": "LIVE",
                "trading_account_id": "account-1",
                "live_trading_ack": "wrong",
                "live_entries_enabled": True,
            },
        ],
    )
    def test_live_requires_all_three_arming_settings(self, overrides: dict[str, object]):
        with pytest.raises(ValidationError):
            _settings(**overrides)

    def test_fully_armed_live_selects_live_state(self):
        settings = _settings(
            trading_mode="LIVE",
            trading_account_id="account-1",
            live_trading_ack=LIVE_TRADING_ACKNOWLEDGEMENT,
            live_entries_enabled=True,
        )

        assert settings.state_db_path == "logs/live-execution-v10.db"
        assert isinstance(settings.trading_account_id, SecretStr)
        assert isinstance(settings.live_trading_ack, SecretStr)

    def test_live_recovery_allows_entries_disabled(self):
        settings = _settings(
            trading_mode="LIVE",
            trading_account_id="account-1",
            live_entries_enabled=False,
        )

        assert settings.state_db_path == "logs/live-execution-v10.db"
        assert settings.live_entries_enabled is False
        assert settings.live_trading_ack is None

    def test_paper_rejects_live_entries_switch(self):
        with pytest.raises(ValidationError, match="PAPER mode"):
            _settings(live_entries_enabled=True)

    def test_paper_and_live_databases_must_be_distinct(self):
        with pytest.raises(ValidationError, match="must be different"):
            _settings(
                paper_state_db_path="logs/state.db",
                live_state_db_path="logs/../logs/state.db",
            )

    def test_legacy_trades_database_is_never_execution_state(self):
        with pytest.raises(ValidationError, match=r"legacy logs/trades\.db"):
            _settings(paper_state_db_path="logs/trades.db")

    def test_analytics_database_cannot_alias_execution_state(self, tmp_path: Path):
        shared = str(tmp_path / "shared.db")
        with pytest.raises(ValidationError, match="must be different"):
            _settings(paper_state_db_path=shared, analytics_db_path=shared)


class TestLlmFeatureGates:
    def test_disabled_features_do_not_require_provider_keys(self):
        settings = _settings(
            sentiment_model="unsupported/model",
            challenger_model="unsupported/model",
        )

        assert settings.sentiment_enabled is False
        assert settings.challenger_enabled is False

    def test_enabled_sentiment_requires_primary_provider_key(self):
        with pytest.raises(ValidationError, match="gemini"):
            _settings(sentiment_enabled=True)

    def test_challenger_cannot_spend_when_sentiment_is_disabled(self):
        with pytest.raises(ValidationError, match="SENTIMENT_ENABLED"):
            _settings(challenger_enabled=True, anthropic_api_key="anthropic-key")

    def test_enabled_sentiment_requires_fallback_provider_key(self):
        with pytest.raises(ValidationError, match="xai"):
            _settings(
                sentiment_enabled=True,
                gemini_api_key="gemini-key",
                sentiment_fallback_model="xai/grok",
            )

    def test_enabled_sentiment_accepts_keys_for_selected_providers(self):
        settings = _settings(
            sentiment_enabled=True,
            gemini_api_key="gemini-key",
            xai_api_key="xai-key",
            sentiment_fallback_model="xai/grok",
        )

        assert settings.sentiment_enabled is True

    def test_enabled_challenger_requires_only_its_selected_provider(self):
        settings = _settings(
            sentiment_enabled=True,
            gemini_api_key="gemini-key",
            challenger_enabled=True,
            challenger_fallback_model=None,
            anthropic_api_key="anthropic-key",
        )

        assert settings.challenger_enabled is True

    def test_live_influence_requires_explicit_sentiment_feature(self):
        with pytest.raises(ValidationError, match="SENTIMENT_ENABLED"):
            _settings(llm_live_influence_enabled=True)

    def test_bullish_gate_requires_explicit_live_influence(self):
        with pytest.raises(ValidationError, match="live influence"):
            _settings(
                sentiment_enabled=True,
                gemini_api_key="gemini-key",
                require_bullish_sentiment=True,
            )


class TestDotenvCompatibilityAndSecrets:
    def test_unknown_direct_setting_is_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _settings(max_postion_size_usd=10.0)

    def test_unknown_dotenv_setting_warns_and_is_ignored(self, tmp_path: Path):
        dotenv = tmp_path / "unknown.env"
        dotenv.write_text(
            "KRAKEN_API_KEY=test\nKRAKEN_PRIVATE_KEY=test\nMAX_POSTION_SIZE_USD=10\n",
            encoding="utf-8",
        )

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.warns(UserWarning, match="max_postion_size_usd"),
        ):
            settings = Settings(_env_file=dotenv)

        assert settings.max_position_size_usd == 50.0  # noqa: PLR2004

    def test_legacy_dotenv_keys_are_migrated_or_deprecated(self, tmp_path: Path):
        dotenv = tmp_path / "legacy.env"
        dotenv.write_text(
            "KRAKEN_API_KEY=test\n"
            "KRAKEN_PRIVATE_KEY=test\n"
            "GOOGLE_API_KEY=legacy-key\n"
            "GEMINI_MODEL=gemini-legacy\n"
            "LOG_MAX_BYTES=10485760\n",
            encoding="utf-8",
        )

        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.warns(DeprecationWarning, match="deprecated"),
        ):
            settings = Settings(_env_file=dotenv)

        assert isinstance(settings.gemini_api_key, SecretStr)
        assert settings.sentiment_model == "gemini/gemini-legacy"
        assert not hasattr(settings, "log_max_bytes")

    def test_secrets_and_validation_inputs_are_not_rendered(self):
        settings = _settings(
            trading_account_id="private-account-id",
            live_trading_ack="private-ack",
            gemini_api_key="private-gemini-key",
        )
        rendered = repr(settings) + settings.model_dump_json()

        assert "kraken-private-key" not in rendered
        assert "private-account-id" not in rendered
        assert "private-ack" not in rendered
        assert "private-gemini-key" not in rendered

        with pytest.raises(ValidationError) as error:
            _settings(
                trading_mode="LIVE",
                trading_account_id="private-account-id",
                live_trading_ack="DO_NOT_RENDER_THIS",
                live_entries_enabled=True,
            )
        assert "DO_NOT_RENDER_THIS" not in str(error.value)
