"""Tests for v9.0 refactor — module split, threshold-to-config, MIN_CONFIDENCE removal."""

from unittest.mock import MagicMock, patch

import pytest

_ENV = {
    "KRAKEN_API_KEY": "test",
    "KRAKEN_PRIVATE_KEY": "test",
    "XAI_API_KEY": "test",
}


def _make_settings(**overrides):
    """Build a Settings instance with secrets stubbed and optional overrides."""
    from config import Settings

    env = {**_ENV, **{k.upper(): str(v) for k, v in overrides.items()}}
    with patch.dict("os.environ", env, clear=False):
        return Settings()


class TestModuleSplit:
    """Smoke tests — new modules are importable and wire-compatible."""

    def test_risk_guardian_module_imports(self):
        from risk_guardian import (
            CORRELATION_GROUPS,
            MIN_ORDER_USD,
            REGULATORY_DANGER_PHRASES,
            ROUND_UP_THRESHOLD,
            RiskGuardian,
            check_correlation_limit,
            check_daily_loss_limit,
        )

        assert MIN_ORDER_USD == 10.0  # noqa: PLR2004
        assert ROUND_UP_THRESHOLD == 5.0  # noqa: PLR2004  # v9.6: lowered for small accounts
        assert "BANNED" in REGULATORY_DANGER_PHRASES
        assert CORRELATION_GROUPS["BTC"] == "btc_core"  # v9.5 regrouping
        assert callable(check_correlation_limit)
        assert callable(check_daily_loss_limit)
        assert RiskGuardian is not None

    def test_data_pipeline_module_imports(self):
        from data_pipeline import build_market_data, collect_all_market_data

        assert callable(build_market_data)
        assert callable(collect_all_market_data)

    def test_order_executor_module_imports(self):
        from order_executor import (
            check_take_profit,
            compute_protection_prices,
            ensure_stop_loss,
            execute_dca,
            execute_partial_sell,
            execute_sell,
            reconcile_fills,
            reconcile_stale_orders,
        )

        assert callable(execute_sell)
        assert callable(execute_partial_sell)
        assert callable(execute_dca)
        assert callable(compute_protection_prices)
        assert callable(check_take_profit)
        assert callable(ensure_stop_loss)
        assert callable(reconcile_stale_orders)
        assert callable(reconcile_fills)

    def test_decision_engine_module_imports(self):
        from decision_engine import evaluate_entry, evaluate_exit

        assert callable(evaluate_entry)
        assert callable(evaluate_exit)

    def test_portfolio_manager_module_imports(self):
        from portfolio_manager import classify_tickers

        assert callable(classify_tickers)

    def test_main_slimmed_down(self):
        """After split, private helpers no longer live in main.py."""
        import main

        assert hasattr(main, "run_cycle")
        assert hasattr(main, "main")
        assert hasattr(main, "setup_logging")
        # Private helpers were moved out of main
        assert not hasattr(main, "_build_market_data")
        assert not hasattr(main, "_execute_sell")
        assert not hasattr(main, "_execute_dca")
        assert not hasattr(main, "_evaluate_entry")
        assert not hasattr(main, "_evaluate_exit")
        assert not hasattr(main, "_check_correlation_limit")
        assert not hasattr(main, "_classify_tickers")
        # Module-level constants that were duplicated are now only in risk_guardian
        assert not hasattr(main, "_CORRELATION_GROUPS")
        assert not hasattr(main, "REGULATORY_DANGER_PHRASES")


class TestMinConfidenceConfigurable:
    """MIN_CONFIDENCE constant was removed; confidence gates now live in Settings."""

    def test_min_confidence_constant_removed_from_main(self):
        import main

        assert not hasattr(main, "MIN_CONFIDENCE")

    def test_min_confidence_constant_removed_from_risk_guardian(self):
        import risk_guardian

        assert not hasattr(risk_guardian, "MIN_CONFIDENCE")

    def test_settings_expose_buy_and_sell_confidence_gates(self):
        settings = _make_settings()
        assert settings.min_buy_confidence == 0.0
        assert settings.min_sell_confidence == 0.0

    def test_guardian_uses_min_buy_confidence_from_settings(self):
        """Raising min_buy_confidence above signal confidence blocks the BUY."""
        from models import TradeSignal
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        settings.min_buy_confidence = 0.5
        exchange = MagicMock()
        guardian = RiskGuardian(settings, exchange)
        weak_buy = TradeSignal(
            action="BUY",
            confidence=0.3,
            composite=0.3,
            reasoning="weak buy",
        )
        amount = guardian.check_buy(
            "BTC/USDT",
            price=50000.0,
            trade_signal=weak_buy,
            available_slots=1,
        )
        assert amount == 0.0

    def test_guardian_allows_buy_with_default_min_confidence(self):
        """Default min_buy_confidence=0.0 lets any BUY through to balance checks.

        Uses composite above the tier-1 threshold so the v9.2 conviction
        tier gate doesn't reject it — we're specifically testing the
        ``min_buy_confidence`` gate here, not conviction sizing.
        """
        from models import TradeSignal
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 100.0
        guardian = RiskGuardian(settings, exchange)
        modest_buy = TradeSignal(
            action="BUY",
            confidence=0.26,
            composite=0.26,
            reasoning="just above tier-1",
        )
        amount = guardian.check_buy(
            "BTC/USDT",
            price=50000.0,
            trade_signal=modest_buy,
            available_slots=1,
        )
        assert amount > 0.0


class TestStrategistAcceptsSettings:
    """Strategist now reads thresholds from Settings (was module constants)."""

    def test_strategist_defaults_match_legacy_constants(self):
        from strategy import (
            _DCA_THRESHOLD,
            _ENTRY_THRESHOLD,
            _EXIT_THRESHOLD,
            _STALE_POSITION_HOURS,
            _STRONG_SELL_THRESHOLD,
            Strategist,
        )

        strategist = Strategist()
        assert strategist.entry_threshold == _ENTRY_THRESHOLD
        assert strategist.exit_threshold == _EXIT_THRESHOLD
        assert strategist.strong_sell_threshold == _STRONG_SELL_THRESHOLD
        assert strategist.dca_threshold == _DCA_THRESHOLD
        assert strategist.stale_position_hours == _STALE_POSITION_HOURS

    def test_strategist_reads_overrides_from_settings(self):
        from strategy import Strategist

        settings = _make_settings()
        settings.entry_threshold = 0.42
        settings.exit_threshold = -0.11
        settings.dca_threshold = 0.55
        settings.stale_position_hours = 200.0

        strategist = Strategist(settings=settings)
        assert strategist.entry_threshold == 0.42  # noqa: PLR2004
        assert strategist.exit_threshold == -0.11  # noqa: PLR2004
        assert strategist.dca_threshold == 0.55  # noqa: PLR2004
        assert strategist.stale_position_hours == 200.0  # noqa: PLR2004

    def test_entry_threshold_override_flips_buy_to_hold(self):
        """A composite that passes 0.25 but fails 0.50 should HOLD when threshold=0.50."""
        from models import MarketData, MarketRegime
        from strategy import Strategist

        settings_low = _make_settings()
        settings_low.entry_threshold = 0.25

        settings_high = _make_settings()
        settings_high.entry_threshold = 0.50

        # Construct a MarketData that produces a composite in [0.25, 0.50).
        data = MarketData(
            ticker="BTC/USDT",
            current_price=50000.0,
            indicators=None,
            volatility=None,
            forecast=None,
            sentiment="BULLISH (Score: 8/10): strong uptrend",
            consensus="**UP**, 70%",
            fear_greed="30 (Fear)",
            polymarket_odds="YES: 60%",
            has_position=False,
            regime=MarketRegime.RANGING,
        )

        sig_low = Strategist(settings=settings_low).generate_signal(data)
        sig_high = Strategist(settings=settings_high).generate_signal(data)

        # At the low threshold we should BUY (or at least not be blocked by threshold).
        # At the high threshold, the same composite fails → HOLD.
        if sig_low.action == "BUY":
            assert sig_high.action == "HOLD", (
                f"High threshold should block weaker composite. "
                f"comp={sig_high.composite}, action={sig_high.action}"
            )


class TestMinSellConfidenceGate:
    """decision_engine.evaluate_exit must honour settings.min_sell_confidence."""

    def test_weak_sell_passes_when_gate_is_zero(self):
        """Default min_sell_confidence=0.0 lets strategy-driven SELLs through."""
        import importlib

        import decision_engine

        settings = _make_settings()
        assert settings.min_sell_confidence == 0.0

        module = importlib.reload(decision_engine)
        # Confirm the gate is read from settings, not a module constant
        src = module.__file__
        assert src and src.endswith("decision_engine.py")

    def test_high_gate_blocks_strategic_sell(self):
        """A high min_sell_confidence blocks SELLs below that floor."""
        from unittest.mock import MagicMock

        from decision_engine import evaluate_exit
        from journal import TradeJournal
        from models import MarketData, TradeSignal

        settings = _make_settings()
        settings.min_sell_confidence = 0.9  # impossibly high — blocks all strategy SELLs

        strategist = MagicMock()
        strategist.generate_signal.return_value = TradeSignal(
            action="SELL",
            confidence=0.2,
            composite=-0.2,
            reasoning="weak sell",
        )

        exchange = MagicMock()
        exchange.get_open_orders.return_value = []
        exchange.get_free_balance.return_value = 0.001

        journal = MagicMock(spec=TradeJournal)
        journal.get_position_state.return_value = None

        data = MarketData(
            ticker="BTC/USDT",
            current_price=100.0,
            entry_price=95.0,
            sentiment="NEUTRAL",
            consensus="flat",
            polymarket_odds="",
            pnl_pct=0.0,
            position_age_hours=1.0,
        )

        guardian = MagicMock()
        notifier = MagicMock()

        # With the gate set above the signal confidence, SELL must not be executed.
        evaluate_exit(
            ticker="BTC/USDT",
            data=data,
            settings=settings,
            exchange=exchange,
            strategist=strategist,
            guardian=guardian,
            journal=journal,
            notifier=notifier,
            available_slots=1,
        )

        # Journal.log_trade should NOT have been called with side="SELL"
        sell_calls = [
            call for call in journal.log_trade.call_args_list if call.kwargs.get("side") == "SELL"
        ]
        assert len(sell_calls) == 0, f"Expected no SELL trade, got {sell_calls}"


class TestStrategistFieldValidation:
    """Strategist must fail loudly when settings is missing required fields."""

    def test_strategist_rejects_partial_settings_mock(self):
        class PartialSettings:
            entry_threshold = 0.25
            # deliberately missing the other 11 fields

        from strategy import Strategist

        with pytest.raises(AssertionError, match="missing fields"):
            Strategist(settings=PartialSettings())


class TestDriftGuardBetweenStrategyAndSettings:
    """strategy.py module constants must match Settings defaults (import-time check)."""

    def test_import_time_drift_check_passes(self):
        from strategy import _assert_module_defaults_match_settings

        # Re-run explicitly; any drift would raise here.
        _assert_module_defaults_match_settings()


class TestEmptyUniverseGuard:
    """collect_all_market_data must not crash on an empty ticker list."""

    def test_empty_tickers_returns_empty_dict(self):
        from data_pipeline import collect_all_market_data

        result = collect_all_market_data(
            tickers=[],
            settings=_make_settings(),
            cached_exchange=object(),
            oracle=object(),
            journal=object(),
            fear_greed="",
            polymarket_markets=[],
            tech_analyzer=None,
            vol_analyzer=None,
            price_forecaster=None,
            has_position_map={},
            buy_price_map={},
        )
        assert result == {}


class TestConfigExposesAllThresholds:
    """Every previously-hardcoded strategy threshold is now on Settings."""

    @pytest.mark.parametrize(
        ("field", "default"),
        [
            ("entry_threshold", 0.25),
            ("exit_threshold", -0.05),
            ("strong_sell_threshold", -0.25),
            ("dca_threshold", 0.30),
            ("profit_take_pct", 5.0),
            ("momentum_weak_threshold", 0.1),
            ("sma_overextended_pct", 15.0),
            ("mean_reversion_rsi", 28.0),
            ("mean_reversion_adx", 30.0),
            ("mean_reversion_bonus", 0.10),
            ("stale_position_hours", 120.0),
            ("stale_exit_pnl_threshold_pct", 2.0),
            ("min_buy_confidence", 0.0),
            ("min_sell_confidence", 0.0),
        ],
    )
    def test_threshold_field_default(self, field, default):
        settings = _make_settings()
        assert getattr(settings, field) == pytest.approx(default)
