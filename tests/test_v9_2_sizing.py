"""Tests for v9.2 — conviction-tiered position sizing + concentration cap."""

from unittest.mock import MagicMock, patch

import pytest

_ENV = {
    "KRAKEN_API_KEY": "test",
    "KRAKEN_PRIVATE_KEY": "test",
    "XAI_API_KEY": "test",
}


def _make_settings(**overrides):
    from config import Settings

    env = {**_ENV, **{k.upper(): str(v) for k, v in overrides.items()}}
    with patch.dict("os.environ", env, clear=False):
        s = Settings()
    for k, v in overrides.items():
        if hasattr(s, k):
            setattr(s, k, v)
    return s


def _make_buy_signal(composite: float):
    from models import TradeSignal

    return TradeSignal(
        action="BUY",
        confidence=abs(composite),
        composite=composite,
        reasoning="test",
    )


# ---------------------------------------------------------------------------
# compute_conviction_multiplier
# ---------------------------------------------------------------------------


class TestConvictionMultiplier:
    @pytest.mark.parametrize(
        ("composite", "expected"),
        [
            (0.0, 0.0),
            (0.10, 0.0),
            (0.24, 0.0),
            (0.25, 0.6),
            (0.30, 0.6),
            (0.34, 0.6),
            (0.35, 1.0),
            (0.49, 1.0),
            (0.50, 1.4),
            (0.69, 1.4),
            (0.70, 2.0),
            (0.85, 2.0),
            (1.0, 2.0),
        ],
    )
    def test_tier_boundaries_default_settings(self, composite, expected):
        from risk_guardian import compute_conviction_multiplier

        settings = _make_settings()
        assert compute_conviction_multiplier(composite, settings) == expected

    def test_custom_tier_thresholds_respected(self):
        from risk_guardian import compute_conviction_multiplier

        settings = _make_settings()
        settings.sizing_tier_1_threshold = 0.10
        settings.sizing_tier_2_threshold = 0.20
        settings.sizing_tier_3_threshold = 0.40
        settings.sizing_tier_4_threshold = 0.80
        assert compute_conviction_multiplier(0.05, settings) == 0.0
        assert compute_conviction_multiplier(0.15, settings) == 0.6
        assert compute_conviction_multiplier(0.25, settings) == 1.0
        assert compute_conviction_multiplier(0.50, settings) == 1.4
        assert compute_conviction_multiplier(0.90, settings) == 2.0

    def test_custom_multipliers_respected(self):
        from risk_guardian import compute_conviction_multiplier

        settings = _make_settings()
        settings.sizing_tier_4_multiplier = 3.5
        assert compute_conviction_multiplier(0.90, settings) == 3.5


# ---------------------------------------------------------------------------
# RiskGuardian.check_buy with conviction sizing
# ---------------------------------------------------------------------------


class TestCheckBuyWithConvictionSizing:
    def test_weak_composite_sized_at_tier_1(self):
        """v9.6: sub-tier-1 entry composites size at tier-1 instead of dying.

        Regime modifiers can lower the strategy's entry threshold below the
        tier-1 floor; the guardian no longer overrules an approved entry.
        """
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        settings.max_position_size_usd = 50.0
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 1000.0
        guardian = RiskGuardian(settings, exchange)
        amount = guardian.check_buy(
            "BTC/USDT",
            price=50_000.0,
            trade_signal=_make_buy_signal(0.20),
            available_slots=1,
        )
        # base 1000 x 0.6 tier-1 = 600 → concentration 400 → hard cap 50
        assert amount == pytest.approx(50.0 / 50_000.0, rel=1e-6)

    def test_tier_2_normal_sizing_matches_legacy(self):
        """Composite 0.35-0.49 → 1.0x → allocation matches v9.1 behaviour."""
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        settings.max_position_size_usd = 50.0
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 200.0
        guardian = RiskGuardian(settings, exchange)

        amount = guardian.check_buy(
            "BTC/USDT",
            price=50_000.0,
            trade_signal=_make_buy_signal(0.40),
            available_slots=4,
        )
        # base = 200/4 = 50, multiplier=1.0, hard cap=50, concentration=0.40*200=80 → target=50
        assert amount == pytest.approx(50.0 / 50_000.0, rel=1e-6)

    def test_tier_3_amplifies_size(self):
        """Composite 0.50-0.69 → 1.4x."""
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        settings.max_position_size_usd = 200.0  # raise hard cap to test tier effect
        settings.max_position_pct_of_balance = 0.90  # relax concentration cap
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 400.0
        guardian = RiskGuardian(settings, exchange)

        amount = guardian.check_buy(
            "BTC/USDT",
            price=100.0,
            trade_signal=_make_buy_signal(0.55),
            available_slots=4,
        )
        # base = 400/4 = 100, multiplier=1.4, target=140 (below 200 hard cap, below 360 concentration)
        assert amount == pytest.approx(140.0 / 100.0, rel=1e-6)

    def test_tier_4_full_conviction(self):
        """Composite ≥0.70 → 2.0x."""
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        settings.max_position_size_usd = 500.0
        settings.max_position_pct_of_balance = 0.90
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 400.0
        guardian = RiskGuardian(settings, exchange)

        amount = guardian.check_buy(
            "BTC/USDT",
            price=100.0,
            trade_signal=_make_buy_signal(0.85),
            available_slots=4,
        )
        # base = 100, multiplier=2.0, target=200
        assert amount == pytest.approx(200.0 / 100.0, rel=1e-6)

    def test_concentration_cap_prevents_oversize(self):
        """40% cap must bind even when tier multiplier would allocate more."""
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        settings.max_position_size_usd = 500.0  # hard cap does not bind
        settings.max_position_pct_of_balance = 0.40  # default
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 100.0
        guardian = RiskGuardian(settings, exchange)

        amount = guardian.check_buy(
            "BTC/USDT",
            price=50.0,
            trade_signal=_make_buy_signal(0.85),
            available_slots=1,
        )
        # base = 100, multiplier=2.0 → tier_scaled=200, but concentration cap = 40
        # → target=40, amount = 40/50 = 0.8
        assert amount == pytest.approx(40.0 / 50.0, rel=1e-6)

    def test_hard_cap_still_binds(self):
        """Legacy max_position_size_usd still wins when lower than tier-scaled value."""
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        settings.max_position_size_usd = 50.0  # binds
        settings.max_position_pct_of_balance = 0.90
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 1_000.0
        guardian = RiskGuardian(settings, exchange)

        amount = guardian.check_buy(
            "BTC/USDT",
            price=100.0,
            trade_signal=_make_buy_signal(0.85),
            available_slots=1,
        )
        # base = 1000, multiplier=2.0 → 2000, concentration cap = 900
        # hard cap = 50 → target=50, amount = 0.5
        assert amount == pytest.approx(50.0 / 100.0, rel=1e-6)

    def test_disabled_flag_restores_flat_behaviour(self):
        """conviction_sizing_enabled=False → target = min(base, hard_cap) like v9.1."""
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        settings.conviction_sizing_enabled = False
        settings.max_position_size_usd = 50.0
        settings.max_position_pct_of_balance = 1.0  # neutralize concentration cap
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 200.0
        guardian = RiskGuardian(settings, exchange)

        # Even a tier-4 composite (0.85) should get the flat $50 allocation.
        amount = guardian.check_buy(
            "BTC/USDT",
            price=100.0,
            trade_signal=_make_buy_signal(0.85),
            available_slots=4,
        )
        # base = 50, multiplier=1.0 (disabled), target = min(50, 200, 50) = 50
        assert amount == pytest.approx(50.0 / 100.0, rel=1e-6)

    def test_disabled_flag_does_not_gate_weak_composite(self):
        """When disabled, a weak composite (below tier-1) should NOT be rejected by sizing."""
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        settings.conviction_sizing_enabled = False
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 100.0
        guardian = RiskGuardian(settings, exchange)

        amount = guardian.check_buy(
            "BTC/USDT",
            price=50.0,
            trade_signal=_make_buy_signal(0.10),
            available_slots=1,
        )
        # With conviction off, low composite should still size normally.
        assert amount > 0.0


# ---------------------------------------------------------------------------
# Config field sanity
# ---------------------------------------------------------------------------


class TestDcaSizingClamp:
    """DCA calls into the same position must not stack tier amplification."""

    def test_dca_multiplier_clamped_to_flat(self):
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        settings.max_position_size_usd = 500.0
        settings.max_position_pct_of_balance = 0.90
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 400.0
        guardian = RiskGuardian(settings, exchange)

        amount = guardian.check_buy(
            "BTC/USDT",
            price=100.0,
            trade_signal=_make_buy_signal(0.85),  # tier-4
            available_slots=4,
            is_dca=True,
        )
        # Without DCA clamp the target would be base x 2.0 = 200.
        # With clamp it should be base x 1.0 = 100 (below concentration and hard caps).
        assert amount == pytest.approx(100.0 / 100.0, rel=1e-6)

    def test_dca_still_respects_tier_1_gate(self):
        """Even with DCA, a composite below tier-1 threshold is rejected."""
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 1000.0
        guardian = RiskGuardian(settings, exchange)

        amount = guardian.check_buy(
            "BTC/USDT",
            price=100.0,
            trade_signal=_make_buy_signal(0.20),
            available_slots=1,
            is_dca=True,
        )
        assert amount == 0.0


class TestIntraCycleBudget:
    """Second entry in the same cycle must respect the budget ceiling."""

    def test_budget_caps_target_usd(self):
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        settings.max_position_size_usd = 500.0
        settings.max_position_pct_of_balance = 0.90
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 400.0
        guardian = RiskGuardian(settings, exchange)

        amount = guardian.check_buy(
            "BTC/USDT",
            price=100.0,
            trade_signal=_make_buy_signal(0.85),  # tier-4 → base 400 x 2.0 = 800
            available_slots=1,
            budget_usd=60.0,  # binds tighter than tier, concentration, hard cap
        )
        # Target should clamp to 60, amount = 60/100 = 0.6
        assert amount == pytest.approx(0.6, rel=1e-6)

    def test_zero_budget_rejects_buy(self):
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 400.0
        guardian = RiskGuardian(settings, exchange)

        amount = guardian.check_buy(
            "BTC/USDT",
            price=100.0,
            trade_signal=_make_buy_signal(0.85),
            available_slots=1,
            budget_usd=0.0,
        )
        # Min order is $10; $0 budget → below min → rejected.
        assert amount == 0.0

    def test_none_budget_does_not_cap(self):
        """Default budget_usd=None behaves like v9.2 without the fix."""
        from risk_guardian import RiskGuardian

        settings = _make_settings()
        settings.max_position_size_usd = 500.0
        settings.max_position_pct_of_balance = 0.90
        exchange = MagicMock()
        exchange.get_free_balance.return_value = 400.0
        guardian = RiskGuardian(settings, exchange)

        amount = guardian.check_buy(
            "BTC/USDT",
            price=100.0,
            trade_signal=_make_buy_signal(0.85),
            available_slots=4,
        )
        # Tier-4 x 100 = 200, below concentration cap (360) and hard cap (500)
        assert amount == pytest.approx(2.0, rel=1e-6)


class TestAscendingThresholdsGuard:
    """Guardrail so a misconfigured .env can't cause silent sizing errors."""

    def test_non_ascending_thresholds_raise(self):
        from risk_guardian import compute_conviction_multiplier

        settings = _make_settings()
        # Swap tier 1 and tier 2 — misconfig.
        settings.sizing_tier_1_threshold = 0.40
        settings.sizing_tier_2_threshold = 0.30

        with pytest.raises(AssertionError, match="ascending"):
            compute_conviction_multiplier(0.50, settings)

    def test_composite_out_of_range_raises(self):
        from risk_guardian import compute_conviction_multiplier

        settings = _make_settings()
        with pytest.raises(AssertionError, match="composite out of range"):
            compute_conviction_multiplier(1.5, settings)
        with pytest.raises(AssertionError, match="composite out of range"):
            compute_conviction_multiplier(-1.5, settings)


class TestSizingConfigDefaults:
    def test_defaults_are_applied(self):
        settings = _make_settings()
        assert settings.conviction_sizing_enabled is True
        assert settings.max_position_pct_of_balance == pytest.approx(0.40)
        assert settings.sizing_tier_1_threshold == pytest.approx(0.25)
        assert settings.sizing_tier_4_multiplier == pytest.approx(2.0)

    def test_hard_cap_ceiling_raised_to_500(self):
        """max_position_size_usd can now go up to $500 (was $100)."""
        settings = _make_settings()
        settings.max_position_size_usd = 350.0  # would fail the old le=100 constraint
        assert settings.max_position_size_usd == pytest.approx(350.0)

    def test_concentration_cap_rejects_out_of_range(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _make_settings(max_position_pct_of_balance=0.01)  # below ge=0.05
        with pytest.raises(ValidationError):
            _make_settings(max_position_pct_of_balance=1.5)  # above le=1.0
