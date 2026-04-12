"""Tests for market regime detection."""

import pytest

from models import MarketRegime, StopMode
from regime import RegimeClassifier


class TestRegimeClassify:
    """Test RegimeClassifier.classify() for each regime and edge case."""

    def setup_method(self):
        self.clf = RegimeClassifier()

    # --- TRENDING_UP ---

    def test_trending_up(self):
        regime = self.clf.classify(
            adx=30,
            ema_bullish_cross=True,
            roc=2.0,
            bb_pct_b=0.7,
            obv_slope=0.1,
        )
        assert regime == MarketRegime.TRENDING_UP

    def test_trending_up_high_adx(self):
        regime = self.clf.classify(
            adx=60,
            ema_bullish_cross=True,
            roc=0.01,
            bb_pct_b=0.5,
            obv_slope=0.0,
        )
        assert regime == MarketRegime.TRENDING_UP

    def test_trending_up_boundary_adx(self):
        """ADX exactly at 25.01 should still qualify as strong trend."""
        regime = self.clf.classify(
            adx=25.01,
            ema_bullish_cross=True,
            roc=1.0,
            bb_pct_b=0.5,
            obv_slope=0.0,
        )
        assert regime == MarketRegime.TRENDING_UP

    # --- TRENDING_DOWN ---

    def test_trending_down(self):
        regime = self.clf.classify(
            adx=30,
            ema_bullish_cross=False,
            roc=-2.0,
            bb_pct_b=0.3,
            obv_slope=-0.1,
        )
        assert regime == MarketRegime.TRENDING_DOWN

    def test_trending_down_extreme(self):
        regime = self.clf.classify(
            adx=50,
            ema_bullish_cross=False,
            roc=-10.0,
            bb_pct_b=0.1,
            obv_slope=-0.2,
        )
        assert regime == MarketRegime.TRENDING_DOWN

    # --- RANGING ---

    def test_ranging_weak_adx(self):
        regime = self.clf.classify(
            adx=15,
            ema_bullish_cross=True,
            roc=1.0,
            bb_pct_b=0.5,
            obv_slope=0.0,
        )
        assert regime == MarketRegime.RANGING

    def test_ranging_adx_exactly_25(self):
        """ADX at exactly 25 should NOT qualify as strong trend (> 25 required)."""
        regime = self.clf.classify(
            adx=25,
            ema_bullish_cross=True,
            roc=1.0,
            bb_pct_b=0.5,
            obv_slope=0.0,
        )
        assert regime == MarketRegime.RANGING

    def test_ranging_mixed_signals(self):
        """Bullish EMA cross but negative ROC => no clear direction."""
        regime = self.clf.classify(
            adx=30,
            ema_bullish_cross=True,
            roc=-1.0,
            bb_pct_b=0.5,
            obv_slope=0.0,
        )
        assert regime == MarketRegime.RANGING

    def test_ranging_bearish_cross_positive_roc(self):
        """Bearish EMA cross but positive ROC => no clear direction."""
        regime = self.clf.classify(
            adx=30,
            ema_bullish_cross=False,
            roc=1.0,
            bb_pct_b=0.5,
            obv_slope=0.0,
        )
        assert regime == MarketRegime.RANGING

    # --- BREAKOUT ---

    def test_breakout_above_bb(self):
        regime = self.clf.classify(
            adx=20,
            ema_bullish_cross=True,
            roc=3.0,
            bb_pct_b=1.2,
            obv_slope=0.5,
        )
        assert regime == MarketRegime.BREAKOUT

    def test_breakout_below_bb(self):
        regime = self.clf.classify(
            adx=20,
            ema_bullish_cross=False,
            roc=-3.0,
            bb_pct_b=-0.1,
            obv_slope=-0.4,
        )
        assert regime == MarketRegime.BREAKOUT

    def test_breakout_requires_volume(self):
        """BB outside bands but low OBV slope = NOT breakout."""
        regime = self.clf.classify(
            adx=15,
            ema_bullish_cross=True,
            roc=1.0,
            bb_pct_b=1.2,
            obv_slope=0.1,
        )
        assert regime == MarketRegime.RANGING

    def test_breakout_obv_exactly_at_threshold(self):
        """OBV slope exactly 0.3 should NOT trigger breakout (> 0.3 required)."""
        regime = self.clf.classify(
            adx=15,
            ema_bullish_cross=None,
            roc=None,
            bb_pct_b=1.5,
            obv_slope=0.3,
        )
        assert regime == MarketRegime.RANGING

    def test_breakout_priority_over_trending(self):
        """Breakout should win even when trending conditions are also met."""
        regime = self.clf.classify(
            adx=35,
            ema_bullish_cross=True,
            roc=5.0,
            bb_pct_b=1.3,
            obv_slope=0.8,
        )
        assert regime == MarketRegime.BREAKOUT

    def test_breakout_bb_exactly_zero(self):
        """BB %B at exactly 0.0 qualifies as outside bands (< 0.0 not satisfied)."""
        regime = self.clf.classify(
            adx=15,
            ema_bullish_cross=None,
            roc=None,
            bb_pct_b=0.0,
            obv_slope=0.5,
        )
        # bb_pct_b == 0.0 does NOT satisfy < 0.0 or > 1.0
        assert regime == MarketRegime.RANGING

    def test_breakout_bb_exactly_one(self):
        """BB %B at exactly 1.0 does NOT satisfy > 1.0."""
        regime = self.clf.classify(
            adx=15,
            ema_bullish_cross=None,
            roc=None,
            bb_pct_b=1.0,
            obv_slope=0.5,
        )
        assert regime == MarketRegime.RANGING

    # --- None / missing data edge cases ---

    def test_all_none_defaults_to_ranging(self):
        regime = self.clf.classify(
            adx=None,
            ema_bullish_cross=None,
            roc=None,
            bb_pct_b=None,
            obv_slope=None,
        )
        assert regime == MarketRegime.RANGING

    def test_only_adx_present(self):
        regime = self.clf.classify(
            adx=40,
            ema_bullish_cross=None,
            roc=None,
            bb_pct_b=None,
            obv_slope=None,
        )
        assert regime == MarketRegime.RANGING

    def test_adx_none_with_other_values(self):
        regime = self.clf.classify(
            adx=None,
            ema_bullish_cross=True,
            roc=5.0,
            bb_pct_b=0.5,
            obv_slope=0.2,
        )
        assert regime == MarketRegime.RANGING

    def test_bb_pct_b_none_no_breakout(self):
        """Can't detect breakout without BB %B."""
        regime = self.clf.classify(
            adx=15,
            ema_bullish_cross=None,
            roc=None,
            bb_pct_b=None,
            obv_slope=0.8,
        )
        assert regime == MarketRegime.RANGING

    def test_obv_slope_none_no_breakout(self):
        """Can't detect breakout without OBV slope."""
        regime = self.clf.classify(
            adx=15,
            ema_bullish_cross=None,
            roc=None,
            bb_pct_b=1.5,
            obv_slope=None,
        )
        assert regime == MarketRegime.RANGING

    def test_roc_zero_with_bullish_cross(self):
        """ROC exactly 0 is not > 0, so no TRENDING_UP."""
        regime = self.clf.classify(
            adx=30,
            ema_bullish_cross=True,
            roc=0.0,
            bb_pct_b=0.5,
            obv_slope=0.0,
        )
        assert regime == MarketRegime.RANGING

    def test_roc_zero_with_bearish_cross(self):
        """ROC exactly 0 is not < 0, so no TRENDING_DOWN."""
        regime = self.clf.classify(
            adx=30,
            ema_bullish_cross=False,
            roc=0.0,
            bb_pct_b=0.5,
            obv_slope=0.0,
        )
        assert regime == MarketRegime.RANGING


class TestEntryThresholdAdjustment:
    """Test get_entry_threshold_adjustment returns correct values."""

    def test_trending_up_easier_entry(self):
        adj = RegimeClassifier.get_entry_threshold_adjustment(MarketRegime.TRENDING_UP)
        assert adj == pytest.approx(-0.02)

    def test_trending_down_harder_entry(self):
        adj = RegimeClassifier.get_entry_threshold_adjustment(MarketRegime.TRENDING_DOWN)
        assert adj == pytest.approx(0.05)
        assert adj > 0

    def test_ranging_easier_entry(self):
        adj = RegimeClassifier.get_entry_threshold_adjustment(MarketRegime.RANGING)
        assert adj == pytest.approx(-0.02)
        assert adj < 0

    def test_breakout_harder_entry(self):
        adj = RegimeClassifier.get_entry_threshold_adjustment(MarketRegime.BREAKOUT)
        assert adj == pytest.approx(0.03)
        assert adj > 0

    def test_all_regimes_covered(self):
        """Every MarketRegime member has an adjustment."""
        for regime in MarketRegime:
            adj = RegimeClassifier.get_entry_threshold_adjustment(regime)
            assert isinstance(adj, float)


class TestStopModeSuggestion:
    """Test get_stop_mode_for_regime returns correct StopMode values."""

    def test_trending_up_trailing(self):
        assert RegimeClassifier.get_stop_mode_for_regime(MarketRegime.TRENDING_UP) == "TRAILING"

    def test_trending_down_trailing_breakeven(self):
        assert (
            RegimeClassifier.get_stop_mode_for_regime(MarketRegime.TRENDING_DOWN)
            == "TRAILING_BREAKEVEN"
        )

    def test_ranging_fixed(self):
        assert RegimeClassifier.get_stop_mode_for_regime(MarketRegime.RANGING) == "FIXED"

    def test_breakout_trailing_breakeven(self):
        assert (
            RegimeClassifier.get_stop_mode_for_regime(MarketRegime.BREAKOUT) == "TRAILING_BREAKEVEN"
        )

    def test_all_regimes_return_valid_stop_mode(self):
        """Every regime maps to a valid StopMode value."""
        valid_values = {m.value for m in StopMode}
        for regime in MarketRegime:
            mode = RegimeClassifier.get_stop_mode_for_regime(regime)
            assert mode in valid_values, f"{regime} returned invalid stop mode: {mode}"
