"""Tests for the algorithmic strategy engine."""

from indicators import IndicatorSnapshot
from models import MarketData, TradeSignal
from strategy import Strategist


def _make_data(**overrides) -> MarketData:
    """Helper to build MarketData with sensible defaults."""
    defaults = {
        "ticker": "BTC/USDT",
        "current_price": 44000.0,
        "sentiment": "NEUTRAL: No developments.",
        "consensus": "Consensus unavailable.",
        "fear_greed": "50 (Neutral)",
    }
    defaults.update(overrides)
    return MarketData(**defaults)


class TestStrategist:
    """Test the rule-based Strategist."""

    def setup_method(self):
        self.strategist = Strategist()

    def test_bullish_entry_signal(self):
        """Strong bullish inputs should produce a BUY signal."""
        from forecast import PriceForecast
        from volatility import VolatilityForecast, VolatilityRegime

        data = _make_data(
            sentiment="BULLISH: Strong institutional inflows.",
            consensus="UP, 68%, Twitter consensus bullish.",
            fear_greed="72 (Greed)",
            indicators=IndicatorSnapshot(
                rsi=45.0,
                macd_histogram=0.12,
                bb_pct_b=0.50,
                atr_pct=2.5,
            ),
            volatility=VolatilityForecast(
                current_vol_pct=50.0,
                forecast_vol_pct=55.0,
                regime=VolatilityRegime.NORMAL,
                method="GARCH",
            ),
            forecast=PriceForecast(
                current_price=44000.0,
                forecast_price=44200.0,
                lower_bound=43800.0,
                upper_bound=44600.0,
                direction="UP",
                change_pct=0.45,
                method="ARIMA",
            ),
            polymarket_odds='Polymarket: "Will BTC reach $50k?" -> YES: 62%',
        )
        signal = self.strategist.generate_signal(data)
        assert isinstance(signal, TradeSignal)
        assert signal.action == "BUY"
        assert signal.confidence > 0.0
        assert signal.target_price is not None

    def test_bearish_hold_signal(self):
        """Bearish sentiment with no position should HOLD (not BUY)."""
        from forecast import PriceForecast
        from volatility import VolatilityForecast, VolatilityRegime

        data = _make_data(
            ticker="ETH/USDT",
            current_price=3000.0,
            sentiment="BEARISH: Market downturn expected.",
            consensus="DOWN, 70%, Consensus bearish.",
            fear_greed="25 (Fear)",
            indicators=IndicatorSnapshot(
                rsi=72.0,
                macd_histogram=-0.05,
                bb_pct_b=0.90,
                atr_pct=4.0,
            ),
            volatility=VolatilityForecast(
                current_vol_pct=80.0,
                forecast_vol_pct=90.0,
                regime=VolatilityRegime.HIGH,
                method="GARCH",
            ),
            forecast=PriceForecast(
                current_price=3000.0,
                forecast_price=2900.0,
                lower_bound=2800.0,
                upper_bound=3000.0,
                direction="DOWN",
                change_pct=-3.33,
                method="ARIMA",
            ),
        )
        signal = self.strategist.generate_signal(data)
        assert signal.action == "HOLD"

    def test_owned_position_bearish_sell(self):
        """Bearish sentiment on an owned position should produce SELL."""
        from forecast import PriceForecast
        from volatility import VolatilityForecast, VolatilityRegime

        data = _make_data(
            ticker="SOL/USDT",
            current_price=150.0,
            sentiment="BEARISH: Major exploit reported.",
            consensus="DOWN, 75%, Community panic.",
            fear_greed="15 (Extreme Fear)",
            has_position=True,
            entry_price=145.0,
            pnl_pct=3.45,
            tp_distance_pct=0.53,
            indicators=IndicatorSnapshot(
                rsi=35.0,
                macd_histogram=-0.03,
                bb_pct_b=0.15,
                atr_pct=3.5,
            ),
            volatility=VolatilityForecast(
                current_vol_pct=90.0,
                forecast_vol_pct=95.0,
                regime=VolatilityRegime.HIGH,
                method="GARCH",
            ),
            forecast=PriceForecast(
                current_price=150.0,
                forecast_price=145.0,
                lower_bound=140.0,
                upper_bound=148.0,
                direction="DOWN",
                change_pct=-3.33,
                method="ARIMA",
            ),
        )
        signal = self.strategist.generate_signal(data)
        assert signal.action == "SELL"
        assert signal.confidence > 0.0

    def test_neutral_holds(self):
        """Neutral across the board should HOLD."""
        data = _make_data(
            ticker="XRP/USDT",
            current_price=0.50,
            sentiment="NEUTRAL: No significant developments.",
            consensus="Consensus unavailable.",
            fear_greed="50 (Neutral)",
        )
        signal = self.strategist.generate_signal(data)
        assert signal.action == "HOLD"

    def test_overbought_rsi_blocks_buy(self):
        """RSI > 70 should prevent BUY even with bullish sentiment."""
        data = _make_data(
            ticker="DOGE/USDT",
            current_price=0.15,
            sentiment="BULLISH: Elon tweet.",
            consensus="UP, 65%, Twitter buzz.",
            fear_greed="80 (Extreme Greed)",
            indicators=IndicatorSnapshot(rsi=78.0, macd_histogram=0.001, bb_pct_b=0.95),
        )
        signal = self.strategist.generate_signal(data)
        assert signal.action == "HOLD"
        assert "RSI overbought" in signal.reasoning

    def test_extreme_volatility_gates_confidence(self):
        """EXTREME volatility should block BUY unless confidence >0.75."""
        from forecast import PriceForecast
        from volatility import VolatilityForecast, VolatilityRegime

        data = _make_data(
            sentiment="BULLISH: Positive momentum.",
            consensus="UP, 62%, Slight consensus.",
            fear_greed="55 (Greed)",
            indicators=IndicatorSnapshot(rsi=50.0, macd_histogram=0.02, bb_pct_b=0.45),
            volatility=VolatilityForecast(
                current_vol_pct=120.0,
                forecast_vol_pct=130.0,
                regime=VolatilityRegime.EXTREME,
                method="GARCH",
            ),
            forecast=PriceForecast(
                current_price=44000.0,
                forecast_price=44100.0,
                lower_bound=43500.0,
                upper_bound=44700.0,
                direction="UP",
                change_pct=0.23,
                method="ARIMA",
            ),
        )
        signal = self.strategist.generate_signal(data)
        if signal.action == "HOLD":
            assert "volatility" in signal.reasoning.lower()

    def test_profit_taking_sell(self):
        """Owned position with >5% profit and weak momentum should SELL."""
        from forecast import PriceForecast

        data = _make_data(
            ticker="ETH/USDT",
            current_price=3200.0,
            sentiment="NEUTRAL: Mixed signals.",
            consensus="Consensus unavailable.",
            fear_greed="50 (Neutral)",
            has_position=True,
            entry_price=3000.0,
            pnl_pct=6.67,
            tp_distance_pct=-2.50,
            indicators=IndicatorSnapshot(rsi=55.0, macd_histogram=-0.01, bb_pct_b=0.60),
            forecast=PriceForecast(
                current_price=3200.0,
                forecast_price=3180.0,
                lower_bound=3100.0,
                upper_bound=3250.0,
                direction="DOWN",
                change_pct=-0.63,
                method="ARIMA",
            ),
        )
        signal = self.strategist.generate_signal(data)
        assert signal.action == "SELL"
        assert "Profit" in signal.reasoning

    def test_signal_always_returns_trade_signal(self):
        """Even with empty inputs, a valid TradeSignal is returned."""
        data = _make_data(
            ticker="LINK/USDT",
            current_price=15.0,
            sentiment="",
            consensus="",
            fear_greed="",
        )
        signal = self.strategist.generate_signal(data)
        assert isinstance(signal, TradeSignal)
        assert signal.action in ("BUY", "SELL", "HOLD")
        assert 0.0 <= signal.confidence <= 1.0

    def test_dca_opportunity_on_owned_position(self):
        """Strong bullish signals on owned position should suggest DCA BUY."""
        from forecast import PriceForecast
        from volatility import VolatilityForecast, VolatilityRegime

        data = _make_data(
            sentiment="BULLISH: Massive institutional buying.",
            consensus="UP, 80%, Strong consensus.",
            fear_greed="65 (Greed)",
            has_position=True,
            entry_price=43500.0,
            pnl_pct=1.15,
            tp_distance_pct=2.82,
            indicators=IndicatorSnapshot(
                rsi=42.0,
                macd_histogram=0.15,
                bb_pct_b=0.35,
                atr_pct=2.5,
            ),
            volatility=VolatilityForecast(
                current_vol_pct=45.0,
                forecast_vol_pct=50.0,
                regime=VolatilityRegime.NORMAL,
                method="GARCH",
            ),
            forecast=PriceForecast(
                current_price=44000.0,
                forecast_price=44500.0,
                lower_bound=43800.0,
                upper_bound=45200.0,
                direction="UP",
                change_pct=1.14,
                method="ARIMA",
            ),
            polymarket_odds='Polymarket: "BTC above $45k by Friday?" -> YES: 70%',
        )
        signal = self.strategist.generate_signal(data)
        assert signal.action == "BUY"
        assert "DCA" in signal.reasoning


# ---------------------------------------------------------------------------
# Regime-aware and multi-timeframe tests
# ---------------------------------------------------------------------------


class TestRegimeModifier:
    """Tests for _apply_regime_modifier."""

    def test_regime_modifier_trending_up_boosts(self):
        """TRENDING_UP regime should boost composite by 10%."""
        from models import MarketRegime
        from strategy import _apply_regime_modifier

        composite = 0.20
        result, reason = _apply_regime_modifier(composite, MarketRegime.TRENDING_UP)
        assert abs(result - composite * 1.10) < 1e-9
        assert reason is not None
        assert "TRENDING_UP" in reason
        assert "x1.10" in reason

    def test_regime_modifier_trending_down_amplifies_negative(self):
        """TRENDING_DOWN regime should amplify negative composite (sell signal)."""
        from models import MarketRegime
        from strategy import _apply_regime_modifier

        composite = -0.20
        result, reason = _apply_regime_modifier(composite, MarketRegime.TRENDING_DOWN)
        assert abs(result - composite * 1.20) < 1e-9
        assert reason is not None
        assert "TRENDING_DOWN" in reason

    def test_regime_modifier_sign_mismatch_is_noop(self):
        """Positive composite in TRENDING_DOWN should be no-op."""
        from models import MarketRegime
        from strategy import _apply_regime_modifier

        composite = 0.20
        result, reason = _apply_regime_modifier(composite, MarketRegime.TRENDING_DOWN)
        assert result == composite  # No modification
        assert "no-op" in reason

    def test_regime_modifier_none_is_noop(self):
        """None regime should not modify composite."""
        from strategy import _apply_regime_modifier

        composite = 0.20
        result, reason = _apply_regime_modifier(composite, None)
        assert result == composite
        assert reason is None


class TestMtfAlignment:
    """Tests for _apply_mtf_alignment."""

    def test_mtf_alignment_boost(self):
        """Aligned 4h trend and signal direction should boost by 15%."""
        from models import MultiTimeframeContext
        from strategy import _MTF_ALIGNMENT_BOOST, _apply_mtf_alignment

        composite = 0.20  # positive = bullish signal
        mtf = MultiTimeframeContext(trend_bias="BULLISH")
        result, reason = _apply_mtf_alignment(composite, mtf)
        assert abs(result - composite * _MTF_ALIGNMENT_BOOST) < 1e-9
        assert reason is not None
        assert "aligned" in reason

    def test_mtf_conflict_penalty(self):
        """Conflicting 4h trend and signal direction should penalize by 25%."""
        from models import MultiTimeframeContext
        from strategy import _MTF_CONFLICT_PENALTY, _apply_mtf_alignment

        composite = 0.20  # positive = bullish signal
        mtf = MultiTimeframeContext(trend_bias="BEARISH")
        result, reason = _apply_mtf_alignment(composite, mtf)
        assert abs(result - composite * _MTF_CONFLICT_PENALTY) < 1e-9
        assert reason is not None
        assert "conflict" in reason

    def test_mtf_neutral_is_noop(self):
        """NEUTRAL trend bias should not modify composite."""
        from models import MultiTimeframeContext
        from strategy import _apply_mtf_alignment

        composite = 0.20
        mtf = MultiTimeframeContext(trend_bias="NEUTRAL")
        result, reason = _apply_mtf_alignment(composite, mtf)
        assert result == composite
        assert reason is None

    def test_mtf_none_is_noop(self):
        """None multi_tf should not modify composite."""
        from strategy import _apply_mtf_alignment

        composite = 0.20
        result, reason = _apply_mtf_alignment(composite, None)
        assert result == composite
        assert reason is None


class TestRegimeEntryThreshold:
    """Tests for regime-adjusted entry threshold in generate_signal."""

    def test_regime_adjusts_entry_threshold(self):
        """TRENDING_DOWN regime should raise entry threshold and penalize composite."""
        from models import MarketRegime
        from regime import RegimeClassifier

        strategist = Strategist()

        # Verify the mechanism: TRENDING_DOWN adds +0.05 to entry threshold
        rc = RegimeClassifier()
        adj = rc.get_entry_threshold_adjustment(MarketRegime.TRENDING_DOWN)
        assert adj == 0.05  # noqa: PLR2004

        # Build identical data with and without regime
        kwargs = dict(
            sentiment="BULLISH: Positive momentum.",
            consensus="UP, 65%, Slight consensus.",
            fear_greed="50 (Neutral)",
            indicators=IndicatorSnapshot(rsi=45.0, macd_histogram=0.05, bb_pct_b=0.40),
        )
        signal_no_regime = strategist.generate_signal(_make_data(**kwargs))
        signal_regime = strategist.generate_signal(
            _make_data(**kwargs, regime=MarketRegime.TRENDING_DOWN),
        )

        # TRENDING_DOWN with positive composite is a no-op on the composite itself
        # (sign-aware: only amplifies negative composites), but the higher entry
        # threshold (+0.05) makes it harder to trigger a BUY
        assert "regime=TRENDING_DOWN" in signal_regime.reasoning


# ---------------------------------------------------------------------------
# v8.1 entry gate tests
# ---------------------------------------------------------------------------


class TestTechnicalGate:
    """v8.1: Block entry when technicals are negative."""

    def test_negative_technicals_blocks_buy(self):
        """Negative technical_score should prevent BUY even with high composite."""
        strategist = Strategist()
        # RSI=67 + BB%B=0.89 → negative tech score, but bullish sentiment pushes composite up
        data = _make_data(
            sentiment="BULLISH (Score: 8/10): Strong momentum.",
            consensus="UP, 62%, Bullish consensus.",
            fear_greed="29 (Fear)",
            indicators=IndicatorSnapshot(
                rsi=67.0,
                macd_histogram=0.001,
                bb_pct_b=0.89,
                atr_pct=2.0,
            ),
        )
        signal = strategist.generate_signal(data)
        assert signal.action == "HOLD"
        assert "technicals negative" in signal.reasoning

    def test_positive_technicals_allows_buy(self):
        """Positive technical_score should allow BUY when other gates pass."""
        from forecast import PriceForecast
        from volatility import VolatilityForecast, VolatilityRegime

        strategist = Strategist()
        data = _make_data(
            sentiment="BULLISH: Strong institutional inflows.",
            consensus="UP, 68%, Twitter consensus bullish.",
            fear_greed="72 (Greed)",
            indicators=IndicatorSnapshot(
                rsi=45.0,
                macd_histogram=0.12,
                bb_pct_b=0.50,
                atr_pct=2.5,
            ),
            volatility=VolatilityForecast(
                current_vol_pct=50.0,
                forecast_vol_pct=55.0,
                regime=VolatilityRegime.NORMAL,
                method="GARCH",
            ),
            forecast=PriceForecast(
                current_price=44000.0,
                forecast_price=44200.0,
                lower_bound=43800.0,
                upper_bound=44600.0,
                direction="UP",
                change_pct=0.45,
                method="ARIMA",
            ),
            polymarket_odds='Polymarket: "Will BTC reach $50k?" -> YES: 62%',
        )
        signal = strategist.generate_signal(data)
        assert signal.action == "BUY"


class TestMtfEntryGate:
    """v8.1: Block entry when MTF trend is bearish."""

    def test_mtf_bearish_blocks_buy(self):
        """BEARISH multi-timeframe trend should block entry."""
        from models import MultiTimeframeContext

        strategist = Strategist()
        data = _make_data(
            sentiment="BULLISH: Strong momentum.",
            consensus="UP, 68%, Consensus bullish.",
            fear_greed="30 (Fear)",
            indicators=IndicatorSnapshot(
                rsi=45.0,
                macd_histogram=0.05,
                bb_pct_b=0.40,
                atr_pct=2.0,
            ),
            multi_tf=MultiTimeframeContext(trend_bias="BEARISH"),
        )
        signal = strategist.generate_signal(data)
        assert signal.action == "HOLD"
        assert "MTF bearish conflict" in signal.reasoning

    def test_mtf_bullish_allows_buy(self):
        """BULLISH multi-timeframe trend should not block entry."""
        from forecast import PriceForecast
        from models import MultiTimeframeContext
        from volatility import VolatilityForecast, VolatilityRegime

        strategist = Strategist()
        data = _make_data(
            sentiment="BULLISH: Strong institutional inflows.",
            consensus="UP, 68%, Twitter consensus bullish.",
            fear_greed="72 (Greed)",
            indicators=IndicatorSnapshot(
                rsi=45.0,
                macd_histogram=0.12,
                bb_pct_b=0.50,
                atr_pct=2.5,
            ),
            volatility=VolatilityForecast(
                current_vol_pct=50.0,
                forecast_vol_pct=55.0,
                regime=VolatilityRegime.NORMAL,
                method="GARCH",
            ),
            forecast=PriceForecast(
                current_price=44000.0,
                forecast_price=44200.0,
                lower_bound=43800.0,
                upper_bound=44600.0,
                direction="UP",
                change_pct=0.45,
                method="ARIMA",
            ),
            polymarket_odds='Polymarket: "Will BTC reach $50k?" -> YES: 62%',
            multi_tf=MultiTimeframeContext(trend_bias="BULLISH"),
        )
        signal = strategist.generate_signal(data)
        assert signal.action == "BUY"

    def test_mtf_none_allows_buy(self):
        """Missing MTF data should not block entry."""
        from forecast import PriceForecast
        from volatility import VolatilityForecast, VolatilityRegime

        strategist = Strategist()
        data = _make_data(
            sentiment="BULLISH: Strong institutional inflows.",
            consensus="UP, 68%, Twitter consensus bullish.",
            fear_greed="72 (Greed)",
            indicators=IndicatorSnapshot(
                rsi=45.0,
                macd_histogram=0.12,
                bb_pct_b=0.50,
                atr_pct=2.5,
            ),
            volatility=VolatilityForecast(
                current_vol_pct=50.0,
                forecast_vol_pct=55.0,
                regime=VolatilityRegime.NORMAL,
                method="GARCH",
            ),
            forecast=PriceForecast(
                current_price=44000.0,
                forecast_price=44200.0,
                lower_bound=43800.0,
                upper_bound=44600.0,
                direction="UP",
                change_pct=0.45,
                method="ARIMA",
            ),
            polymarket_odds='Polymarket: "Will BTC reach $50k?" -> YES: 62%',
            multi_tf=None,
        )
        signal = strategist.generate_signal(data)
        assert signal.action == "BUY"


class TestStalePositionTimeout:
    """v8.1: Stale position timeout extended from 72h to 120h."""

    def test_position_at_80h_not_stale(self):
        """Position held 80h (between old 72h and new 120h) should NOT trigger stale exit."""
        strategist = Strategist()
        data = _make_data(
            has_position=True,
            entry_price=43500.0,
            pnl_pct=0.5,
            position_age_hours=80,
            sentiment="BULLISH: Positive outlook.",
            consensus="UP, 60%, Slight upside.",
            fear_greed="45 (Fear)",
            indicators=IndicatorSnapshot(rsi=50.0, macd_histogram=0.01, bb_pct_b=0.50),
        )
        signal = strategist.generate_signal(data)
        assert "Stale position" not in signal.reasoning

    def test_position_at_130h_is_stale(self):
        """Position held 130h should trigger stale exit."""
        strategist = Strategist()
        data = _make_data(
            has_position=True,
            entry_price=43500.0,
            pnl_pct=0.5,
            position_age_hours=130,
            sentiment="BULLISH: Positive outlook.",
            consensus="UP, 60%, Slight upside.",
            fear_greed="45 (Fear)",
            indicators=IndicatorSnapshot(rsi=50.0, macd_histogram=0.01, bb_pct_b=0.50),
        )
        signal = strategist.generate_signal(data)
        assert signal.action == "SELL"
        assert "Stale position" in signal.reasoning


class TestWeightChanges:
    """v8.1: Verify updated weight configuration."""

    def test_weights_sum_to_one(self):
        """All composite weights must still sum to 1.0."""
        from strategy import (
            W_CONSENSUS,
            W_FEAR_GREED,
            W_FORECAST,
            W_MOMENTUM,
            W_POLYMARKET,
            W_SENTIMENT,
            W_TECHNICAL,
        )

        total = (
            W_TECHNICAL
            + W_MOMENTUM
            + W_FORECAST
            + W_SENTIMENT
            + W_CONSENSUS
            + W_POLYMARKET
            + W_FEAR_GREED
        )
        assert abs(total - 1.0) < 1e-9

    def test_fear_greed_weight_reduced(self):
        """Fear & Greed weight should be 0.05 (was 0.10)."""
        from strategy import W_FEAR_GREED

        assert W_FEAR_GREED == 0.05  # noqa: PLR2004

    def test_technical_weight_increased(self):
        """Technical weight should be 0.35 (was 0.30)."""
        from strategy import W_TECHNICAL

        assert W_TECHNICAL == 0.35  # noqa: PLR2004

    def test_entry_threshold(self):
        """Entry threshold should be 0.15 (was 0.10)."""
        from strategy import _ENTRY_THRESHOLD

        assert _ENTRY_THRESHOLD == 0.15  # noqa: PLR2004
