"""Tests for v9.3 — walk-forward optimizer, live-strategy backtest, weight overrides."""

import pytest

# ---------------------------------------------------------------------------
# Strategist weights_override
# ---------------------------------------------------------------------------


class TestStrategistWeightsOverride:
    def test_default_weights_match_module_constants(self):
        from strategy import (
            W_CONSENSUS,
            W_FEAR_GREED,
            W_FORECAST,
            W_FUNDING,
            W_MOMENTUM,
            W_POLYMARKET,
            W_SENTIMENT,
            W_TECHNICAL,
            Strategist,
        )

        strategist = Strategist()
        assert strategist._weights == {  # noqa: SLF001
            "technical": W_TECHNICAL,
            "momentum": W_MOMENTUM,
            "forecast": W_FORECAST,
            "sentiment": W_SENTIMENT,
            "consensus": W_CONSENSUS,
            "polymarket": W_POLYMARKET,
            "fear_greed": W_FEAR_GREED,
            "funding": W_FUNDING,
        }

    def test_override_applied(self):
        from strategy import Strategist

        custom = {
            "technical": 0.38,
            "momentum": 0.10,
            "forecast": 0.05,
            "sentiment": 0.15,
            "consensus": 0.10,
            "polymarket": 0.15,
            "fear_greed": 0.05,
            "funding": 0.02,
        }
        strategist = Strategist(weights_override=custom)
        assert strategist._weights == custom  # noqa: SLF001

    def test_override_must_sum_to_one(self):
        from strategy import Strategist

        bad = {
            "technical": 0.5,
            "momentum": 0.5,
            "forecast": 0.5,
            "sentiment": 0.0,
            "consensus": 0.0,
            "polymarket": 0.0,
            "fear_greed": 0.0,
            "funding": 0.0,
        }
        with pytest.raises(AssertionError, match="sum to 1.0"):
            Strategist(weights_override=bad)

    def test_override_requires_all_keys(self):
        from strategy import Strategist

        incomplete = {
            "technical": 0.35,
            "momentum": 0.10,
            # missing the other 5
        }
        with pytest.raises(AssertionError, match="missing keys"):
            Strategist(weights_override=incomplete)

    def test_override_changes_composite(self):
        """Heavily weighting technical should flip a BUY on strong tech signals."""
        from indicators import IndicatorSnapshot
        from models import MarketData, MarketRegime
        from strategy import Strategist

        # Strong bullish technicals (RSI 25 oversold, MACD positive, BB pct_b 0.0)
        indicators = IndicatorSnapshot(
            rsi=25.0,
            macd_histogram=0.01,
            bb_pct_b=0.05,
            atr=1.0,
            atr_pct=1.0,
            sma_7=100.0,
            sma_7_deviation_pct=0.0,
            adx=35.0,
            roc=2.0,
            ema_bullish_cross=True,
            obv_slope=0.5,
        )
        data = MarketData(
            ticker="BTC/USDT",
            current_price=100.0,
            indicators=indicators,
            volatility=None,
            forecast=None,
            sentiment="BULLISH (Score: 8/10)",
            consensus="UP, 70%",
            fear_greed="30 (Fear)",
            polymarket_odds="YES: 65%",
            has_position=False,
            regime=MarketRegime.RANGING,
        )

        default_signal = Strategist().generate_signal(data)

        heavy_tech = Strategist(
            weights_override={
                "technical": 0.70,
                "momentum": 0.05,
                "forecast": 0.05,
                "sentiment": 0.05,
                "consensus": 0.05,
                "polymarket": 0.04,
                "fear_greed": 0.04,
                "funding": 0.02,
            },
        )
        heavy_tech_signal = heavy_tech.generate_signal(data)

        # With the heavy technical weight the composite should differ from default.
        assert heavy_tech_signal.composite != pytest.approx(default_signal.composite)


# ---------------------------------------------------------------------------
# BacktestResult includes Sortino / Calmar / Profit Factor
# ---------------------------------------------------------------------------


class TestBacktestResultExtraMetrics:
    def test_fields_default_to_zero(self):
        from backtest import BacktestResult

        result = BacktestResult(
            total_return_pct=5.0,
            sharpe_ratio=1.2,
            max_drawdown_pct=-3.0,
            win_rate_pct=55.0,
            buy_and_hold_return_pct=2.0,
            num_trades=10,
        )
        assert result.sortino_ratio == 0.0
        assert result.calmar_ratio == 0.0
        assert result.profit_factor == 0.0

    def test_summary_includes_sortino_and_calmar(self):
        from backtest import BacktestResult

        result = BacktestResult(
            total_return_pct=5.0,
            sharpe_ratio=1.2,
            max_drawdown_pct=-3.0,
            win_rate_pct=55.0,
            buy_and_hold_return_pct=2.0,
            num_trades=10,
            sortino_ratio=1.8,
            calmar_ratio=0.5,
        )
        summary = result.summary()
        assert "Sortino: 1.80" in summary
        assert "Calmar: 0.50" in summary


# ---------------------------------------------------------------------------
# CompositeStrategy factory hook
# ---------------------------------------------------------------------------


class TestCompositeStrategyFactory:
    def test_factory_attribute_defaults_to_none(self):
        from backtest import CompositeStrategy

        assert CompositeStrategy._strategist_factory is None  # noqa: SLF001

    def test_factory_can_be_set_and_cleared(self):
        from backtest import CompositeStrategy
        from strategy import Strategist

        def factory():
            return Strategist()

        previous = CompositeStrategy._strategist_factory  # noqa: SLF001
        try:
            CompositeStrategy._strategist_factory = factory  # noqa: SLF001
            assert CompositeStrategy._strategist_factory is factory  # noqa: SLF001
        finally:
            CompositeStrategy._strategist_factory = previous  # noqa: SLF001


# ---------------------------------------------------------------------------
# WalkForwardOptimizer — window splitting
# ---------------------------------------------------------------------------


class TestWalkForwardWindows:
    def test_iter_windows_exact_fit(self):
        from walk_forward import WalkForwardOptimizer

        candles = [[i] for i in range(240)]  # 60 train + 20 test + step 20
        wfo = WalkForwardOptimizer(candles, train_size=60, test_size=20, step=20)
        windows = wfo.iter_windows()

        # Expected: windows at (0..60, 60..80), (20..80, 80..100), (40..100, 100..120), ...
        assert len(windows) > 1
        first = windows[0]
        assert first == (0, 60, 60, 80)
        # Step of 20 so second window advances by 20.
        second = windows[1]
        assert second == (20, 80, 80, 100)

    def test_iter_windows_stops_when_test_exceeds_series(self):
        from walk_forward import WalkForwardOptimizer

        candles = [[i] for i in range(100)]
        wfo = WalkForwardOptimizer(candles, train_size=60, test_size=20, step=20)
        windows = wfo.iter_windows()
        # 60 train + 20 test = 80 for first; step 20 gives next at [20..80, 80..100] = 100 end.
        # Step 40 would need 100..120 which exceeds, so stops.
        assert all(test_end <= len(candles) for (_, _, _, test_end) in windows)

    def test_run_aggregates_metrics(self):
        from walk_forward import WalkForwardOptimizer

        candles = [[i] for i in range(240)]
        wfo = WalkForwardOptimizer(candles, train_size=60, test_size=20, step=20)

        call_log: list[tuple[int, int]] = []

        def optimize_fn(train):
            call_log.append(("train", len(train)))
            return ({"dummy": 1.0}, 2.0)

        def evaluate_fn(test, params):
            call_log.append(("test", len(test)))
            assert params == {"dummy": 1.0}
            return (1.5, 3.0, -1.0)

        summary = wfo.run(optimize_fn, evaluate_fn)

        assert summary.num_windows >= 2
        assert summary.mean_is_sharpe == pytest.approx(2.0)
        assert summary.mean_oos_sharpe == pytest.approx(1.5)
        assert summary.overfit_gap == pytest.approx(0.5)
        assert summary.sharpe_stability == 0.0  # all OOS Sharpes are identical
        assert summary.aggregate_oos_return_pct == pytest.approx(3.0 * summary.num_windows)

    def test_run_rejects_insufficient_data(self):
        from walk_forward import WalkForwardOptimizer

        candles = [[i] for i in range(50)]  # not enough for a 60/20 window
        wfo = WalkForwardOptimizer(candles, train_size=60, test_size=20, step=20)
        with pytest.raises(AssertionError, match="Not enough candles"):
            wfo.run(lambda _: ({}, 0.0), lambda _c, _p: (0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# Optimizer wiring — params actually plumbed to Strategist
# ---------------------------------------------------------------------------


class TestOptimizerPlumbing:
    def test_build_strategist_factory_produces_overridden_strategist(self):
        from optimizer import _build_strategist_factory

        weights = {
            "technical": 0.30,
            "momentum": 0.10,
            "forecast": 0.05,
            "sentiment": 0.20,
            "consensus": 0.10,
            "polymarket": 0.18,
            "fear_greed": 0.05,
            "funding": 0.02,
        }
        factory = _build_strategist_factory(weights, entry_threshold=0.33)
        strategist = factory()

        assert strategist._weights == weights  # noqa: SLF001
        assert strategist.entry_threshold == pytest.approx(0.33)

    def test_run_backtest_with_params_restores_factory_on_error(self):
        """The helper must restore CompositeStrategy._strategist_factory even if backtest throws."""
        from backtest import CompositeStrategy
        from optimizer import _run_backtest_with_params

        # Save original
        original = CompositeStrategy._strategist_factory  # noqa: SLF001

        weights = {
            "technical": 0.35,
            "momentum": 0.10,
            "forecast": 0.05,
            "sentiment": 0.20,
            "consensus": 0.10,
            "polymarket": 0.15,
            "fear_greed": 0.05,
        }

        # Empty candle list will fail inside backtesting.py; the helper must
        # still restore the factory.
        with pytest.raises(Exception):  # noqa: B017, PT011
            _run_backtest_with_params([], weights, entry_threshold=0.25)

        assert CompositeStrategy._strategist_factory is original  # noqa: SLF001

    def test_normalize_weights_sums_to_one(self):
        from optimizer import _normalize_weights

        raw = {"a": 2.0, "b": 3.0, "c": 5.0}
        norm = _normalize_weights(raw)
        assert sum(norm.values()) == pytest.approx(1.0)
        assert norm["a"] == pytest.approx(0.2)
        assert norm["c"] == pytest.approx(0.5)
