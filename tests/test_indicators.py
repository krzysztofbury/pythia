"""Tests for the Technical Analysis module."""

import numpy as np

from indicators import IndicatorSnapshot, TechnicalAnalyzer
from models import MultiTimeframeContext


class TestTechnicalAnalyzer:
    """Tests for TechnicalAnalyzer."""

    def test_compute_returns_snapshot_with_sufficient_candles(
        self,
        sample_candles: list[list],
    ) -> None:
        """Full indicator computation with 100 candles."""
        analyzer = TechnicalAnalyzer()
        snap = analyzer.compute(sample_candles, current_price=100.0, vwap=99.5)

        assert snap.rsi is not None
        assert 0 <= snap.rsi <= 100
        assert snap.macd_histogram is not None
        assert snap.bb_upper is not None
        assert snap.bb_lower is not None
        assert snap.bb_pct_b is not None
        assert snap.atr is not None
        assert snap.atr_pct is not None
        assert snap.sma_7 is not None
        assert snap.sma_7_deviation_pct is not None
        assert snap.vwap == 99.5
        assert snap.vwap_deviation_pct is not None

    def test_compute_returns_empty_snapshot_with_insufficient_candles(
        self,
        sample_candles_short: list[list],
    ) -> None:
        """With < 20 candles, should return empty snapshot."""
        analyzer = TechnicalAnalyzer()
        snap = analyzer.compute(sample_candles_short)

        assert snap.rsi is None
        assert snap.macd_histogram is None

    def test_candles_to_dataframe(self, sample_candles: list[list]) -> None:
        """DataFrame should have correct columns and length."""
        df = TechnicalAnalyzer.candles_to_dataframe(sample_candles)

        assert len(df) == 100
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"

    def test_to_prompt_string_non_empty(self, sample_candles: list[list]) -> None:
        """Prompt string should contain key indicator labels."""
        analyzer = TechnicalAnalyzer()
        snap = analyzer.compute(sample_candles, current_price=100.0)
        prompt = snap.to_prompt_string()

        assert "RSI(14)" in prompt
        assert "MACD" in prompt
        assert "BB %B" in prompt
        assert "ATR(14)" in prompt
        assert "SMA(7)" in prompt

    def test_to_prompt_string_empty_snapshot(self) -> None:
        """Empty snapshot should return empty string."""
        snap = IndicatorSnapshot()
        assert snap.to_prompt_string() == ""

    # --- New multi-timeframe tests ---

    def test_ema_fields_populated_with_sufficient_candles(
        self,
        sample_candles: list[list],
    ) -> None:
        """EMA 21/50 fields should be populated with 100 candles."""
        analyzer = TechnicalAnalyzer()
        snap = analyzer.compute(sample_candles, current_price=100.0)

        assert snap.ema_21 is not None
        assert snap.ema_50 is not None
        assert isinstance(snap.ema_bullish_cross, bool)

    def test_ema_in_prompt_string(self, sample_candles: list[list]) -> None:
        """Prompt string should include EMA crossover info when available."""
        analyzer = TechnicalAnalyzer()
        snap = analyzer.compute(sample_candles, current_price=100.0)
        prompt = snap.to_prompt_string()

        assert "EMA(21/50)" in prompt
        assert "Cross]" in prompt

    def test_compute_short_timeframe_returns_expected_keys(
        self,
        sample_candles: list[list],
    ) -> None:
        """Short timeframe computation should return dict with rsi, bb_pct_b, macd_histogram."""
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_short_timeframe(sample_candles)

        assert "rsi" in result
        assert "bb_pct_b" in result
        assert "macd_histogram" in result
        assert result["rsi"] is not None
        assert result["bb_pct_b"] is not None

    def test_compute_short_timeframe_insufficient_data(
        self,
        sample_candles_short: list[list],
    ) -> None:
        """Short timeframe with < 20 candles should return dict with None values."""
        analyzer = TechnicalAnalyzer()
        result = analyzer.compute_short_timeframe(sample_candles_short)

        assert result == {"rsi": None, "bb_pct_b": None, "macd_histogram": None}

    def test_compute_trend_context_returns_multi_timeframe_context(
        self,
        sample_candles: list[list],
    ) -> None:
        """Trend context should return MultiTimeframeContext with trend_bias set."""
        analyzer = TechnicalAnalyzer()
        ctx = analyzer.compute_trend_context(sample_candles)

        assert isinstance(ctx, MultiTimeframeContext)
        assert ctx.trend_bias in ("BULLISH", "BEARISH", "NEUTRAL")
        assert ctx.ema_21 is not None
        assert ctx.ema_50 is not None
        assert isinstance(ctx.ema_bullish_cross, bool)

    def test_compute_trend_context_insufficient_data(
        self,
        sample_candles_short: list[list],
    ) -> None:
        """Trend context with < 50 candles should return default MultiTimeframeContext."""
        analyzer = TechnicalAnalyzer()
        ctx = analyzer.compute_trend_context(sample_candles_short)

        assert isinstance(ctx, MultiTimeframeContext)
        assert ctx.trend_bias == "NEUTRAL"
        assert ctx.ema_21 is None
        assert ctx.ema_50 is None
        assert ctx.ema_bullish_cross is False

    def test_compute_trend_context_insufficient_data_30_candles(self) -> None:
        """Trend context with 30 candles (< 50 threshold) should return defaults."""
        np.random.seed(123)
        candles = []
        ts = 1_700_000_000_000
        price = 100.0
        for i in range(30):
            o = price + np.random.normal(0, 0.5)
            h = o + abs(np.random.normal(0, 0.3))
            low = o - abs(np.random.normal(0, 0.3))
            c = o + np.random.normal(0, 0.2)
            candles.append(
                [
                    ts + i * 14_400_000,
                    round(o, 4),
                    round(h, 4),
                    round(low, 4),
                    round(c, 4),
                    500.0,
                ]
            )
            price = c

        analyzer = TechnicalAnalyzer()
        ctx = analyzer.compute_trend_context(candles)

        assert ctx.trend_bias == "NEUTRAL"
        assert ctx.ema_21 is None
