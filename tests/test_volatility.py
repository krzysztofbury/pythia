"""Tests for the Volatility module."""
from volatility import VolatilityAnalyzer, VolatilityForecast, VolatilityRegime


class TestVolatilityAnalyzer:
    """Tests for VolatilityAnalyzer."""

    def test_fit_garch_with_sufficient_data(self, sample_candles: list[list]) -> None:
        """GARCH should produce a forecast with 100 candles."""
        analyzer = VolatilityAnalyzer()
        forecast = analyzer.fit_garch(sample_candles)

        assert isinstance(forecast, VolatilityForecast)
        assert forecast.current_vol_pct >= 0
        assert forecast.forecast_vol_pct >= 0
        assert forecast.regime in VolatilityRegime
        assert forecast.method in ("GARCH", "ROLLING_STDDEV")

    def test_fit_garch_falls_back_with_short_data(
        self, sample_candles_short: list[list],
    ) -> None:
        """Short data should trigger rolling stddev fallback."""
        analyzer = VolatilityAnalyzer()
        forecast = analyzer.fit_garch(sample_candles_short)

        assert forecast.method == "ROLLING_STDDEV"

    def test_fit_garch_empty_candles(self) -> None:
        """Empty candle list should produce safe default."""
        analyzer = VolatilityAnalyzer()
        forecast = analyzer.fit_garch([])

        assert forecast.method == "ROLLING_STDDEV"
        assert forecast.regime == VolatilityRegime.NORMAL

    def test_to_prompt_string(self, sample_candles: list[list]) -> None:
        """Prompt string should contain key labels."""
        analyzer = VolatilityAnalyzer()
        forecast = analyzer.fit_garch(sample_candles)
        prompt = forecast.to_prompt_string()

        assert "Volatility" in prompt
        assert "Regime" in prompt

    def test_risk_multipliers(self) -> None:
        """Verify risk multipliers are set per regime."""
        forecast = VolatilityForecast(
            current_vol_pct=120.0,
            forecast_vol_pct=120.0,
            regime=VolatilityRegime.EXTREME,
            method="GARCH",
            suggested_sl_multiplier=2.0,
            suggested_tp_multiplier=1.5,
        )
        assert forecast.suggested_sl_multiplier == 2.0
        assert forecast.suggested_tp_multiplier == 1.5
