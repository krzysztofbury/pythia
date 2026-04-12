"""Tests for the ARIMA Forecast module."""
from forecast import PriceForecast, PriceForecaster


class TestPriceForecaster:
    """Tests for PriceForecaster."""

    def test_forecast_with_sufficient_data(self, sample_candles: list[list]) -> None:
        """ARIMA should produce a forecast with 100 candles."""
        forecaster = PriceForecaster(max_order=3)
        result = forecaster.forecast(sample_candles)

        assert isinstance(result, PriceForecast)
        assert result.forecast_price > 0
        assert result.lower_bound <= result.forecast_price <= result.upper_bound
        assert result.direction in ("UP", "DOWN", "FLAT")
        assert result.method in ("ARIMA", "NAIVE")

    def test_forecast_falls_back_with_short_data(
        self, sample_candles_short: list[list],
    ) -> None:
        """Short data should trigger naive fallback."""
        forecaster = PriceForecaster()
        result = forecaster.forecast(sample_candles_short)

        assert result.method == "NAIVE"

    def test_forecast_empty_candles(self) -> None:
        """Empty candles should return safe default."""
        forecaster = PriceForecaster()
        result = forecaster.forecast([])

        assert result.method == "NAIVE"
        assert result.forecast_price == 0.0

    def test_to_prompt_string(self, sample_candles: list[list]) -> None:
        """Prompt string should contain key labels."""
        forecaster = PriceForecaster(max_order=3)
        result = forecaster.forecast(sample_candles)
        prompt = result.to_prompt_string()

        assert "Price Forecast" in prompt
        assert "95% CI" in prompt

    def test_change_pct_calculation(self) -> None:
        """Verify change percentage is computed correctly."""
        result = PriceForecast(
            current_price=100.0,
            forecast_price=105.0,
            lower_bound=95.0,
            upper_bound=115.0,
            direction="UP",
            change_pct=5.0,
            method="ARIMA",
        )
        assert result.change_pct == 5.0
        assert result.direction == "UP"
