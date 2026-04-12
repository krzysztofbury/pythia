"""ARIMA-based price forecasting using pmdarima."""

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pmdarima as pm

from models import fmt_price as _fmt_price

logger = logging.getLogger(__name__)


@dataclass
class PriceForecast:
    """ARIMA forecast result."""

    current_price: float
    forecast_price: float
    lower_bound: float  # 95% CI lower
    upper_bound: float  # 95% CI upper
    direction: str  # "UP", "DOWN", "FLAT"
    change_pct: float
    method: str  # "ARIMA" or "NAIVE"

    def to_prompt_string(self) -> str:
        """Format for the AI strategist prompt."""
        return (
            f"Price Forecast ({self.method}): "
            f"${_fmt_price(self.forecast_price)} ({self.change_pct:+.2f}% {self.direction}) "
            f"[95% CI: ${_fmt_price(self.lower_bound)} – ${_fmt_price(self.upper_bound)}]"
        )


class PriceForecaster:
    """Auto-ARIMA price forecaster."""

    def __init__(self, max_order: int = 5) -> None:
        """Initialize with maximum ARIMA order bounds."""
        self.max_order = max_order

    def forecast(self, candles: list[list]) -> PriceForecast:
        """Fit auto_arima on log-transformed close prices and forecast 1 period ahead.

        Args:
            candles: ccxt OHLCV candle list

        """
        if len(candles) < 30:  # noqa: PLR2004
            logger.warning(
                "Insufficient candles (%d) for ARIMA. Using naive forecast.",
                len(candles),
            )
            return self._naive_fallback(candles)

        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        closes = df["close"].astype(float).values
        current_price = float(closes[-1])

        if current_price <= 0:
            return self._naive_fallback(candles)

        try:
            # Log-transform for stationarity
            log_prices = np.log(closes)

            model = pm.auto_arima(
                log_prices,
                start_p=0,
                start_q=0,
                max_p=self.max_order,
                max_q=self.max_order,
                max_d=2,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
            )

            # 1-step ahead forecast with 95% CI
            fc, conf_int = model.predict(n_periods=1, return_conf_int=True, alpha=0.05)

            # Back-transform from log space
            forecast_price = float(np.exp(fc[0]))
            lower_bound = float(np.exp(conf_int[0, 0]))
            upper_bound = float(np.exp(conf_int[0, 1]))

            change_pct = ((forecast_price - current_price) / current_price) * 100
            direction = _classify_direction(change_pct)

            return PriceForecast(
                current_price=current_price,
                forecast_price=forecast_price,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                direction=direction,
                change_pct=change_pct,
                method="ARIMA",
            )
        except Exception:
            logger.warning("ARIMA forecast failed. Using naive fallback.", exc_info=True)
            return self._naive_fallback(candles)

    def _naive_fallback(self, candles: list[list]) -> PriceForecast:
        """Naive forecast: last price with stddev-based CI."""
        if not candles:
            return PriceForecast(
                current_price=0.0,
                forecast_price=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
                direction="FLAT",
                change_pct=0.0,
                method="NAIVE",
            )

        closes = [float(c[4]) for c in candles]
        current_price = closes[-1]

        # Use recent stddev for CI bounds
        if len(closes) >= 2:  # noqa: PLR2004
            std = float(np.std(closes[-20:], ddof=1))
        else:
            std = 0.0

        return PriceForecast(
            current_price=current_price,
            forecast_price=current_price,
            lower_bound=current_price - 2 * std,
            upper_bound=current_price + 2 * std,
            direction="FLAT",
            change_pct=0.0,
            method="NAIVE",
        )


def _classify_direction(change_pct: float) -> str:
    """Classify forecast direction.

    Threshold widened from ±0.1% to ±0.3% — ARIMA 1h crypto forecasts
    rarely exceed 0.1%, making the old threshold effectively always FLAT.
    """
    if math.isnan(change_pct) or math.isinf(change_pct):
        logger.warning("Non-finite change_pct: %s, defaulting to FLAT", change_pct)
        return "FLAT"
    if change_pct > 0.3:  # noqa: PLR2004
        return "UP"
    if change_pct < -0.3:  # noqa: PLR2004
        return "DOWN"
    return "FLAT"
