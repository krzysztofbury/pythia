"""GARCH-based volatility modeling and regime classification."""
import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)


class VolatilityRegime(str, Enum):
    """Volatility regime classification."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class VolatilityForecast:
    """GARCH volatility forecast result."""

    current_vol_pct: float
    forecast_vol_pct: float
    regime: VolatilityRegime
    method: str  # "GARCH" or "ROLLING_STDDEV"
    suggested_sl_multiplier: float = 1.0
    suggested_tp_multiplier: float = 1.0

    def to_prompt_string(self) -> str:
        """Format for the AI strategist prompt."""
        return (
            f"Volatility ({self.method}): Current {self.current_vol_pct:.2f}%, "
            f"Forecast {self.forecast_vol_pct:.2f}% "
            f"[Regime: {self.regime.value}] "
            f"(SL adj: {self.suggested_sl_multiplier:.2f}x, "
            f"TP adj: {self.suggested_tp_multiplier:.2f}x)"
        )


# Regime thresholds (annualized vol percentages)
_REGIME_THRESHOLDS = {
    VolatilityRegime.LOW: 30.0,
    VolatilityRegime.NORMAL: 60.0,
    VolatilityRegime.HIGH: 100.0,
    # Above HIGH → EXTREME
}

# Risk multipliers per regime
_RISK_MULTIPLIERS = {
    VolatilityRegime.LOW: (0.8, 0.8),       # Tighter SL/TP
    VolatilityRegime.NORMAL: (1.0, 1.0),     # Default
    VolatilityRegime.HIGH: (1.5, 1.3),       # Wider SL, slightly wider TP
    VolatilityRegime.EXTREME: (2.0, 1.5),    # Much wider SL/TP
}


def _classify_regime(annualized_vol_pct: float) -> VolatilityRegime:
    """Classify volatility into a regime bucket."""
    if annualized_vol_pct < _REGIME_THRESHOLDS[VolatilityRegime.LOW]:
        return VolatilityRegime.LOW
    if annualized_vol_pct < _REGIME_THRESHOLDS[VolatilityRegime.NORMAL]:
        return VolatilityRegime.NORMAL
    if annualized_vol_pct < _REGIME_THRESHOLDS[VolatilityRegime.HIGH]:
        return VolatilityRegime.HIGH
    return VolatilityRegime.EXTREME


class VolatilityAnalyzer:
    """GARCH(1,1) volatility model with rolling stddev fallback."""

    @staticmethod
    def _compute_log_returns(df: pd.DataFrame) -> pd.Series:
        """Compute log returns from close prices."""
        return np.log(df["close"] / df["close"].shift(1)).dropna()

    def fit_garch(self, candles: list[list]) -> VolatilityForecast:
        """Fit GARCH(1,1) on candle data and produce a volatility forecast.

        Args:
            candles: ccxt OHLCV candle list (ideally 4h candles, 200+)

        """
        if len(candles) < 30:  # noqa: PLR2004
            logger.warning("Insufficient candles (%d) for GARCH. Using fallback.", len(candles))
            return self._rolling_fallback(candles)

        df = pd.DataFrame(
            candles, columns=["timestamp", "open", "high", "low", "close", "volume"],
        ).astype({"close": float})

        returns = self._compute_log_returns(df)

        if len(returns) < 30:  # noqa: PLR2004
            return self._rolling_fallback(candles)

        # Scale returns to percentage for GARCH numerical stability
        returns_pct = returns * 100

        try:
            model = arch_model(returns_pct, vol="Garch", p=1, q=1, mean="Zero", rescale=False)
            result = model.fit(disp="off", show_warning=False)

            # Current conditional volatility (last fitted value)
            cond_vol = result.conditional_volatility
            current_vol = float(cond_vol.iloc[-1])

            # 1-step ahead forecast
            forecast = result.forecast(horizon=1)
            forecast_var = forecast.variance.iloc[-1, 0]
            forecast_vol = float(np.sqrt(forecast_var))

            # Annualize (assuming 4h candles → 6 per day × 365)
            periods_per_year = 6 * 365
            annualized_current = current_vol * np.sqrt(periods_per_year)
            annualized_forecast = forecast_vol * np.sqrt(periods_per_year)

            regime = _classify_regime(annualized_forecast)
            sl_mult, tp_mult = _RISK_MULTIPLIERS[regime]

            return VolatilityForecast(
                current_vol_pct=annualized_current,
                forecast_vol_pct=annualized_forecast,
                regime=regime,
                method="GARCH",
                suggested_sl_multiplier=sl_mult,
                suggested_tp_multiplier=tp_mult,
            )
        except Exception:
            logger.warning("GARCH failed to converge. Falling back to rolling stddev.", exc_info=True)
            return self._rolling_fallback(candles)

    def _rolling_fallback(self, candles: list[list]) -> VolatilityForecast:
        """Fallback: 20-period rolling standard deviation of returns."""
        if len(candles) < 5:  # noqa: PLR2004
            return VolatilityForecast(
                current_vol_pct=0.0,
                forecast_vol_pct=0.0,
                regime=VolatilityRegime.NORMAL,
                method="ROLLING_STDDEV",
            )

        df = pd.DataFrame(
            candles, columns=["timestamp", "open", "high", "low", "close", "volume"],
        ).astype({"close": float})

        returns = self._compute_log_returns(df) * 100  # percentage
        window = min(20, len(returns))
        rolling_std = returns.rolling(window=window).std().dropna()

        if rolling_std.empty:
            return VolatilityForecast(
                current_vol_pct=0.0,
                forecast_vol_pct=0.0,
                regime=VolatilityRegime.NORMAL,
                method="ROLLING_STDDEV",
            )

        current_vol = float(rolling_std.iloc[-1])
        # Annualize (assume 4h candles)
        periods_per_year = 6 * 365
        annualized = current_vol * np.sqrt(periods_per_year)

        regime = _classify_regime(annualized)
        sl_mult, tp_mult = _RISK_MULTIPLIERS[regime]

        return VolatilityForecast(
            current_vol_pct=annualized,
            forecast_vol_pct=annualized,  # Same for fallback
            regime=regime,
            method="ROLLING_STDDEV",
            suggested_sl_multiplier=sl_mult,
            suggested_tp_multiplier=tp_mult,
        )
