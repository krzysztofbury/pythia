"""Market regime classifier — rule-based trend/range/breakout detection.

Uses ADX for trend strength, EMA crossover for direction, ROC for momentum,
Bollinger %B and OBV for breakout detection.
"""

import logging

from models import MarketRegime

logger = logging.getLogger(__name__)


class RegimeClassifier:
    """Classifies market into regimes using technical indicators."""

    def classify(
        self,
        adx: float | None,
        ema_bullish_cross: bool | None,
        roc: float | None,
        bb_pct_b: float | None,
        obv_slope: float | None,
    ) -> MarketRegime:
        """Classify market regime from indicator values.

        Rules (evaluated in priority order):
        1. BREAKOUT: BB%B > 1.0 or < 0.0 AND |OBV slope| > 0.3
        2. TRENDING_UP: ADX > 25 AND EMA21 > EMA50 AND ROC > 0
        3. TRENDING_DOWN: ADX > 25 AND EMA21 < EMA50 AND ROC < 0
        4. RANGING: default (ADX < 20 or insufficient data)
        """
        # Breakout detection (highest priority)
        if (
            bb_pct_b is not None
            and obv_slope is not None
            and (bb_pct_b > 1.0 or bb_pct_b < 0.0)
            and abs(obv_slope) > 0.3
        ):
            return MarketRegime.BREAKOUT

        # Trend detection requires ADX and direction
        if adx is not None and adx > 25:
            has_direction = ema_bullish_cross is not None and roc is not None
            if has_direction:
                if ema_bullish_cross and roc > 0:
                    return MarketRegime.TRENDING_UP
                if not ema_bullish_cross and roc < 0:
                    return MarketRegime.TRENDING_DOWN

        return MarketRegime.RANGING

    @staticmethod
    def get_entry_threshold_adjustment(regime: MarketRegime) -> float:
        """Return entry threshold adjustment per regime.

        Positive = harder to enter, negative = easier.
        Applied as: effective_threshold = base_threshold + adjustment
        """
        adjustments = {
            MarketRegime.TRENDING_UP: -0.02,  # Easier entry in uptrends
            MarketRegime.TRENDING_DOWN: +0.05,  # Much harder in downtrends
            MarketRegime.RANGING: -0.02,  # Easier for mean reversion
            MarketRegime.BREAKOUT: +0.03,  # Wait for confirmation
        }
        assert regime in adjustments, f"No threshold adjustment for regime {regime}"  # noqa: S101
        return adjustments[regime]

    @staticmethod
    def get_stop_mode_for_regime(regime: MarketRegime) -> str:
        """Suggest stop mode based on regime.

        Returns StopMode value string.
        """
        from models import StopMode

        modes = {
            MarketRegime.TRENDING_UP: StopMode.TRAILING,
            MarketRegime.TRENDING_DOWN: StopMode.TRAILING_BREAKEVEN,
            MarketRegime.RANGING: StopMode.FIXED,
            MarketRegime.BREAKOUT: StopMode.TRAILING_BREAKEVEN,
        }
        assert regime in modes, f"No stop mode for regime {regime}"  # noqa: S101
        return modes[regime].value
