"""Technical analysis indicators using the `ta` library."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import MACD, ADXIndicator, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from models import fmt_price as _fmt_price

logger = logging.getLogger(__name__)


@dataclass
class IndicatorSnapshot:
    """Point-in-time snapshot of all computed indicators."""

    rsi: float | None = None
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_pct_b: float | None = None
    atr: float | None = None
    atr_pct: float | None = None
    adx: float | None = None  # Trend strength (0-100)
    roc: float | None = None  # Rate of change (%)
    obv_slope: float | None = None  # OBV trend direction (-1 to +1)
    sma_7: float | None = None
    sma_7_deviation_pct: float | None = None
    vwap: float | None = None
    vwap_deviation_pct: float | None = None
    ema_21: float | None = None
    ema_50: float | None = None
    ema_bullish_cross: bool = False  # True when EMA21 > EMA50

    def to_prompt_string(self) -> str:
        """Format indicators for the AI strategist prompt."""
        parts: list[str] = []

        if self.rsi is not None:
            label = _rsi_label(self.rsi)
            parts.append(f"RSI(14): {self.rsi:.1f} [{label}]")

        if self.macd_histogram is not None:
            label = "Bullish" if self.macd_histogram > 0 else "Bearish"
            parts.append(f"MACD Histogram: {_fmt_price(abs(self.macd_histogram))} [{label}]")

        if self.bb_pct_b is not None:
            parts.append(f"BB %B: {self.bb_pct_b:.2f}")

        if self.atr_pct is not None:
            label = _atr_label(self.atr_pct)
            parts.append(f"ATR(14): {self.atr_pct:.2f}% [{label}]")

        if self.adx is not None:
            label = "Strong" if self.adx >= 25 else "Weak"  # noqa: PLR2004
            parts.append(f"ADX: {self.adx:.1f} [{label} trend]")

        if self.roc is not None:
            parts.append(f"ROC(12): {self.roc:+.2f}%")

        if self.obv_slope is not None:
            label = "Accumulating" if self.obv_slope > 0 else "Distributing"
            parts.append(f"OBV: {label}")

        if self.sma_7 is not None and self.sma_7_deviation_pct is not None:
            parts.append(
                f"SMA(7): {_fmt_price(self.sma_7)} (dev: {self.sma_7_deviation_pct:+.2f}%)",
            )

        if self.vwap is not None and self.vwap_deviation_pct is not None:
            parts.append(f"VWAP: {_fmt_price(self.vwap)} (dev: {self.vwap_deviation_pct:+.2f}%)")

        if self.ema_21 is not None and self.ema_50 is not None:
            cross = "Bullish" if self.ema_bullish_cross else "Bearish"
            parts.append(
                f"EMA(21/50): {_fmt_price(self.ema_21)} / {_fmt_price(self.ema_50)}"
                f" [{cross} Cross]",
            )

        return ", ".join(parts) if parts else ""


def _rsi_label(rsi: float) -> str:
    if rsi >= 70:  # noqa: PLR2004
        return "Overbought"
    if rsi >= 60:  # noqa: PLR2004
        return "Neutral-Bullish"
    if rsi <= 30:  # noqa: PLR2004
        return "Oversold"
    if rsi <= 40:  # noqa: PLR2004
        return "Neutral-Bearish"
    return "Neutral"


def _atr_label(atr_pct: float) -> str:
    if atr_pct >= 5:  # noqa: PLR2004
        return "High Vol"
    if atr_pct >= 2:  # noqa: PLR2004
        return "Moderate Vol"
    return "Low Vol"


class TechnicalAnalyzer:
    """Computes technical indicators from OHLCV candles."""

    @staticmethod
    def candles_to_dataframe(candles: list[list]) -> pd.DataFrame:
        """Convert ccxt OHLCV candles to a pandas DataFrame."""
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.set_index("timestamp").astype(float)

    def compute(
        self,
        candles: list[list],
        current_price: float | None = None,
        vwap: float | None = None,
    ) -> IndicatorSnapshot:
        """Compute all indicators and return a snapshot.

        Args:
            candles: ccxt OHLCV candle list
            current_price: Current spot price (for deviation calcs)
            vwap: 24h VWAP from exchange ticker

        """
        if len(candles) < 20:  # noqa: PLR2004
            logger.warning(
                "Insufficient candles (%d) for full indicator computation.",
                len(candles),
            )
            return IndicatorSnapshot()

        df = self.candles_to_dataframe(candles)
        snap = IndicatorSnapshot()

        try:
            rsi = RSIIndicator(close=df["close"], window=14)
            snap.rsi = _last_valid(rsi.rsi())
        except Exception:
            logger.debug("RSI computation failed.", exc_info=True)

        try:
            macd = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
            snap.macd_line = _last_valid(macd.macd())
            snap.macd_signal = _last_valid(macd.macd_signal())
            snap.macd_histogram = _last_valid(macd.macd_diff())
        except Exception:
            logger.debug("MACD computation failed.", exc_info=True)

        try:
            bb = BollingerBands(close=df["close"], window=20, window_dev=2)
            snap.bb_upper = _last_valid(bb.bollinger_hband())
            snap.bb_middle = _last_valid(bb.bollinger_mavg())
            snap.bb_lower = _last_valid(bb.bollinger_lband())
            snap.bb_pct_b = _last_valid(bb.bollinger_pband())
        except Exception:
            logger.debug("Bollinger Bands computation failed.", exc_info=True)

        try:
            atr = AverageTrueRange(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=14,
            )
            atr_val = _last_valid(atr.average_true_range())
            snap.atr = atr_val
            last_close = df["close"].iloc[-1]
            if atr_val is not None and last_close > 0:
                snap.atr_pct = (atr_val / last_close) * 100
        except Exception:
            logger.debug("ATR computation failed.", exc_info=True)

        try:
            adx_ind = ADXIndicator(
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=14,
            )
            snap.adx = _last_valid(adx_ind.adx())
        except Exception:
            logger.debug("ADX computation failed.", exc_info=True)

        try:
            roc_ind = ROCIndicator(close=df["close"], window=12)
            snap.roc = _last_valid(roc_ind.roc())
        except Exception:
            logger.debug("ROC computation failed.", exc_info=True)

        try:
            obv_ind = OnBalanceVolumeIndicator(
                close=df["close"],
                volume=df["volume"],
            )
            obv_series = obv_ind.on_balance_volume()
            # Compute OBV slope: normalized direction over the last 10 periods
            obv_valid = obv_series.dropna()
            if len(obv_valid) >= 10:  # noqa: PLR2004
                obv_recent = obv_valid.iloc[-10:]
                obv_range = obv_recent.max() - obv_recent.min()
                if obv_range > 0:
                    snap.obv_slope = float(
                        (obv_recent.iloc[-1] - obv_recent.iloc[0]) / obv_range,
                    )
                else:
                    snap.obv_slope = 0.0
        except Exception:
            logger.debug("OBV computation failed.", exc_info=True)

        try:
            sma = SMAIndicator(close=df["close"], window=7)
            sma_val = _last_valid(sma.sma_indicator())
            snap.sma_7 = sma_val
            price = current_price or df["close"].iloc[-1]
            if sma_val and sma_val > 0:
                snap.sma_7_deviation_pct = ((price - sma_val) / sma_val) * 100
        except Exception:
            logger.debug("SMA computation failed.", exc_info=True)

        if vwap is not None:
            snap.vwap = vwap
            price = current_price or df["close"].iloc[-1]
            if vwap > 0:
                snap.vwap_deviation_pct = ((price - vwap) / vwap) * 100

        # EMA 21/50 crossover
        try:
            ema_21_ind = EMAIndicator(close=df["close"], window=21)
            ema_50_ind = EMAIndicator(close=df["close"], window=50)
            ema_21_val = ema_21_ind.ema_indicator().iloc[-1]
            ema_50_val = ema_50_ind.ema_indicator().iloc[-1]
            if not pd.isna(ema_21_val):
                snap.ema_21 = float(ema_21_val)
            if not pd.isna(ema_50_val):
                snap.ema_50 = float(ema_50_val)
            if snap.ema_21 is not None and snap.ema_50 is not None:
                snap.ema_bullish_cross = snap.ema_21 > snap.ema_50
        except Exception:
            logger.debug("EMA computation failed.", exc_info=True)

        return snap

    def compute_short_timeframe(self, candles: list[list]) -> dict:
        """Compute minimal indicator set for short timeframe (15m) entry timing.

        Returns dict with rsi, bb_pct_b, macd_histogram for the latest bar.
        """
        if len(candles) < 20:  # noqa: PLR2004
            return {"rsi": None, "bb_pct_b": None, "macd_histogram": None}

        df = self.candles_to_dataframe(candles)
        close = df["close"]
        result: dict[str, float | None] = {}

        try:
            rsi_ind = RSIIndicator(close=close, window=14)
            rsi_val = rsi_ind.rsi().iloc[-1]
            result["rsi"] = float(rsi_val) if not pd.isna(rsi_val) else None
        except Exception:
            result["rsi"] = None

        try:
            bb_ind = BollingerBands(close=close, window=20, window_dev=2)
            bb_pct_b = bb_ind.bollinger_pband().iloc[-1]
            result["bb_pct_b"] = float(bb_pct_b) if not pd.isna(bb_pct_b) else None
        except Exception:
            result["bb_pct_b"] = None

        try:
            macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
            macd_hist = macd_ind.macd_diff().iloc[-1]
            result["macd_histogram"] = float(macd_hist) if not pd.isna(macd_hist) else None
        except Exception:
            result["macd_histogram"] = None

        return result

    def compute_trend_context(self, candles: list[list]) -> "MultiTimeframeContext":
        """Compute trend context from longer timeframe (4h) candles.

        Returns MultiTimeframeContext with EMA crossover and trend bias.
        """
        from models import MultiTimeframeContext

        ctx = MultiTimeframeContext()
        if len(candles) < 50:  # noqa: PLR2004
            return ctx

        df = self.candles_to_dataframe(candles)
        close = df["close"]

        try:
            ema_21_ind = EMAIndicator(close=close, window=21)
            ema_50_ind = EMAIndicator(close=close, window=50)
            ema_21_val = ema_21_ind.ema_indicator().iloc[-1]
            ema_50_val = ema_50_ind.ema_indicator().iloc[-1]

            if not pd.isna(ema_21_val):
                ctx.ema_21 = float(ema_21_val)
            if not pd.isna(ema_50_val):
                ctx.ema_50 = float(ema_50_val)

            if ctx.ema_21 is not None and ctx.ema_50 is not None:
                ctx.ema_bullish_cross = ctx.ema_21 > ctx.ema_50
                if ctx.ema_bullish_cross:
                    ctx.trend_bias = "BULLISH"
                else:
                    ctx.trend_bias = "BEARISH"
        except Exception:
            logger.debug("Trend context EMA computation failed.", exc_info=True)

        # ADX for trend strength confirmation
        try:
            adx_ind = ADXIndicator(
                high=df["high"],
                low=df["low"],
                close=close,
                window=14,
            )
            adx_val = adx_ind.adx().iloc[-1]
            if not pd.isna(adx_val) and float(adx_val) < 20:  # noqa: PLR2004
                ctx.trend_bias = "NEUTRAL"  # Weak trend, override to neutral
        except Exception:
            logger.debug("Trend context ADX computation failed.", exc_info=True)

        return ctx


def _last_valid(series: pd.Series) -> float | None:
    """Extract the last non-NaN value from a pandas Series."""
    valid = series.dropna()
    if valid.empty:
        return None
    val = valid.iloc[-1]
    if np.isnan(val) or np.isinf(val):
        return None
    return float(val)
