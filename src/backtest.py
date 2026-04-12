"""Backtesting framework using Backtesting.py with indicator-based strategy."""

import logging
import sys
from dataclasses import dataclass

import pandas as pd
from backtesting import Backtest, Strategy

from indicators import TechnicalAnalyzer
from models import MarketData
from strategy import Strategist

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Summary of a backtest run."""

    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    buy_and_hold_return_pct: float
    num_trades: int

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Return: {self.total_return_pct:+.2f}% | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"MaxDD: {self.max_drawdown_pct:.2f}% | "
            f"Win Rate: {self.win_rate_pct:.1f}% | "
            f"Trades: {self.num_trades} | "
            f"Buy&Hold: {self.buy_and_hold_return_pct:+.2f}%"
        )


class IndicatorStrategy(Strategy):
    """Rule-based proxy strategy for backtesting.

    BUY: RSI < 30 (oversold) AND MACD histogram > 0
    SELL: RSI > 70 (overbought) OR price > upper Bollinger Band
    """

    rsi_lower = 30
    rsi_upper = 70
    bb_window = 20
    bb_std = 2

    def init(self) -> None:
        """Pre-compute indicators."""
        from ta.momentum import RSIIndicator
        from ta.trend import MACD
        from ta.volatility import BollingerBands

        close_series = pd.Series(self.data.Close)

        rsi_ind = RSIIndicator(close=close_series, window=14)
        self.rsi = self.I(lambda: rsi_ind.rsi().to_numpy(), name="RSI")

        macd_ind = MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)
        self.macd_hist = self.I(lambda: macd_ind.macd_diff().to_numpy(), name="MACD_Hist")

        bb_ind = BollingerBands(close=close_series, window=self.bb_window, window_dev=self.bb_std)
        self.bb_upper = self.I(lambda: bb_ind.bollinger_hband().to_numpy(), name="BB_Upper")

    def next(self) -> None:
        """Execute strategy logic on each bar."""
        if pd.isna(self.rsi[-1]) or pd.isna(self.macd_hist[-1]) or pd.isna(self.bb_upper[-1]):
            return

        if not self.position:
            if self.rsi[-1] < self.rsi_lower and self.macd_hist[-1] > 0:
                self.buy()
        elif self.rsi[-1] > self.rsi_upper or self.data.Close[-1] > self.bb_upper[-1]:
            self.sell()


class CompositeStrategy(Strategy):
    """Backtest strategy that uses the actual Strategist composite scorer.

    Builds a MarketData from bar data and calls Strategist.generate_signal().
    """

    ticker = "BTC/USDT"
    snapshots: list = []  # Historical market snapshots for realistic backtesting
    _snapshot_idx: int = 0

    def init(self) -> None:
        """Initialize the composite scorer components."""
        from ta.momentum import RSIIndicator
        from ta.trend import MACD, SMAIndicator
        from ta.volatility import AverageTrueRange, BollingerBands

        close_series = pd.Series(self.data.Close)

        rsi_ind = RSIIndicator(close=close_series, window=14)
        self.rsi = self.I(lambda: rsi_ind.rsi().to_numpy(), name="RSI")

        macd_ind = MACD(close=close_series, window_slow=26, window_fast=12, window_sign=9)
        self.macd_hist = self.I(lambda: macd_ind.macd_diff().to_numpy(), name="MACD_Hist")

        bb_ind = BollingerBands(close=close_series, window=20, window_dev=2)
        self.bb_pct_b = self.I(lambda: bb_ind.bollinger_pband().to_numpy(), name="BB_PctB")

        atr_ind = AverageTrueRange(
            high=pd.Series(self.data.High),
            low=pd.Series(self.data.Low),
            close=close_series,
            window=14,
        )
        self.atr = self.I(lambda: atr_ind.average_true_range().to_numpy(), name="ATR")

        sma_ind = SMAIndicator(close=close_series, window=7)
        self.sma7 = self.I(lambda: sma_ind.sma_indicator().to_numpy(), name="SMA7")

        self._strategist = Strategist()

    def next(self) -> None:
        """Build MarketData from current bar and call Strategist."""
        if pd.isna(self.rsi[-1]) or pd.isna(self.macd_hist[-1]):
            return

        from indicators import IndicatorSnapshot

        price = self.data.Close[-1]
        sma_val = self.sma7[-1] if not pd.isna(self.sma7[-1]) else None
        atr_val = self.atr[-1] if not pd.isna(self.atr[-1]) else None

        snap = IndicatorSnapshot(
            rsi=self.rsi[-1] if not pd.isna(self.rsi[-1]) else None,
            macd_histogram=self.macd_hist[-1] if not pd.isna(self.macd_hist[-1]) else None,
            bb_pct_b=self.bb_pct_b[-1] if not pd.isna(self.bb_pct_b[-1]) else None,
            atr=atr_val,
            atr_pct=(atr_val / price * 100) if atr_val and price > 0 else None,
            sma_7=sma_val,
            sma_7_deviation_pct=(
                ((price - sma_val) / sma_val * 100) if sma_val and sma_val > 0 else None
            ),
        )

        # Use historical snapshot data if available
        sentiment = "NEUTRAL: backtest"
        consensus = "Consensus unavailable."
        fear_greed = ""
        polymarket_odds = ""

        if self.snapshots and self._snapshot_idx < len(self.snapshots):
            hist = self.snapshots[self._snapshot_idx]
            sentiment = hist.get("sentiment", sentiment)
            consensus = hist.get("consensus", consensus)
            fear_greed = hist.get("fear_greed", "")
            polymarket_odds = hist.get("polymarket_odds", "")
            self._snapshot_idx += 1

        data = MarketData(
            ticker=self.ticker,
            current_price=price,
            indicators=snap,
            has_position=bool(self.position),
            entry_price=self.position.entry_price if self.position else None,
            pnl_pct=(
                ((price - self.position.entry_price) / self.position.entry_price * 100)
                if self.position
                else None
            ),
            sentiment=sentiment,
            consensus=consensus,
            fear_greed=fear_greed,
            polymarket_odds=polymarket_odds,
        )

        signal = self._strategist.generate_signal(data)

        if not self.position and signal.action == "BUY":
            self.buy()
        elif self.position and signal.action == "SELL":
            self.sell()


class BacktestRunner:
    """Runs backtests using historical data."""

    def __init__(self, exchange: object | None = None) -> None:
        """Initialize with optional exchange for data fetching."""
        self._exchange = exchange
        self._analyzer = TechnicalAnalyzer()

    def run(
        self,
        candles: list[list],
        cash: float = 10000.0,
        commission: float = 0.0016,
        strategy_class: type[Strategy] = IndicatorStrategy,
    ) -> BacktestResult:
        """Run a backtest on historical candle data.

        Args:
            candles: ccxt OHLCV candle list
            cash: Starting capital
            commission: Trading commission (0.16% Kraken maker)
            strategy_class: Strategy class to use (IndicatorStrategy or CompositeStrategy)

        """
        df = self._analyzer.candles_to_dataframe(candles)

        # Backtesting.py requires capitalized columns
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
        )

        bt = Backtest(
            df,
            strategy_class,
            cash=cash,
            commission=commission,
            exclusive_orders=True,
        )

        stats = bt.run()

        # Buy-and-hold baseline for comparison.
        buy_hold_return = ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100

        return BacktestResult(
            total_return_pct=float(stats.get("Return [%]", 0.0)),
            sharpe_ratio=float(stats.get("Sharpe Ratio", 0.0) or 0.0),
            max_drawdown_pct=float(stats.get("Max. Drawdown [%]", 0.0)),
            win_rate_pct=float(stats.get("Win Rate [%]", 0.0) or 0.0),
            buy_and_hold_return_pct=float(buy_hold_return),
            num_trades=int(stats.get("# Trades", 0)),
        )

    def run_from_exchange(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
        cash: float = 10000.0,
    ) -> BacktestResult | None:
        """Fetch data from exchange and run backtest."""
        if self._exchange is None:
            logger.error("No exchange configured for backtesting.")
            return None

        try:
            candles = self._exchange.fetch_ohlcv(symbol, timeframe, limit)
            return self.run(candles, cash=cash)
        except Exception:
            logger.exception("Backtest failed for %s", symbol)
            return None


def main() -> None:
    """Standalone backtest entry point."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run backtests")
    parser.add_argument(
        "--use-snapshots", action="store_true", help="Load historical snapshots from journal"
    )
    parser.add_argument(
        "--ticker", default=None, help="Single ticker to backtest (default: all universe)"
    )
    args = parser.parse_args()

    from config import Settings
    from exchange import KrakenExchange

    try:
        settings = Settings()
    except Exception:  # noqa: BLE001
        logger.critical("Failed to load settings. Check .env file.")
        sys.exit(1)

    exchange = KrakenExchange(settings.kraken_api_key, settings.kraken_private_key)
    runner = BacktestRunner(exchange)

    tickers = [args.ticker] if args.ticker else settings.universe_list

    for ticker in tickers:
        if not exchange.symbol_exists(ticker):
            logger.warning("Skipping %s: not supported.", ticker)
            continue

        logger.info("Backtesting %s...", ticker)

        if args.use_snapshots:
            from journal import TradeJournal

            journal = TradeJournal()
            snapshots = journal.get_snapshots_for_backtest(ticker)
            journal.close()

            if snapshots:
                logger.info("Loaded %d historical snapshots for %s", len(snapshots), ticker)
                CompositeStrategy.snapshots = snapshots
                CompositeStrategy._snapshot_idx = 0  # noqa: SLF001
                CompositeStrategy.ticker = ticker
                result = runner.run_from_exchange(ticker, timeframe="1h", limit=500)
                # Reset class-level state after run
                CompositeStrategy.snapshots = []
                CompositeStrategy._snapshot_idx = 0  # noqa: SLF001
            else:
                logger.warning("No snapshots found for %s, running without.", ticker)
                result = runner.run_from_exchange(ticker, timeframe="1h", limit=500)
        else:
            result = runner.run_from_exchange(ticker, timeframe="1h", limit=500)

        if result:
            logger.info("%s: %s", ticker, result.summary())
        else:
            logger.warning("%s: Backtest failed.", ticker)


if __name__ == "__main__":
    main()
