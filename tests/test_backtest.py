"""Tests for the Backtesting module."""
import numpy as np

from backtest import BacktestResult, BacktestRunner, CompositeStrategy, IndicatorStrategy


def _make_oscillating_candles(n: int = 300) -> list[list]:
    """Generate candles with oscillating pattern to trigger RSI signals."""
    np.random.seed(42)
    candles = []
    ts = 1_700_000_000_000
    price = 100.0

    for i in range(n):
        cycle = np.sin(i / 15) * 5
        noise = np.random.normal(0, 0.3)
        o = price + cycle + noise
        h = o + abs(np.random.normal(0, 0.5))
        l_ = o - abs(np.random.normal(0, 0.5))
        c = o + np.random.normal(0, 0.2)
        candles.append([
            ts + i * 3_600_000,
            round(o, 4), round(h, 4), round(l_, 4), round(c, 4),
            round(np.random.uniform(100, 1000), 2),
        ])
        price = c

    return candles


class TestBacktestRunner:
    """Tests for BacktestRunner."""

    def test_run_with_indicator_strategy(self) -> None:
        """Backtest should produce results with IndicatorStrategy."""
        candles = _make_oscillating_candles()
        runner = BacktestRunner()
        result = runner.run(candles, cash=10000.0, strategy_class=IndicatorStrategy)

        assert isinstance(result, BacktestResult)
        assert isinstance(result.total_return_pct, float)
        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown_pct, float)
        assert isinstance(result.num_trades, int)

    def test_run_with_composite_strategy(self) -> None:
        """Backtest should produce results with CompositeStrategy."""
        candles = _make_oscillating_candles()
        runner = BacktestRunner()
        result = runner.run(candles, cash=10000.0, strategy_class=CompositeStrategy)

        assert isinstance(result, BacktestResult)
        assert isinstance(result.total_return_pct, float)

    def test_run_summary_string(self) -> None:
        """Summary string should contain key metric labels."""
        result = BacktestResult(
            total_return_pct=12.5,
            sharpe_ratio=1.2,
            max_drawdown_pct=-5.3,
            win_rate_pct=60.0,
            buy_and_hold_return_pct=8.0,
            num_trades=15,
        )
        summary = result.summary()

        assert "Return" in summary
        assert "Sharpe" in summary
        assert "MaxDD" in summary
        assert "Win Rate" in summary
        assert "Buy&Hold" in summary

    def test_run_from_exchange_without_exchange(self) -> None:
        """Should return None if no exchange is configured."""
        runner = BacktestRunner(exchange=None)
        result = runner.run_from_exchange("BTC/USDT")

        assert result is None
