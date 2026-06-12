"""Walk-forward optimization — rolling train/test windows to detect overfitting.

A single 70/30 train/test split can flatter a lucky configuration. Walk-forward
splits the candles into overlapping rolling windows (train → test → advance),
optimizing parameters on each train window and scoring on the held-out test
window. The aggregate out-of-sample Sharpe across all windows is the honest
performance number; the in-sample number shows how much fit degrades out of
sample (overfitting gap).

Default sizing for 1h candles: 1440 bars ≈ 60 days train, 480 bars ≈ 20 days
test, step 480 bars → roughly monthly re-optimization cadence.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    """Metrics for a single walk-forward window."""

    window_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    out_of_sample_return_pct: float
    out_of_sample_max_dd_pct: float
    best_params: dict = field(default_factory=dict)


@dataclass
class WalkForwardSummary:
    """Aggregated metrics across all walk-forward windows."""

    num_windows: int
    mean_is_sharpe: float
    mean_oos_sharpe: float
    sharpe_stability: float  # stddev of OOS Sharpe — lower is better
    overfit_gap: float  # mean IS minus mean OOS
    aggregate_oos_return_pct: float
    worst_window_oos_sharpe: float
    windows: list[WindowResult] = field(default_factory=list)

    def summary_line(self) -> str:
        """One-liner for logs."""
        return (
            f"Walk-forward: {self.num_windows} windows | "
            f"mean OOS Sharpe {self.mean_oos_sharpe:.2f} "
            f"(IS {self.mean_is_sharpe:.2f}, gap {self.overfit_gap:+.2f}) | "
            f"stability {self.sharpe_stability:.2f} | "
            f"worst {self.worst_window_oos_sharpe:.2f} | "
            f"aggregate OOS return {self.aggregate_oos_return_pct:+.2f}%"
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return var**0.5


class WalkForwardOptimizer:
    """Run rolling-window optimization over a candle series."""

    def __init__(
        self,
        candles: list[list],
        *,
        train_size: int = 1440,
        test_size: int = 480,
        step: int = 480,
    ) -> None:
        """Validate window sizing and store candles."""
        assert train_size > 0, f"train_size must be positive, got {train_size}"  # noqa: S101
        assert test_size > 0, f"test_size must be positive, got {test_size}"  # noqa: S101
        assert step > 0, f"step must be positive, got {step}"  # noqa: S101
        self.candles = candles
        self.train_size = train_size
        self.test_size = test_size
        self.step = step

    def iter_windows(self) -> list[tuple[int, int, int, int]]:
        """Produce (train_start, train_end, test_start, test_end) index tuples."""
        windows: list[tuple[int, int, int, int]] = []
        i = 0
        while True:
            train_start = i
            train_end = i + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size
            if test_end > len(self.candles):
                break
            windows.append((train_start, train_end, test_start, test_end))
            i += self.step
        return windows

    def run(
        self,
        optimize_fn: Callable[[list[list]], tuple[dict, float]],
        evaluate_fn: Callable[[list[list], dict], tuple[float, float, float]],
    ) -> WalkForwardSummary:
        """Execute walk-forward.

        ``optimize_fn(train_candles) -> (best_params, in_sample_sharpe)``.
        ``evaluate_fn(test_candles, best_params) -> (oos_sharpe, oos_return_pct, oos_max_dd_pct)``.

        Both callables are injected so this module has no hard dependency on
        optuna or backtesting.py — keeps tests fast and deterministic.
        """
        windows = self.iter_windows()
        assert windows, (  # noqa: S101
            f"Not enough candles for walk-forward: need at least "
            f"{self.train_size + self.test_size}, got {len(self.candles)}"
        )

        results: list[WindowResult] = []
        for idx, (ts, te, ss, se) in enumerate(windows):
            train_candles = self.candles[ts:te]
            test_candles = self.candles[ss:se]

            best_params, is_sharpe = optimize_fn(train_candles)
            oos_sharpe, oos_return, oos_max_dd = evaluate_fn(test_candles, best_params)

            results.append(
                WindowResult(
                    window_index=idx,
                    train_start=ts,
                    train_end=te,
                    test_start=ss,
                    test_end=se,
                    in_sample_sharpe=is_sharpe,
                    out_of_sample_sharpe=oos_sharpe,
                    out_of_sample_return_pct=oos_return,
                    out_of_sample_max_dd_pct=oos_max_dd,
                    best_params=best_params,
                ),
            )
            logger.info(
                "Window %d: IS Sharpe %.2f → OOS Sharpe %.2f (return %.2f%%, DD %.2f%%)",
                idx,
                is_sharpe,
                oos_sharpe,
                oos_return,
                oos_max_dd,
            )

        is_sharpes = [r.in_sample_sharpe for r in results]
        oos_sharpes = [r.out_of_sample_sharpe for r in results]
        returns = [r.out_of_sample_return_pct for r in results]
        mean_is = _mean(is_sharpes)
        mean_oos = _mean(oos_sharpes)

        return WalkForwardSummary(
            num_windows=len(results),
            mean_is_sharpe=mean_is,
            mean_oos_sharpe=mean_oos,
            sharpe_stability=_stddev(oos_sharpes),
            overfit_gap=mean_is - mean_oos,
            aggregate_oos_return_pct=sum(returns),
            worst_window_oos_sharpe=min(oos_sharpes) if oos_sharpes else 0.0,
            windows=results,
        )
