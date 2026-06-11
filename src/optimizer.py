"""Optuna-based strategy optimization with walk-forward validation.

v9.3 — Parameters sampled by Optuna are now actually threaded into the
backtest via ``CompositeStrategy._strategist_factory``; previous versions
sampled weights but evaluated with hard-coded defaults, which made every
trial identical. Walk-forward windows replace the lucky-split 70/30 approach.

Usage:
    PYTHONPATH=src python src/optimizer.py --symbol BTC/USDC --trials 50
    PYTHONPATH=src python src/optimizer.py --symbol BTC/USDC --walk-forward
"""

from __future__ import annotations

import json
import logging
import sys

logger = logging.getLogger(__name__)


_WEIGHT_KEYS = (
    "technical",
    "momentum",
    "forecast",
    "sentiment",
    "consensus",
    "polymarket",
    "fear_greed",
)


def _normalize_weights(raw: dict) -> dict:
    total = sum(raw.values())
    if total <= 0:
        msg = f"Invalid weights sample (sum={total}): {raw}"
        raise ValueError(msg)
    return {k: v / total for k, v in raw.items()}


def _build_strategist_factory(weights: dict, entry_threshold: float):
    """Return a zero-arg callable producing a configured Strategist.

    The returned factory is assigned to ``CompositeStrategy._strategist_factory``
    so each backtest bar calls ``Strategist.generate_signal`` with the sampled
    weights and entry threshold.
    """
    from strategy import Strategist

    class _StubSettings:
        """Minimal settings adapter — only the fields Strategist asserts on."""

        entry_threshold = 0.25
        exit_threshold = -0.05
        strong_sell_threshold = -0.25
        dca_threshold = 0.30
        profit_take_pct = 5.0
        momentum_weak_threshold = 0.1
        sma_overextended_pct = 15.0
        mean_reversion_rsi = 28.0
        mean_reversion_adx = 30.0
        mean_reversion_bonus = 0.10
        stale_position_hours = 120.0
        stale_exit_pnl_threshold_pct = 2.0

    stub = _StubSettings()
    stub.entry_threshold = entry_threshold

    def factory() -> Strategist:
        return Strategist(settings=stub, weights_override=weights)

    return factory


def _run_backtest_with_params(candles: list[list], weights: dict, entry_threshold: float) -> dict:
    """Run a single backtest with the given params; return key stats."""
    from backtest import BacktestRunner, CompositeStrategy

    previous_factory = CompositeStrategy._strategist_factory  # noqa: SLF001
    CompositeStrategy._strategist_factory = _build_strategist_factory(  # noqa: SLF001
        weights,
        entry_threshold,
    )
    try:
        runner = BacktestRunner()
        result = runner.run(
            candles,
            cash=10_000.0,
            strategy_class=CompositeStrategy,
        )
    finally:
        CompositeStrategy._strategist_factory = previous_factory  # noqa: SLF001

    return {
        "sharpe": result.sharpe_ratio,
        "sortino": result.sortino_ratio,
        "calmar": result.calmar_ratio,
        "return_pct": result.total_return_pct,
        "max_dd_pct": result.max_drawdown_pct,
        "trades": result.num_trades,
    }


class StrategyOptimizer:
    """Optimize strategy parameters using Optuna.

    A single-window run uses a 70/30 split; pass ``walk_forward=True`` to
    re-optimize across rolling windows for honest out-of-sample metrics.
    """

    def __init__(self, candles: list[list], ticker: str = "BTC/USDC") -> None:
        """Store candles + ticker (used only for logging/output keys)."""
        assert len(candles) >= 100, f"Need at least 100 candles, got {len(candles)}"  # noqa: S101
        self.candles = candles
        self.ticker = ticker

    # ------------------------------------------------------------------
    # Internal helpers shared between single-split and walk-forward paths
    # ------------------------------------------------------------------

    def _sample_params(self, trial) -> tuple[dict, float]:
        """Sample weights + entry threshold from the Optuna trial."""
        raw_weights = {
            "technical": trial.suggest_float("w_technical", 0.15, 0.40),
            "momentum": trial.suggest_float("w_momentum", 0.05, 0.20),
            "forecast": trial.suggest_float("w_forecast", 0.00, 0.10),
            "sentiment": trial.suggest_float("w_sentiment", 0.10, 0.30),
            "consensus": trial.suggest_float("w_consensus", 0.05, 0.15),
            "polymarket": trial.suggest_float("w_polymarket", 0.05, 0.25),
            "fear_greed": trial.suggest_float("w_fear_greed", 0.05, 0.15),
        }
        weights = _normalize_weights(raw_weights)
        entry_threshold = trial.suggest_float("entry_threshold", 0.15, 0.40)
        return weights, entry_threshold

    def _objective_for(self, train_candles: list[list]):
        """Produce an Optuna objective closed over train_candles."""

        def objective(trial) -> float:
            weights, entry_threshold = self._sample_params(trial)
            if len(train_candles) < 100:
                return -999.0
            try:
                stats = _run_backtest_with_params(train_candles, weights, entry_threshold)
            except Exception:  # noqa: BLE001
                logger.debug("Trial failed", exc_info=True)
                return -999.0
            return stats["sharpe"] or 0.0

        return objective

    # ------------------------------------------------------------------
    # Single-window optimization (70/30 split) — backward compat
    # ------------------------------------------------------------------

    def optimize(self, n_trials: int = 200) -> dict:
        """Run a single-split optimization and return best params."""
        import optuna

        split_idx = int(len(self.candles) * 0.7)
        train_candles = self.candles[:split_idx]

        study = optuna.create_study(direction="maximize")
        study.optimize(
            self._objective_for(train_candles),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best = study.best_params
        best_value = study.best_value
        raw_w = {k: best[f"w_{k}"] for k in _WEIGHT_KEYS}
        weights = _normalize_weights(raw_w)
        params = {**weights, "entry_threshold": round(best["entry_threshold"], 4)}

        logger.info("Best IS Sharpe: %.4f", best_value)
        logger.info("Best params: %s", params)
        return {"sharpe": best_value, "params": params}

    def validate(self, best_params: dict) -> dict:
        """Score best params on the held-out 30% of data."""
        split_idx = int(len(self.candles) * 0.7)
        test_candles = self.candles[split_idx:]

        if len(test_candles) < 50:
            return {"error": "Insufficient test data"}

        weights = {k: best_params[k] for k in _WEIGHT_KEYS}
        entry_threshold = best_params["entry_threshold"]

        try:
            stats = _run_backtest_with_params(test_candles, weights, entry_threshold)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
        return {
            "test_sharpe": stats["sharpe"],
            "test_sortino": stats["sortino"],
            "test_calmar": stats["calmar"],
            "test_return_pct": stats["return_pct"],
            "test_max_dd_pct": stats["max_dd_pct"],
            "test_trades": stats["trades"],
        }

    # ------------------------------------------------------------------
    # Walk-forward optimization
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        n_trials_per_window: int = 50,
        train_size: int = 1440,
        test_size: int = 480,
        step: int = 480,
    ) -> dict:
        """Re-optimize across rolling windows; return aggregate metrics."""
        import optuna

        from walk_forward import WalkForwardOptimizer

        # Silence Optuna per-trial logs during many windows.
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        wfo = WalkForwardOptimizer(
            self.candles,
            train_size=train_size,
            test_size=test_size,
            step=step,
        )

        def _optimize_on_train(train_candles: list[list]) -> tuple[dict, float]:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                self._objective_for(train_candles),
                n_trials=n_trials_per_window,
                show_progress_bar=False,
            )
            best = study.best_params
            raw_w = {k: best[f"w_{k}"] for k in _WEIGHT_KEYS}
            weights = _normalize_weights(raw_w)
            params = {**weights, "entry_threshold": best["entry_threshold"]}
            return params, study.best_value

        def _evaluate(test_candles: list[list], params: dict) -> tuple[float, float, float]:
            weights = {k: params[k] for k in _WEIGHT_KEYS}
            stats = _run_backtest_with_params(test_candles, weights, params["entry_threshold"])
            return stats["sharpe"] or 0.0, stats["return_pct"], stats["max_dd_pct"]

        summary = wfo.run(_optimize_on_train, _evaluate)
        logger.info(summary.summary_line())
        return {
            "mean_oos_sharpe": summary.mean_oos_sharpe,
            "mean_is_sharpe": summary.mean_is_sharpe,
            "sharpe_stability": summary.sharpe_stability,
            "overfit_gap": summary.overfit_gap,
            "aggregate_oos_return_pct": summary.aggregate_oos_return_pct,
            "worst_window_oos_sharpe": summary.worst_window_oos_sharpe,
            "num_windows": summary.num_windows,
            "windows": [
                {
                    "index": w.window_index,
                    "is_sharpe": w.in_sample_sharpe,
                    "oos_sharpe": w.out_of_sample_sharpe,
                    "oos_return_pct": w.out_of_sample_return_pct,
                    "oos_max_dd_pct": w.out_of_sample_max_dd_pct,
                    "params": w.best_params,
                }
                for w in summary.windows
            ],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_results(results: dict, db_path: str = "logs/trades.db") -> None:
        """Save optimization results to SQLite (replaces optimization.json)."""
        from journal import TradeJournal

        journal = TradeJournal(db_path)
        journal.save_optimization_run(
            symbol=results.get("symbol", "unknown"),
            trials=results.get("trials", 0),
            best_sharpe=results.get("sharpe", 0.0),
            params_json=json.dumps(results.get("params", {})),
            validation_json=json.dumps(results.get("validation", {})),
        )
        logger.info("Results saved to %s (optimization_runs table)", db_path)

    @staticmethod
    def load_optimized_params(
        db_path: str = "logs/trades.db",
        symbol: str | None = None,
    ) -> dict | None:
        """Load latest optimized parameters from SQLite."""
        from journal import TradeJournal

        journal = TradeJournal(db_path)
        run = journal.get_latest_optimization(symbol)
        if not run:
            return None
        return json.loads(run["params_json"])


def main() -> None:
    """CLI entry point for optimization."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Optimize strategy parameters")
    parser.add_argument("--symbol", default="BTC/USDC", help="Trading pair")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials per window")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument("--limit", type=int, default=2000, help="Candles to fetch")
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Use walk-forward optimization (recommended).",
    )
    parser.add_argument("--train-size", type=int, default=1440, help="Train window bars")
    parser.add_argument("--test-size", type=int, default=480, help="Test window bars")
    parser.add_argument("--step", type=int, default=480, help="Rolling step bars")
    args = parser.parse_args()

    from config import Settings
    from exchange import KrakenExchange

    try:
        settings = Settings()
    except Exception:  # noqa: BLE001
        logger.critical("Failed to load settings. Check .env file.")
        sys.exit(1)

    exchange = KrakenExchange(settings.kraken_api_key, settings.kraken_private_key)
    logger.info("Fetching %d %s candles for %s...", args.limit, args.timeframe, args.symbol)

    candles = exchange.fetch_ohlcv(args.symbol, args.timeframe, args.limit)
    if not candles:
        logger.error("No candle data received.")
        sys.exit(1)

    optimizer = StrategyOptimizer(candles, args.symbol)

    if args.walk_forward:
        results = optimizer.walk_forward(
            n_trials_per_window=args.trials,
            train_size=args.train_size,
            test_size=args.test_size,
            step=args.step,
        )
        results["symbol"] = args.symbol
        results["trials"] = args.trials
        # Aggregate best params = median weights across windows.
        if results["windows"]:
            last = results["windows"][-1]["params"]
            results["params"] = last
            results["sharpe"] = results["mean_oos_sharpe"]
            results["validation"] = {
                "mean_oos_sharpe": results["mean_oos_sharpe"],
                "sharpe_stability": results["sharpe_stability"],
                "overfit_gap": results["overfit_gap"],
            }
    else:
        train_results = optimizer.optimize(n_trials=args.trials)
        logger.info("Validating on held-out data...")
        val_results = optimizer.validate(train_results["params"])
        results = {
            **train_results,
            "validation": val_results,
            "symbol": args.symbol,
            "trials": args.trials,
        }

    optimizer.save_results(results)
    logger.info("Done. Results: %s", json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
