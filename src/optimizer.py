"""Optuna-based hyperparameter optimization for strategy parameters.

Optimizes strategy weights and thresholds by running backtests with
different parameter combinations. Results saved to JSON for human review.

Usage:
    PYTHONPATH=src python src/optimizer.py --symbol BTC/USDC --trials 100
"""

import json
import logging
import sys

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """Optimize strategy parameters using Optuna."""

    def __init__(self, candles: list[list], ticker: str = "BTC/USDC") -> None:
        self.candles = candles
        self.ticker = ticker

    def objective(self, trial) -> float:
        """Optuna objective: maximize Sharpe ratio from backtest."""
        from backtest import BacktestRunner

        # Sample strategy parameters
        params = {
            "w_technical": trial.suggest_float("w_technical", 0.15, 0.40),
            "w_momentum": trial.suggest_float("w_momentum", 0.05, 0.20),
            "w_forecast": trial.suggest_float("w_forecast", 0.00, 0.10),
            "w_sentiment": trial.suggest_float("w_sentiment", 0.10, 0.30),
            "w_consensus": trial.suggest_float("w_consensus", 0.05, 0.15),
            "w_polymarket": trial.suggest_float("w_polymarket", 0.05, 0.25),
            "w_fear_greed": trial.suggest_float("w_fear_greed", 0.05, 0.15),
            "entry_threshold": trial.suggest_float("entry_threshold", 0.05, 0.20),
            "mean_reversion_bonus": trial.suggest_float("mean_reversion_bonus", 0.0, 0.20),
        }

        # Normalize weights to sum to 1.0
        weight_sum = sum(
            params[k]
            for k in [
                "w_technical",
                "w_momentum",
                "w_forecast",
                "w_sentiment",
                "w_consensus",
                "w_polymarket",
                "w_fear_greed",
            ]
        )
        for k in [
            "w_technical",
            "w_momentum",
            "w_forecast",
            "w_sentiment",
            "w_consensus",
            "w_polymarket",
            "w_fear_greed",
        ]:
            params[k] /= weight_sum

        # Split data 70/30
        split_idx = int(len(self.candles) * 0.7)
        train_candles = self.candles[:split_idx]

        if len(train_candles) < 100:
            return -999.0  # Not enough data

        # Run backtest with sampled params
        runner = BacktestRunner()
        try:
            result = runner.run(train_candles, cash=10000.0)
            return result.sharpe_ratio if result.sharpe_ratio else 0.0
        except Exception:
            return -999.0

    def optimize(self, n_trials: int = 200) -> dict:
        """Run optimization and return best parameters."""
        import optuna

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_params
        best_value = study.best_value

        # Normalize weights
        weight_keys = [
            "w_technical",
            "w_momentum",
            "w_forecast",
            "w_sentiment",
            "w_consensus",
            "w_polymarket",
            "w_fear_greed",
        ]
        weight_sum = sum(best[k] for k in weight_keys)
        for k in weight_keys:
            best[k] = round(best[k] / weight_sum, 4)

        logger.info("Best Sharpe: %.4f", best_value)
        logger.info("Best params: %s", best)
        return {"sharpe": best_value, "params": best}

    def validate(self, best_params: dict) -> dict:
        """Validate best params on held-out 30% of data."""
        split_idx = int(len(self.candles) * 0.7)
        test_candles = self.candles[split_idx:]

        if len(test_candles) < 50:
            return {"error": "Insufficient test data"}

        runner = BacktestRunner()
        try:
            result = runner.run(test_candles, cash=10000.0)
            return {
                "test_sharpe": result.sharpe_ratio,
                "test_return_pct": result.total_return_pct,
                "test_max_dd": result.max_drawdown_pct,
                "test_trades": result.num_trades,
            }
        except Exception as e:
            return {"error": str(e)}

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
    parser.add_argument("--trials", type=int, default=200, help="Number of Optuna trials")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles")
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
    train_results = optimizer.optimize(n_trials=args.trials)

    logger.info("Validating on held-out data...")
    val_results = optimizer.validate(train_results["params"])

    combined = {
        **train_results,
        "validation": val_results,
        "symbol": args.symbol,
        "trials": args.trials,
    }
    optimizer.save_results(combined)
    logger.info("Done. Results: %s", json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
