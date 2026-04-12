"""Portfolio analytics: win rate, P&L, drawdown, Sharpe, tax liability."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from journal import TradeJournal

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """Aggregated portfolio performance metrics."""

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float | None = None
    total_tax_liability: float = 0.0
    trade_returns: list[float] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary for logging."""
        sharpe_str = f"{self.sharpe_ratio:.2f}" if self.sharpe_ratio is not None else "N/A"
        return (
            f"Trades: {self.total_trades} | "
            f"Win Rate: {self.win_rate:.1%} ({self.wins}W/{self.losses}L) | "
            f"P&L: ${self.total_pnl:+.2f} | "
            f"Max Drawdown: {self.max_drawdown_pct:.2%} | "
            f"Sharpe: {sharpe_str} | "
            f"Tax: ${self.total_tax_liability:.2f}"
        )


class PortfolioAnalyzer:
    """Queries the SQLite trade journal and computes portfolio metrics."""

    def __init__(
        self,
        journal: TradeJournal | None = None,
        db_path: str = "logs/trades.db",
    ) -> None:
        """Initialize with a TradeJournal instance or path."""
        self._journal = journal or TradeJournal(db_path=db_path)

    def compute_metrics(self) -> PortfolioMetrics:
        """Compute portfolio metrics from the trade journal."""
        conn = self._journal._conn  # noqa: SLF001
        rows = conn.execute(
            "SELECT side, profit, tax, value, date FROM trades ORDER BY id ASC",
        ).fetchall()

        if not rows:
            return PortfolioMetrics()

        metrics = PortfolioMetrics()

        # Only count SELL trades as completed round-trips
        sells = [(profit, tax, value) for side, profit, tax, value, _date in rows if side == "SELL"]
        metrics.total_trades = len(sells)

        if metrics.total_trades == 0:
            return metrics

        profits = np.array([p for p, _, _ in sells])
        taxes = np.array([t for _, t, _ in sells])
        values = np.array([v for _, _, v in sells])

        metrics.wins = int(np.sum(profits > 0))
        metrics.losses = int(np.sum(profits <= 0))
        metrics.win_rate = metrics.wins / metrics.total_trades if metrics.total_trades else 0.0
        metrics.total_pnl = float(np.sum(profits))
        metrics.total_tax_liability = float(np.sum(taxes))

        # Compute percentage returns per trade (profit / trade value)
        safe_values = np.where(values > 0, values, 1.0)
        pct_returns = profits / safe_values
        metrics.trade_returns = pct_returns.tolist()

        # Max drawdown from cumulative P&L curve.
        if len(profits) > 1:
            cumulative_pnl = np.cumsum(profits)
            peak = np.maximum.accumulate(cumulative_pnl)
            drawdowns = cumulative_pnl - peak
            # Avoid division by zero when peak is 0 (no gains yet).
            peak_nonzero = np.where(peak > 0, peak, 1.0)
            drawdown_pct = drawdowns / peak_nonzero
            metrics.max_drawdown_pct = float(np.min(drawdown_pct)) if len(drawdown_pct) > 0 else 0.0

        # Sharpe ratio using daily portfolio returns (proper method)
        daily_pnl = self._compute_daily_returns(rows)
        if len(daily_pnl) >= 2:  # noqa: PLR2004
            mean_daily = np.mean(daily_pnl)
            std_daily = np.std(daily_pnl, ddof=1)
            if std_daily > 0:
                metrics.sharpe_ratio = float((mean_daily / std_daily) * np.sqrt(365))

        return metrics

    @staticmethod
    def _compute_daily_returns(rows: list) -> np.ndarray:
        """Compute daily net P&L from all trades for proper Sharpe calculation."""
        daily_pnl: dict[str, float] = defaultdict(float)
        for side, profit, _tax, _value, date in rows:
            if side == "SELL":
                daily_pnl[date] += profit

        if not daily_pnl:
            return np.array([])

        return np.array(list(daily_pnl.values()))

    def log_summary(self) -> None:
        """Compute and log portfolio metrics."""
        try:
            metrics = self.compute_metrics()
            logger.info("Portfolio: %s", metrics.summary())
        except Exception:
            logger.exception("Failed to compute portfolio analytics.")
