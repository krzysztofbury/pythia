"""Signal analytics — per-asset stats, MAE/MFE, factor attribution.

Reads from ``trades``, ``market_snapshots``, and ``signals`` tables. Pure
post-hoc analysis — no side effects on trading behaviour.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from journal import TradeJournal

logger = logging.getLogger(__name__)


@dataclass
class PerAssetStats:
    """Aggregated metrics for a single ticker."""

    ticker: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    def summary_line(self) -> str:
        """One-line formatted summary."""
        return (
            f"{self.ticker}: {self.trades} trades | "
            f"WR {self.win_rate:.1%} ({self.wins}W/{self.losses}L) | "
            f"P&L ${self.total_pnl:+.2f} (avg ${self.avg_pnl:+.2f}) | "
            f"best ${self.best_trade:+.2f} / worst ${self.worst_trade:+.2f}"
        )


@dataclass
class TradeExcursion:
    """Maximum Adverse / Favorable Excursion for one round-trip trade."""

    trade_id: int
    ticker: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    pnl: float
    mae_pct: float  # worst drawdown during hold
    mfe_pct: float  # best unrealized profit during hold


@dataclass
class FactorAttribution:
    """Pearson correlation between one composite component and trade P&L."""

    component: str
    correlation_with_pnl: float
    sample_size: int
    mean_when_profitable: float = 0.0
    mean_when_losing: float = 0.0


@dataclass
class SignalAnalyticsSummary:
    """Full analytics output returned by ``SignalAnalyzer.run``."""

    per_asset: list[PerAssetStats] = field(default_factory=list)
    recent_excursions: list[TradeExcursion] = field(default_factory=list)
    factors: list[FactorAttribution] = field(default_factory=list)


# Regex for parsing signal reasoning rows like:
#   "Algo [BTC/USDC]: comp=+0.138 tech=+0.025 mom=+0.020 fc=+0.000 sent=+0.333"
_COMPONENT_PATTERN = re.compile(
    r"\b(?P<key>tech|mom|fc|sent|cons|poly|fg|fund)=(?P<sign>[+-]?)(?P<value>\d+\.\d+)",
    re.IGNORECASE,
)

_KEY_TO_COMPONENT = {
    "tech": "technical",
    "mom": "momentum",
    "fc": "forecast",
    "sent": "sentiment",
    "cons": "consensus",
    "poly": "polymarket",
    "fg": "fear_greed",
    "fund": "funding",
}


def _parse_component_scores(reasoning: str) -> dict[str, float]:
    """Best-effort extraction of per-component scores from free-text reasoning.

    Returns an empty dict when the row can't be parsed — we prefer to drop a
    sample rather than fabricate numbers.
    """
    scores: dict[str, float] = {}
    for match in _COMPONENT_PATTERN.finditer(reasoning):
        raw_key = match.group("key").lower()
        component = _KEY_TO_COMPONENT.get(raw_key)
        if component is None:
            continue
        sign = -1.0 if match.group("sign") == "-" else 1.0
        try:
            scores[component] = sign * float(match.group("value"))
        except ValueError:
            continue
    return scores


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation; returns 0.0 when variance is zero."""
    assert len(xs) == len(ys), f"xs/ys length mismatch: {len(xs)} vs {len(ys)}"  # noqa: S101
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    if denom == 0.0:
        return 0.0
    return num / denom


class SignalAnalyzer:
    """Queries post-hoc analytics from the trade journal."""

    def __init__(
        self,
        journal: TradeJournal | None = None,
        db_path: str = "logs/trades.db",
    ) -> None:
        """Use an existing journal if provided, otherwise open the given path."""
        self._journal = journal or TradeJournal(db_path=db_path)

    # ------------------------------------------------------------------
    # Per-asset
    # ------------------------------------------------------------------

    def per_asset_metrics(self) -> list[PerAssetStats]:
        """Aggregate SELL-side P&L per ticker."""
        conn = self._journal._conn  # noqa: SLF001
        rows = conn.execute(
            "SELECT instrument, profit FROM trades WHERE side = 'SELL' ORDER BY instrument",
        ).fetchall()

        if not rows:
            return []

        by_ticker: dict[str, list[float]] = {}
        for ticker, profit in rows:
            by_ticker.setdefault(ticker, []).append(float(profit or 0.0))

        stats: list[PerAssetStats] = []
        for ticker, profits in by_ticker.items():
            n = len(profits)
            wins = sum(1 for p in profits if p > 0)
            losses = n - wins
            total = sum(profits)
            avg = total / n if n else 0.0
            stats.append(
                PerAssetStats(
                    ticker=ticker,
                    trades=n,
                    wins=wins,
                    losses=losses,
                    win_rate=wins / n if n else 0.0,
                    total_pnl=total,
                    avg_pnl=avg,
                    best_trade=max(profits),
                    worst_trade=min(profits),
                ),
            )
        stats.sort(key=lambda s: s.total_pnl, reverse=True)
        return stats

    # ------------------------------------------------------------------
    # MAE / MFE
    # ------------------------------------------------------------------

    def compute_mae_mfe(
        self,
        ticker: str | None = None,
        limit: int = 50,
    ) -> list[TradeExcursion]:
        """Compute MAE/MFE for recent SELL trades; optionally filter by ticker."""
        conn = self._journal._conn  # noqa: SLF001
        where = " WHERE side = 'SELL'"
        params: list = []
        if ticker:
            where += " AND instrument = ?"
            params.append(ticker)
        rows = conn.execute(
            "SELECT id, instrument, timestamp, price, profit FROM trades"
            + where
            + " ORDER BY id DESC LIMIT ?",
            (*params, limit),
        ).fetchall()

        results: list[TradeExcursion] = []
        for trade_id, symbol, exit_time, exit_price, pnl in rows:
            # Find the ORIGINAL position entry — the first BUY after the most
            # recent prior SELL for this ticker (or the first BUY ever, if no
            # previous exit). Using "most recent BUY" would pick a DCA add and
            # misrepresent MAE/MFE vs. the position's true cost basis.
            prior_sell = conn.execute(
                "SELECT timestamp FROM trades "
                "WHERE instrument = ? AND side = 'SELL' AND timestamp < ? "
                "ORDER BY id DESC LIMIT 1",
                (symbol, exit_time),
            ).fetchone()
            since_ts = prior_sell[0] if prior_sell else ""
            buy_row = conn.execute(
                "SELECT id, timestamp, price FROM trades "
                "WHERE instrument = ? AND side = 'BUY' "
                "AND timestamp > ? AND timestamp <= ? "
                "ORDER BY id ASC LIMIT 1",
                (symbol, since_ts, exit_time),
            ).fetchone()
            if buy_row is None:
                continue
            _buy_id, entry_time, entry_price = buy_row
            if not entry_price or entry_price <= 0:
                continue

            # Scan snapshots between entry and exit for min/max price.
            snaps = conn.execute(
                "SELECT price FROM market_snapshots "
                "WHERE ticker = ? AND timestamp BETWEEN ? AND ? "
                "ORDER BY timestamp ASC",
                (symbol, entry_time, exit_time),
            ).fetchall()
            if not snaps:
                # No snapshots covering this hold; use entry/exit prices only.
                low = min(entry_price, exit_price)
                high = max(entry_price, exit_price)
            else:
                prices = [row[0] for row in snaps]
                low = min(prices + [entry_price, exit_price])
                high = max(prices + [entry_price, exit_price])

            mae_pct = (low - entry_price) / entry_price * 100
            mfe_pct = (high - entry_price) / entry_price * 100

            results.append(
                TradeExcursion(
                    trade_id=int(trade_id),
                    ticker=symbol,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=float(entry_price),
                    exit_price=float(exit_price),
                    pnl=float(pnl or 0.0),
                    mae_pct=float(mae_pct),
                    mfe_pct=float(mfe_pct),
                ),
            )
        return results

    # ------------------------------------------------------------------
    # Factor attribution
    # ------------------------------------------------------------------

    def factor_attribution(self) -> list[FactorAttribution]:
        """Correlate per-component scores (parsed from reasoning) with trade P&L.

        Pairs each completed SELL with the most recent BUY signal on the same
        ticker (the one that likely triggered the entry) and reads the
        component scores from the signal's reasoning column.
        """
        conn = self._journal._conn  # noqa: SLF001

        sells = conn.execute(
            "SELECT instrument, timestamp, profit FROM trades WHERE side = 'SELL' ORDER BY id ASC",
        ).fetchall()
        if not sells:
            return []

        paired: list[tuple[dict[str, float], float]] = []
        for ticker, exit_time, profit in sells:
            sig_row = conn.execute(
                "SELECT reasoning FROM signals "
                "WHERE ticker = ? AND action = 'BUY' AND timestamp <= ? "
                "ORDER BY id DESC LIMIT 1",
                (ticker, exit_time),
            ).fetchone()
            if sig_row is None:
                continue
            scores = _parse_component_scores(sig_row[0] or "")
            if not scores:
                continue
            paired.append((scores, float(profit or 0.0)))

        if not paired:
            return []

        components = sorted({c for scores, _ in paired for c in scores})
        results: list[FactorAttribution] = []
        for component in components:
            xs: list[float] = []
            ys: list[float] = []
            profitable_vals: list[float] = []
            losing_vals: list[float] = []
            for scores, pnl in paired:
                if component not in scores:
                    continue
                xs.append(scores[component])
                ys.append(pnl)
                if pnl > 0:
                    profitable_vals.append(scores[component])
                else:
                    losing_vals.append(scores[component])
            if len(xs) < 2:
                continue
            corr = _pearson(xs, ys)
            results.append(
                FactorAttribution(
                    component=component,
                    correlation_with_pnl=corr,
                    sample_size=len(xs),
                    mean_when_profitable=(
                        sum(profitable_vals) / len(profitable_vals) if profitable_vals else 0.0
                    ),
                    mean_when_losing=(sum(losing_vals) / len(losing_vals) if losing_vals else 0.0),
                ),
            )
        results.sort(key=lambda f: abs(f.correlation_with_pnl), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def run(self) -> SignalAnalyticsSummary:
        """Execute all three analyses; useful for CLI reports."""
        return SignalAnalyticsSummary(
            per_asset=self.per_asset_metrics(),
            recent_excursions=self.compute_mae_mfe(limit=25),
            factors=self.factor_attribution(),
        )

    def log_summary(self) -> None:
        """Log a compact digest of the analytics."""
        try:
            summary = self.run()
        except Exception:
            logger.exception("Signal analytics failed.")
            return

        if summary.per_asset:
            logger.info("Signal Analytics: per-asset breakdown:")
            for stat in summary.per_asset:
                logger.info("  %s", stat.summary_line())
        else:
            logger.info("Signal Analytics: no completed trades yet.")

        if summary.factors:
            logger.info("Signal Analytics: factor attribution (|corr| desc):")
            for f in summary.factors:
                logger.info(
                    "  %s: corr=%+0.3f n=%d (avg when profitable=%+0.3f, when losing=%+0.3f)",
                    f.component,
                    f.correlation_with_pnl,
                    f.sample_size,
                    f.mean_when_profitable,
                    f.mean_when_losing,
                )
