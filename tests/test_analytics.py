"""Tests for the Portfolio Analytics module."""

from analytics import PortfolioAnalyzer


class TestPortfolioAnalyzer:
    """Tests for PortfolioAnalyzer."""

    def test_compute_metrics_from_journal(self, sqlite_journal) -> None:
        """Compute metrics from a sample trade journal."""
        analyzer = PortfolioAnalyzer(journal=sqlite_journal)
        metrics = analyzer.compute_metrics()

        # 3 SELL trades in sample data
        assert metrics.total_trades == 3
        # 2 wins (BTC +1.00, SOL +5.00), 1 loss (ETH -1.00)
        assert metrics.wins == 2
        assert metrics.losses == 1
        assert abs(metrics.win_rate - 2 / 3) < 0.01
        # Total P&L: 1.00 + (-1.00) + 5.00 = 5.00
        assert abs(metrics.total_pnl - 5.0) < 0.01
        # Tax: 0.19 + 0.00 + 0.95 = 1.14
        assert abs(metrics.total_tax_liability - 1.14) < 0.01

    def test_compute_metrics_empty_journal(self, tmp_path) -> None:
        """Empty journal should return zero metrics."""
        from journal import TradeJournal

        journal = TradeJournal(db_path=str(tmp_path / "empty.db"))
        analyzer = PortfolioAnalyzer(journal=journal)
        metrics = analyzer.compute_metrics()

        assert metrics.total_trades == 0
        assert metrics.total_pnl == 0.0

    def test_summary_string(self, sqlite_journal) -> None:
        """Summary string should contain key metric labels."""
        analyzer = PortfolioAnalyzer(journal=sqlite_journal)
        metrics = analyzer.compute_metrics()
        summary = metrics.summary()

        assert "Win Rate" in summary
        assert "P&L" in summary
        assert "Sharpe" in summary
        assert "Tax" in summary
