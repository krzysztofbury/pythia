"""Tests for the PaperPortfolio."""
from paper_portfolio import PaperPortfolio


class TestPaperPortfolio:
    """Tests for PaperPortfolio position tracking."""

    def test_buy_creates_position(self):
        """Recording a buy should create a position."""
        portfolio = PaperPortfolio(initial_balance=1000.0)

        assert not portfolio.has_open_position("BTC/USDT")

        portfolio.record_buy("BTC/USDT", 44000.0, 0.001)

        assert portfolio.has_open_position("BTC/USDT")
        assert portfolio.get_cost_basis("BTC/USDT") == 44000.0
        assert portfolio.get_free_balance("BTC") == 0.001

    def test_sell_closes_position(self):
        """Selling the full quantity should close the position."""
        portfolio = PaperPortfolio(initial_balance=1000.0)

        portfolio.record_buy("BTC/USDT", 44000.0, 0.001)
        portfolio.record_sell("BTC/USDT", 0.001)

        assert not portfolio.has_open_position("BTC/USDT")
        assert portfolio.get_cost_basis("BTC/USDT") is None

    def test_dca_updates_average_cost(self):
        """Multiple buys should update the weighted average cost."""
        portfolio = PaperPortfolio(initial_balance=10000.0)

        portfolio.record_buy("SOL/USDT", 100.0, 1.0)  # 1 @ $100
        portfolio.record_buy("SOL/USDT", 120.0, 1.0)  # 1 @ $120

        assert portfolio.has_open_position("SOL/USDT")
        assert portfolio.get_free_balance("SOL") == 2.0

        cost = portfolio.get_cost_basis("SOL/USDT")
        assert cost is not None
        assert abs(cost - 110.0) < 0.01  # (100*1 + 120*1) / 2

    def test_partial_sell(self):
        """Partial sell should reduce quantity but keep position open."""
        portfolio = PaperPortfolio(initial_balance=1000.0)

        portfolio.record_buy("ETH/USDT", 3000.0, 0.1)
        portfolio.record_sell("ETH/USDT", 0.05)

        assert portfolio.has_open_position("ETH/USDT")
        assert abs(portfolio.get_free_balance("ETH") - 0.05) < 1e-10

    def test_quote_balance_tracking(self):
        """Quote currency balance should be the initial balance."""
        portfolio = PaperPortfolio(initial_balance=5000.0, quote_currency="USDT")
        assert portfolio.get_free_balance("USDT") == 5000.0

    def test_rebuild_from_db(self, tmp_path):
        """Portfolio should rebuild from SQLite journal."""
        from journal import TradeJournal

        db_path = str(tmp_path / "test.db")
        journal = TradeJournal(db_path=db_path)

        journal.log_trade(
            account="Kraken", instrument="BTC/USDT", side="BUY",
            price=44000.0, quantity=0.001, value=44.0,
            currency="USDT", reasoning="test",
        )
        journal.log_trade(
            account="Kraken", instrument="BTC/USDT", side="BUY",
            price=42000.0, quantity=0.001, value=42.0,
            currency="USDT", reasoning="DCA",
        )
        journal.close()

        portfolio = PaperPortfolio(
            initial_balance=1000.0,
            quote_currency="USDT",
            db_path=db_path,
        )

        assert portfolio.has_open_position("BTC/USDT")
        assert abs(portfolio.get_free_balance("BTC") - 0.002) < 1e-10

        cost = portfolio.get_cost_basis("BTC/USDT")
        assert cost is not None
        assert abs(cost - 43000.0) < 0.01  # (44000*0.001 + 42000*0.001) / 0.002
