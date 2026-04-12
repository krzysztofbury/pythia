"""Tests for the SQLite TradeJournal."""
from pathlib import Path

from journal import TradeJournal


class TestTradeJournal:
    """Tests for SQLite-backed TradeJournal."""

    def test_log_and_retrieve(self, tmp_path: Path) -> None:
        """Log a trade and retrieve it."""
        journal = TradeJournal(db_path=str(tmp_path / "test.db"))
        journal.log_trade(
            account="Kraken", instrument="BTC/USDT", side="BUY",
            price=42000.0, quantity=0.001, value=42.0,
            currency="USDT", reasoning="Test buy",
        )
        assert journal.get_last_buy_price("BTC/USDT") == 42000.0
        assert journal.get_last_buy_price("ETH/USDT") is None
        journal.close()

    def test_trade_count(self, tmp_path: Path) -> None:
        """Net active buys = buys - sells."""
        journal = TradeJournal(db_path=str(tmp_path / "test.db"))
        journal.log_trade(
            account="Kraken", instrument="BTC/USDT", side="BUY",
            price=42000.0, quantity=0.001, value=42.0,
            currency="USDT", reasoning="Buy 1",
        )
        journal.log_trade(
            account="Kraken", instrument="BTC/USDT", side="BUY",
            price=43000.0, quantity=0.001, value=43.0,
            currency="USDT", reasoning="Buy 2",
        )
        assert journal.get_trade_count("BTC/USDT", "BUY") == 2

        journal.log_trade(
            account="Kraken", instrument="BTC/USDT", side="SELL",
            price=44000.0, quantity=0.002, value=88.0,
            currency="USDT", reasoning="Sell all", profit=2.0, tax=0.38,
        )
        assert journal.get_trade_count("BTC/USDT", "BUY") == 1
        journal.close()

    def test_recent_history(self, tmp_path: Path) -> None:
        """get_recent_history returns formatted strings."""
        journal = TradeJournal(db_path=str(tmp_path / "test.db"))
        assert "No previous trades" in journal.get_recent_history()

        journal.log_trade(
            account="Kraken", instrument="BTC/USDT", side="BUY",
            price=42000.0, quantity=0.001, value=42.0,
            currency="USDT", reasoning="Test",
        )
        history = journal.get_recent_history(ticker="BTC/USDT")
        assert "BUY" in history
        assert "42000" in history
        journal.close()

    def test_average_cost_basis(self, tmp_path: Path) -> None:
        """Weighted average cost basis across multiple buys."""
        journal = TradeJournal(db_path=str(tmp_path / "test.db"))

        # Buy 1: 0.5 @ $100
        journal.log_trade(
            account="Kraken", instrument="SOL/USDT", side="BUY",
            price=100.0, quantity=0.5, value=50.0,
            currency="USDT", reasoning="Buy 1",
        )
        assert journal.get_average_cost_basis("SOL/USDT") == 100.0

        # Buy 2: 0.5 @ $120 -> avg should be $110
        journal.log_trade(
            account="Kraken", instrument="SOL/USDT", side="BUY",
            price=120.0, quantity=0.5, value=60.0,
            currency="USDT", reasoning="DCA",
        )
        avg = journal.get_average_cost_basis("SOL/USDT")
        assert avg is not None
        assert abs(avg - 110.0) < 0.01

        # Sell all -> no position
        journal.log_trade(
            account="Kraken", instrument="SOL/USDT", side="SELL",
            price=130.0, quantity=1.0, value=130.0,
            currency="USDT", reasoning="TP", profit=20.0, tax=3.8,
        )
        assert journal.get_average_cost_basis("SOL/USDT") is None
        journal.close()

    def test_migration(self, tmp_path: Path, sample_journal_content: str) -> None:
        """Markdown -> SQLite migration preserves all trades."""
        md_path = tmp_path / "trades.md"
        md_path.write_text(sample_journal_content, encoding="utf-8")
        db_path = str(tmp_path / "migrated.db")

        from migrate_journal import migrate_md_to_sqlite

        count = migrate_md_to_sqlite(str(md_path), db_path)
        assert count == 6

        journal = TradeJournal(db_path=db_path)
        assert journal.get_last_buy_price("BTC/USDT") == 42000.0
        assert journal.get_trade_count("BTC/USDT", "BUY") == 0  # 1 buy - 1 sell = 0
        journal.close()
