"""Tests for fill reconciliation and P&L with fees."""

import pytest

from journal import TradeJournal


@pytest.fixture
def journal(tmp_path):
    """Fresh journal with migrated schema."""
    db_path = str(tmp_path / "trades.db")
    return TradeJournal(db_path=db_path)


class TestSchemaMigration:
    def test_trades_table_has_fee_column(self, journal):
        row = journal._conn.execute("PRAGMA table_info(trades)").fetchall()
        col_names = [r[1] for r in row]
        assert "fee" in col_names

    def test_trades_table_has_fill_source_column(self, journal):
        row = journal._conn.execute("PRAGMA table_info(trades)").fetchall()
        col_names = [r[1] for r in row]
        assert "fill_source" in col_names

    def test_log_trade_accepts_fee(self, journal):
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="BUY",
            price=70000.0,
            quantity=0.001,
            value=70.0,
            currency="USDC",
            reasoning="test",
            fee=0.175,
        )
        row = journal._conn.execute("SELECT fee FROM trades WHERE instrument='BTC/USDC'").fetchone()
        assert row[0] == pytest.approx(0.175)

    def test_log_trade_accepts_fill_source(self, journal):
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="SELL",
            price=71000.0,
            quantity=0.001,
            value=71.0,
            currency="USDC",
            reasoning="SL fill",
            fee=0.284,
            fill_source="exchange",
        )
        row = journal._conn.execute("SELECT fill_source FROM trades WHERE side='SELL'").fetchone()
        assert row[0] == "exchange"
