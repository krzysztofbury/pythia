"""Tests for the P&L reconciliation migration script."""

import datetime
import sqlite3

import pytest

from journal import TradeJournal


@pytest.fixture
def journal(tmp_path):
    """Fresh journal with full schema."""
    db_path = str(tmp_path / "trades.db")
    return TradeJournal(db_path=db_path)


@pytest.fixture
def populated_journal(journal):
    """Journal with sample trades that have incorrect P&L (no fees)."""
    # BUY BTC @ 70000, qty 0.001 (no fee recorded)
    journal.log_trade(
        account="Kraken",
        instrument="BTC/USDC",
        side="BUY",
        price=70000.0,
        quantity=0.001,
        value=70.0,
        currency="USDC",
        reasoning="test buy",
    )
    # SELL BTC @ 71000, qty 0.001, profit recorded without fees
    journal.log_trade(
        account="Kraken",
        instrument="BTC/USDC",
        side="SELL",
        price=71000.0,
        quantity=0.001,
        value=71.0,
        currency="USDC",
        reasoning="test sell",
        profit=1.0,
        tax=0.19,
    )
    return journal


def _fetch_trades_as_dicts(conn, query, params=()):
    """Fetch rows as list of dicts (journal conn has no row_factory)."""
    cursor = conn.execute(query, params)
    cols = [desc[0] for desc in cursor.description]
    return [dict(zip(cols, row, strict=False)) for row in cursor.fetchall()]


class TestMatchFillToJournal:
    def test_matches_by_side_qty_time(self, populated_journal):
        """Fill matching should find a journal trade by side + qty + timestamp."""
        from migrate_reconcile_pnl import match_fill_to_journal

        trades = _fetch_trades_as_dicts(
            populated_journal._conn,
            "SELECT id, timestamp, side, price, quantity FROM trades ORDER BY id",
        )

        fill = {
            "side": "buy",
            "amount": 0.001,
            "timestamp": int(
                datetime.datetime.fromisoformat(trades[0]["timestamp"]).timestamp() * 1000,
            ),
        }
        match = match_fill_to_journal(fill, trades)
        assert match is not None
        assert match["id"] == trades[0]["id"]

    def test_no_match_wrong_side(self, populated_journal):
        """Fill with wrong side should not match."""
        from migrate_reconcile_pnl import match_fill_to_journal

        trades = _fetch_trades_as_dicts(
            populated_journal._conn,
            "SELECT id, timestamp, side, price, quantity FROM trades WHERE side='BUY'",
        )

        fill = {"side": "sell", "amount": 0.001, "timestamp": 0}
        match = match_fill_to_journal(fill, trades)
        assert match is None

    def test_no_match_qty_too_different(self, populated_journal):
        """Fill with >5% quantity difference should not match."""
        from migrate_reconcile_pnl import match_fill_to_journal

        trades = _fetch_trades_as_dicts(
            populated_journal._conn,
            "SELECT id, timestamp, side, price, quantity FROM trades WHERE side='BUY'",
        )

        fill = {
            "side": "buy",
            "amount": 0.002,  # 100% off
            "timestamp": int(
                datetime.datetime.fromisoformat(trades[0]["timestamp"]).timestamp() * 1000,
            ),
        }
        match = match_fill_to_journal(fill, trades)
        assert match is None


class TestRunMigration:
    def test_corrects_pnl_with_fees(self, tmp_path):
        """Migration should recalculate P&L to include fees."""
        from unittest.mock import MagicMock, patch

        from migrate_reconcile_pnl import run_migration

        db_path = str(tmp_path / "trades.db")
        journal = TradeJournal(db_path=db_path)

        # Record a buy and sell with no fees
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="BUY",
            price=70000.0,
            quantity=0.001,
            value=70.0,
            currency="USDC",
            reasoning="buy",
        )
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="SELL",
            price=71000.0,
            quantity=0.001,
            value=71.0,
            currency="USDC",
            reasoning="sell",
            profit=1.0,
        )
        journal.close()

        # Mock exchange to return fills with fees
        mock_settings = MagicMock()
        mock_settings.kraken_api_key = MagicMock()
        mock_settings.kraken_private_key = MagicMock()
        mock_settings.quote_currency = "USDC"
        mock_settings.tax_rate = 0.19

        now_ms = int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp() * 1000)

        mock_exchange = MagicMock()
        mock_exchange.fetch_my_trades.return_value = [
            {
                "id": "TBUY1",
                "timestamp": now_ms - 5000,
                "symbol": "BTC/USDC",
                "side": "buy",
                "price": 70000.0,
                "amount": 0.001,
                "cost": 70.0,
                "fee": {"cost": 0.175, "currency": "USDC"},
            },
            {
                "id": "TSELL1",
                "timestamp": now_ms - 3000,
                "symbol": "BTC/USDC",
                "side": "sell",
                "price": 71000.0,
                "amount": 0.001,
                "cost": 71.0,
                "fee": {"cost": 0.1775, "currency": "USDC"},
            },
        ]

        with patch("migrate_reconcile_pnl.KrakenExchange", return_value=mock_exchange):
            report = run_migration(db_path, mock_settings, dry_run=False)

        assert report["fee_corrections"] >= 1
        assert report["pnl_corrections"] >= 1
        # New P&L should be less than old (fees deducted)
        assert report["new_total_pnl"] < report["old_total_pnl"]
        # Approximate: gross=1.0, fees=0.175+0.1775=0.3525, profit≈0.6475
        assert report["new_total_pnl"] == pytest.approx(0.6475, abs=0.05)

    def test_inserts_missing_sl_fill(self, tmp_path):
        """Migration should insert Kraken fills missing from journal."""
        from unittest.mock import MagicMock, patch

        from migrate_reconcile_pnl import run_migration

        db_path = str(tmp_path / "trades.db")
        journal = TradeJournal(db_path=db_path)

        # Only record a BUY — the SL sell is missing
        journal.log_trade(
            account="Kraken",
            instrument="DOGE/USDC",
            side="BUY",
            price=0.09,
            quantity=200.0,
            value=18.0,
            currency="USDC",
            reasoning="buy",
        )
        journal.close()

        mock_settings = MagicMock()
        mock_settings.kraken_api_key = MagicMock()
        mock_settings.kraken_private_key = MagicMock()
        mock_settings.quote_currency = "USDC"
        mock_settings.tax_rate = 0.19

        now_ms = int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp() * 1000)

        mock_exchange = MagicMock()
        mock_exchange.fetch_my_trades.return_value = [
            {
                "id": "TBUY_DOGE",
                "timestamp": now_ms - 10000,
                "symbol": "DOGE/USDC",
                "side": "buy",
                "price": 0.09,
                "amount": 200.0,
                "cost": 18.0,
                "fee": {"cost": 0.045, "currency": "USDC"},
            },
            {
                "id": "TSL_DOGE",
                "timestamp": now_ms - 5000,
                "symbol": "DOGE/USDC",
                "side": "sell",
                "price": 0.088,
                "amount": 200.0,
                "cost": 17.6,
                "fee": {"cost": 0.044, "currency": "USDC"},
            },
        ]

        with patch("migrate_reconcile_pnl.KrakenExchange", return_value=mock_exchange):
            report = run_migration(db_path, mock_settings, dry_run=False)

        assert report["missing_fills_inserted"] >= 1
        # Verify the missing sell was inserted
        conn = sqlite3.connect(db_path)
        sells = conn.execute(
            "SELECT price, fee, fill_source FROM trades WHERE side='SELL' AND instrument='DOGE/USDC'",
        ).fetchall()
        conn.close()
        assert len(sells) >= 1
        assert sells[0][0] == 0.088
        assert sells[0][2] == "exchange"

    def test_dry_run_does_not_modify(self, tmp_path):
        """Dry run should not change the database."""
        from unittest.mock import MagicMock, patch

        from migrate_reconcile_pnl import run_migration

        db_path = str(tmp_path / "trades.db")
        journal = TradeJournal(db_path=db_path)
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="BUY",
            price=70000.0,
            quantity=0.001,
            value=70.0,
            currency="USDC",
            reasoning="buy",
        )
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="SELL",
            price=71000.0,
            quantity=0.001,
            value=71.0,
            currency="USDC",
            reasoning="sell",
            profit=1.0,
        )
        journal.close()

        mock_settings = MagicMock()
        mock_settings.kraken_api_key = MagicMock()
        mock_settings.kraken_private_key = MagicMock()
        mock_settings.quote_currency = "USDC"
        mock_settings.tax_rate = 0.19

        mock_exchange = MagicMock()
        mock_exchange.fetch_my_trades.return_value = []

        with patch("migrate_reconcile_pnl.KrakenExchange", return_value=mock_exchange):
            report = run_migration(db_path, mock_settings, dry_run=True)

        # P&L should be unchanged
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT profit FROM trades WHERE side='SELL'").fetchone()
        conn.close()
        assert row[0] == 1.0  # Original value preserved
