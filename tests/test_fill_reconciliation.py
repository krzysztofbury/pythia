"""Tests for fill reconciliation and P&L with fees."""

import datetime
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

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


class TestFetchMyTrades:
    def test_fetch_my_trades_returns_fills(self):
        from exchange import KrakenExchange

        mock_fills = [
            {
                "id": "TXXXXX",
                "timestamp": 1712880000000,
                "symbol": "BTC/USDC",
                "side": "sell",
                "price": 70000.0,
                "amount": 0.001,
                "cost": 70.0,
                "fee": {"cost": 0.175, "currency": "USDC"},
            },
        ]
        with patch("exchange.ccxt.kraken") as MockKraken:
            instance = MockKraken.return_value
            instance.load_markets.return_value = {}
            instance.markets = {}
            instance.fetch_my_trades.return_value = mock_fills
            ex = KrakenExchange(api_key=SecretStr("test"), private_key=SecretStr("test"))
            result = ex.fetch_my_trades("BTC/USDC", since_ms=1712800000000)
        assert len(result) == 1
        assert result[0]["id"] == "TXXXXX"
        assert result[0]["fee"]["cost"] == 0.175
        instance.fetch_my_trades.assert_called_once_with("BTC/USDC", since=1712800000000, limit=200)

    def test_fetch_my_trades_no_since(self):
        from exchange import KrakenExchange

        with patch("exchange.ccxt.kraken") as MockKraken:
            instance = MockKraken.return_value
            instance.load_markets.return_value = {}
            instance.markets = {}
            instance.fetch_my_trades.return_value = []
            ex = KrakenExchange(api_key=SecretStr("test"), private_key=SecretStr("test"))
            result = ex.fetch_my_trades("DOGE/USDC")
        assert result == []
        instance.fetch_my_trades.assert_called_once_with("DOGE/USDC", since=None, limit=200)


class TestFillSync:
    def test_get_last_sync_timestamp_empty(self, journal):
        assert journal.get_last_sync_timestamp("BTC/USDC") is None

    def test_get_last_sync_timestamp_returns_latest(self, journal):
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="BUY",
            price=70000.0,
            quantity=0.001,
            value=70.0,
            currency="USDC",
            reasoning="test",
        )
        ts = journal.get_last_sync_timestamp("BTC/USDC")
        assert ts is not None
        assert isinstance(ts, int)
        now_ms = int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp() * 1000)
        assert abs(ts - now_ms) < 5000

    def test_is_fill_recorded_false(self, journal):
        assert journal.is_fill_recorded("TXXXXXX") is False

    def test_is_fill_recorded_true(self, journal):
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="SELL",
            price=71000.0,
            quantity=0.001,
            value=71.0,
            currency="USDC",
            reasoning="SL fill [TXYZ123]",
            fee=0.18,
            fill_source="exchange",
        )
        assert journal.is_fill_recorded("TXYZ123") is True

    def test_compute_pnl_with_fees(self, journal):
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="BUY",
            price=70000.0,
            quantity=0.001,
            value=70.0,
            currency="USDC",
            reasoning="test buy",
            fee=0.175,
        )
        pnl, tax = journal.compute_pnl_with_fees(
            ticker="BTC/USDC",
            sell_price=71000.0,
            sell_qty=0.001,
            sell_fee=0.1775,
        )
        assert pnl == pytest.approx(0.6475, abs=0.01)
        assert tax == pytest.approx(0.6475 * 0.19, abs=0.01)

    def test_compute_pnl_with_fees_loss(self, journal):
        journal.log_trade(
            account="Kraken",
            instrument="DOGE/USDC",
            side="BUY",
            price=0.09,
            quantity=200.0,
            value=18.0,
            currency="USDC",
            reasoning="test",
            fee=0.045,
        )
        pnl, tax = journal.compute_pnl_with_fees(
            ticker="DOGE/USDC",
            sell_price=0.088,
            sell_qty=200.0,
            sell_fee=0.044,
        )
        assert pnl < 0
        assert tax == 0.0


class TestReconcileFills:
    def test_reconcile_records_missing_sl_fill(self, journal):
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

        mock_exchange = MagicMock()
        mock_exchange.fetch_my_trades.return_value = [
            {
                "id": "TSL001",
                "timestamp": 1712900000000,
                "datetime": "2026-04-12T12:00:00.000Z",
                "symbol": "BTC/USDC",
                "side": "sell",
                "price": 69000.0,
                "amount": 0.001,
                "cost": 69.0,
                "fee": {"cost": 0.276, "currency": "USDC"},
                "order": "OSL001",
                "type": "market",
            },
        ]

        from main import _reconcile_fills

        mock_notifier = MagicMock()
        _reconcile_fills(
            tickers=["BTC/USDC"],
            exchange=mock_exchange,
            journal=journal,
            notifier=mock_notifier,
            quote_currency="USDC",
            tax_rate=0.19,
        )

        assert journal.is_fill_recorded("TSL001")
        row = journal._conn.execute(
            "SELECT side, price, fee, fill_source FROM trades WHERE reasoning LIKE '%TSL001%'",
        ).fetchone()
        assert row[0] == "SELL"
        assert row[1] == 69000.0
        assert row[2] == pytest.approx(0.276)
        assert row[3] == "exchange"

    def test_reconcile_skips_already_recorded(self, journal):
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="SELL",
            price=69000.0,
            quantity=0.001,
            value=69.0,
            currency="USDC",
            reasoning="SL fill [TEXIST1]",
            fee=0.276,
            fill_source="exchange",
        )

        mock_exchange = MagicMock()
        mock_exchange.fetch_my_trades.return_value = [
            {
                "id": "TEXIST1",
                "timestamp": 1712900000000,
                "datetime": "2026-04-12T12:00:00.000Z",
                "symbol": "BTC/USDC",
                "side": "sell",
                "price": 69000.0,
                "amount": 0.001,
                "cost": 69.0,
                "fee": {"cost": 0.276, "currency": "USDC"},
                "order": "O1",
                "type": "market",
            },
        ]

        from main import _reconcile_fills

        mock_notifier = MagicMock()
        _reconcile_fills(
            tickers=["BTC/USDC"],
            exchange=mock_exchange,
            journal=journal,
            notifier=mock_notifier,
            quote_currency="USDC",
            tax_rate=0.19,
        )

        count = journal._conn.execute("SELECT COUNT(*) FROM trades WHERE side='SELL'").fetchone()[0]
        assert count == 1
