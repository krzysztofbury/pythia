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

    def test_trades_table_has_exchange_fill_id_column(self, journal):
        row = journal._conn.execute("PRAGMA table_info(trades)").fetchall()
        col_names = [r[1] for r in row]
        assert "exchange_fill_id" in col_names

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

    def test_fee_defaults_to_zero(self, journal):
        """Trades without explicit fee should default to 0.0."""
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
        row = journal._conn.execute("SELECT fee FROM trades ORDER BY id DESC LIMIT 1").fetchone()
        assert row[0] == 0.0

    def test_fill_source_defaults_to_bot(self, journal):
        """Trades without explicit fill_source should default to 'bot'."""
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
        row = journal._conn.execute(
            "SELECT fill_source FROM trades ORDER BY id DESC LIMIT 1",
        ).fetchone()
        assert row[0] == "bot"

    def test_migration_adds_columns_to_existing_db(self, tmp_path):
        """ALTER TABLE migration should add columns to a pre-existing DB without them."""
        import sqlite3

        db_path = str(tmp_path / "old.db")
        # Create a DB with the OLD schema (no fee/fill_source)
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                day_of_week TEXT NOT NULL,
                account TEXT NOT NULL,
                instrument TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                value REAL NOT NULL,
                currency TEXT NOT NULL,
                profit REAL NOT NULL DEFAULT 0.0,
                tax REAL NOT NULL DEFAULT 0.0,
                reasoning TEXT NOT NULL DEFAULT ''
            )
        """)
        conn.execute(
            "INSERT INTO trades (timestamp, date, day_of_week, account, instrument, side, price, quantity, value, currency, reasoning) "
            "VALUES ('2026-01-01T00:00:00+00:00', '2026-01-01', 'Wednesday', 'Kraken', 'BTC/USDC', 'BUY', 70000, 0.001, 70, 'USDC', 'old trade')",
        )
        conn.commit()
        conn.close()

        # Now open with TradeJournal — migration should run
        journal = TradeJournal(db_path=db_path)
        cols = [r[1] for r in journal._conn.execute("PRAGMA table_info(trades)").fetchall()]
        assert "fee" in cols
        assert "fill_source" in cols
        assert "exchange_fill_id" in cols
        # Old data should have defaults
        row = journal._conn.execute(
            "SELECT fee, fill_source, exchange_fill_id FROM trades"
        ).fetchone()
        assert row[0] == 0.0
        assert row[1] == "bot"
        assert row[2] is None


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
            exchange_fill_id="TXYZ123",
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

    def test_compute_pnl_with_fees_dca_multiple_buys(self, journal):
        """P&L with DCA: avg cost basis from multiple buys, proportional fees."""
        # Buy 1: 0.001 BTC @ 70000, fee=0.175
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="BUY",
            price=70000.0,
            quantity=0.001,
            value=70.0,
            currency="USDC",
            reasoning="first buy",
            fee=0.175,
        )
        # Buy 2 (DCA): 0.001 BTC @ 68000, fee=0.170
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="BUY",
            price=68000.0,
            quantity=0.001,
            value=68.0,
            currency="USDC",
            reasoning="DCA buy",
            fee=0.170,
        )
        # Avg cost = (70000*0.001 + 68000*0.001) / 0.002 = 69000
        # Total buy fees = 0.345, total buy qty = 0.002
        # Sell 0.002 @ 71000, sell_fee=0.355
        pnl, tax = journal.compute_pnl_with_fees(
            ticker="BTC/USDC",
            sell_price=71000.0,
            sell_qty=0.002,
            sell_fee=0.355,
        )
        # gross = (71000 - 69000) * 0.002 = 4.0
        # proportional buy fee = 0.345 * (0.002/0.002) = 0.345
        # profit = 4.0 - 0.345 - 0.355 = 3.3
        assert pnl == pytest.approx(3.3, abs=0.01)
        assert tax == pytest.approx(3.3 * 0.19, abs=0.01)

    def test_compute_pnl_no_prior_buys(self, journal):
        """Selling with no prior buy records should return zero profit."""
        pnl, tax = journal.compute_pnl_with_fees(
            ticker="UNKNOWN/USDC",
            sell_price=100.0,
            sell_qty=1.0,
            sell_fee=0.25,
        )
        # avg_cost = sell_price (fallback), gross = 0, profit = -0.25
        assert pnl == pytest.approx(-0.25, abs=0.01)
        assert tax == 0.0

    def test_get_last_sync_timestamp_ignores_other_tickers(self, journal):
        """Timestamp should only reflect trades for the requested ticker."""
        journal.log_trade(
            account="Kraken",
            instrument="ETH/USDC",
            side="BUY",
            price=2000.0,
            quantity=0.01,
            value=20.0,
            currency="USDC",
            reasoning="eth buy",
        )
        # BTC should still have no timestamp
        assert journal.get_last_sync_timestamp("BTC/USDC") is None

    def test_is_fill_recorded_different_id_returns_false(self, journal):
        """A different exchange_fill_id should not match."""
        journal.log_trade(
            account="Kraken",
            instrument="BTC/USDC",
            side="BUY",
            price=70000.0,
            quantity=0.001,
            value=70.0,
            currency="USDC",
            reasoning="Some reasoning",
            exchange_fill_id="TXYZ123",
        )
        assert journal.is_fill_recorded("TXYZ999") is False


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

        from order_executor import reconcile_fills as _reconcile_fills

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
            "SELECT side, price, fee, fill_source FROM trades WHERE exchange_fill_id = 'TSL001'",
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
            exchange_fill_id="TEXIST1",
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

        from order_executor import reconcile_fills as _reconcile_fills

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

    def test_reconcile_records_buy_fill(self, journal):
        """A Kraken BUY fill not in our journal gets recorded (zero P&L)."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_my_trades.return_value = [
            {
                "id": "TBUY01",
                "timestamp": 1712900000000,
                "symbol": "ETH/USDC",
                "side": "buy",
                "price": 2000.0,
                "amount": 0.01,
                "cost": 20.0,
                "fee": {"cost": 0.05, "currency": "USDC"},
            },
        ]

        from order_executor import reconcile_fills as _reconcile_fills

        mock_notifier = MagicMock()
        _reconcile_fills(
            tickers=["ETH/USDC"],
            exchange=mock_exchange,
            journal=journal,
            notifier=mock_notifier,
            quote_currency="USDC",
            tax_rate=0.19,
        )

        assert journal.is_fill_recorded("TBUY01")
        row = journal._conn.execute(
            "SELECT side, profit, fee, fill_source FROM trades WHERE exchange_fill_id = 'TBUY01'",
        ).fetchone()
        assert row[0] == "BUY"
        assert row[1] == 0.0  # BUY fills have no P&L
        assert row[2] == pytest.approx(0.05)
        assert row[3] == "exchange"

    def test_reconcile_handles_exchange_error(self, journal):
        """Exchange errors during reconciliation should not crash."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_my_trades.side_effect = Exception("Connection timeout")

        from order_executor import reconcile_fills as _reconcile_fills

        mock_notifier = MagicMock()
        # Should not raise
        _reconcile_fills(
            tickers=["BTC/USDC"],
            exchange=mock_exchange,
            journal=journal,
            notifier=mock_notifier,
            quote_currency="USDC",
            tax_rate=0.19,
        )

    def test_reconcile_skips_invalid_fills(self, journal):
        """Fills with zero price or missing side should be ignored."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_my_trades.return_value = [
            {"id": "BAD1", "side": "", "price": 0, "amount": 1, "cost": 0, "fee": {}},
            {"id": "BAD2", "side": "sell", "price": 0, "amount": 1, "cost": 0, "fee": {}},
            {"id": "", "side": "sell", "price": 100, "amount": 1, "cost": 100, "fee": {}},
        ]

        from order_executor import reconcile_fills as _reconcile_fills

        _reconcile_fills(
            tickers=["BTC/USDC"],
            exchange=mock_exchange,
            journal=journal,
            notifier=MagicMock(),
            quote_currency="USDC",
            tax_rate=0.19,
        )
        count = journal._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        assert count == 0

    def test_reconcile_sell_deletes_position_state(self, journal):
        """A reconciled SELL fill should delete position_state for that ticker."""
        from models import PositionState, StopMode

        # Set up position state
        state = PositionState(
            ticker="BTC/USDC",
            entry_price=70000.0,
            high_watermark=71000.0,
            stop_mode=StopMode.TRAILING,
            current_sl=69000.0,
            current_tp=73000.0,
        )
        journal.upsert_position_state(state)
        assert journal.get_position_state("BTC/USDC") is not None

        # Record a BUY so P&L calc works
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
                "id": "TSL099",
                "side": "sell",
                "price": 69000.0,
                "amount": 0.001,
                "cost": 69.0,
                "fee": {"cost": 0.17, "currency": "USDC"},
            },
        ]

        from order_executor import reconcile_fills as _reconcile_fills

        _reconcile_fills(
            tickers=["BTC/USDC"],
            exchange=mock_exchange,
            journal=journal,
            notifier=MagicMock(),
            quote_currency="USDC",
            tax_rate=0.19,
        )
        assert journal.get_position_state("BTC/USDC") is None

    def test_reconcile_sell_sends_notification(self, journal):
        """A reconciled SELL fill should trigger a trade notification."""
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
        mock_exchange = MagicMock()
        mock_exchange.fetch_my_trades.return_value = [
            {
                "id": "TSL100",
                "side": "sell",
                "price": 69500.0,
                "amount": 0.001,
                "cost": 69.5,
                "fee": {"cost": 0.17, "currency": "USDC"},
            },
        ]
        from order_executor import reconcile_fills as _reconcile_fills

        mock_notifier = MagicMock()
        _reconcile_fills(
            tickers=["BTC/USDC"],
            exchange=mock_exchange,
            journal=journal,
            notifier=mock_notifier,
            quote_currency="USDC",
            tax_rate=0.19,
        )
        mock_notifier.notify_trade.assert_called_once()
        call_args = mock_notifier.notify_trade.call_args
        assert call_args[0][0] == "BTC/USDC"
        assert call_args[0][1] == "SELL"

    def test_reconcile_multiple_fills_in_one_call(self, journal):
        """Multiple fills from one fetch should all be processed."""
        journal.log_trade(
            account="Kraken",
            instrument="LINK/USDC",
            side="BUY",
            price=9.0,
            quantity=2.0,
            value=18.0,
            currency="USDC",
            reasoning="buy",
        )
        mock_exchange = MagicMock()
        mock_exchange.fetch_my_trades.return_value = [
            {
                "id": "T1",
                "side": "sell",
                "price": 8.5,
                "amount": 1.0,
                "cost": 8.5,
                "fee": {"cost": 0.02},
            },
            {
                "id": "T2",
                "side": "buy",
                "price": 8.3,
                "amount": 1.5,
                "cost": 12.45,
                "fee": {"cost": 0.03},
            },
        ]
        from order_executor import reconcile_fills as _reconcile_fills

        _reconcile_fills(
            tickers=["LINK/USDC"],
            exchange=mock_exchange,
            journal=journal,
            notifier=MagicMock(),
            quote_currency="USDC",
            tax_rate=0.19,
        )
        assert journal.is_fill_recorded("T1")
        assert journal.is_fill_recorded("T2")
