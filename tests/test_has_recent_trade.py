"""Tests for TradeJournal.has_recent_trade dedup helper."""

import datetime
from pathlib import Path

from journal import TradeJournal


class TestHasRecentTrade:
    """Verify duplicate-detection via has_recent_trade."""

    def _make_journal(self, tmp_path: Path) -> TradeJournal:
        return TradeJournal(db_path=str(tmp_path / "test.db"))

    def test_exact_match_within_window(self, tmp_path: Path) -> None:
        """A trade logged just now should be detected."""
        j = self._make_journal(tmp_path)
        j.log_trade(
            account="Kraken",
            instrument="BTC/USDT",
            side="SELL",
            price=50000.0,
            quantity=0.01,
            value=500.0,
            currency="USDT",
            reasoning="TP hit",
        )
        assert j.has_recent_trade("BTC/USDT", "SELL", 0.01) is True
        j.close()

    def test_quantity_within_5pct(self, tmp_path: Path) -> None:
        """Quantities within 5% should match."""
        j = self._make_journal(tmp_path)
        j.log_trade(
            account="Kraken",
            instrument="ETH/USDT",
            side="BUY",
            price=3000.0,
            quantity=1.0,
            value=3000.0,
            currency="USDT",
            reasoning="entry",
        )
        # 4% off -> should match
        assert j.has_recent_trade("ETH/USDT", "BUY", 1.04) is True
        # 6% off -> should NOT match
        assert j.has_recent_trade("ETH/USDT", "BUY", 1.06) is False
        j.close()

    def test_different_side_no_match(self, tmp_path: Path) -> None:
        """Same ticker and qty but different side should not match."""
        j = self._make_journal(tmp_path)
        j.log_trade(
            account="Kraken",
            instrument="SOL/USDT",
            side="BUY",
            price=100.0,
            quantity=5.0,
            value=500.0,
            currency="USDT",
            reasoning="buy",
        )
        assert j.has_recent_trade("SOL/USDT", "SELL", 5.0) is False
        j.close()

    def test_different_ticker_no_match(self, tmp_path: Path) -> None:
        """Same side and qty but different ticker should not match."""
        j = self._make_journal(tmp_path)
        j.log_trade(
            account="Kraken",
            instrument="BTC/USDT",
            side="SELL",
            price=50000.0,
            quantity=0.01,
            value=500.0,
            currency="USDT",
            reasoning="sell",
        )
        assert j.has_recent_trade("ETH/USDT", "SELL", 0.01) is False
        j.close()

    def test_old_trade_outside_window(self, tmp_path: Path) -> None:
        """A trade older than the window should not match."""
        j = self._make_journal(tmp_path)
        # Insert a trade with a timestamp 20 minutes ago directly
        old_ts = (
            datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(minutes=20)
        ).isoformat()
        j._conn.execute(  # noqa: SLF001
            "INSERT INTO trades "
            "(timestamp, date, day_of_week, account, instrument, side, "
            " price, quantity, value, currency, profit, tax, reasoning, "
            " fee, fill_source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                old_ts,
                "2026-04-12",
                "Sunday",
                "Kraken",
                "BTC/USDT",
                "SELL",
                50000.0,
                0.01,
                500.0,
                "USDT",
                0.0,
                0.0,
                "old sell",
                0.0,
                "bot",
            ),
        )
        j._conn.commit()  # noqa: SLF001

        # Default window is 15 minutes, so 20-minute-old trade should NOT match
        assert j.has_recent_trade("BTC/USDT", "SELL", 0.01) is False
        # But with a 25-minute window it should
        assert j.has_recent_trade("BTC/USDT", "SELL", 0.01, window_minutes=25) is True
        j.close()

    def test_no_trades_returns_false(self, tmp_path: Path) -> None:
        """Empty journal should return False."""
        j = self._make_journal(tmp_path)
        assert j.has_recent_trade("BTC/USDT", "BUY", 1.0) is False
        j.close()
