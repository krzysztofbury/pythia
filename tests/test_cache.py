"""Tests for the CachedExchange proxy."""
from unittest.mock import MagicMock

from cache import CachedExchange


class TestCachedExchange:
    """Tests for CachedExchange caching behavior."""

    def _make_exchange(self):
        """Create a mock exchange."""
        exchange = MagicMock()
        exchange.fetch_price.return_value = 44000.0
        exchange.fetch_ohlcv.return_value = [[1, 2, 3, 4, 5, 6]]
        exchange.get_24h_stats.return_value = {"vwap": 43500.0, "volume": 1000.0}
        exchange.get_market_structure.return_value = {"bid": 43999, "ask": 44001, "spread_pct": 0.00005}
        exchange.get_open_orders.return_value = []
        exchange._client.fetch_balance.return_value = {
            "free": {"USDT": 500.0, "BTC": 0.001},
            "total": {"USDT": 500.0, "BTC": 0.001},
        }
        return exchange

    def test_price_cached_on_second_call(self):
        """fetch_price should only call the real exchange once."""
        mock = self._make_exchange()
        cached = CachedExchange(mock)

        p1 = cached.fetch_price("BTC/USDT")
        p2 = cached.fetch_price("BTC/USDT")

        assert p1 == p2 == 44000.0
        mock.fetch_price.assert_called_once()

    def test_ohlcv_cached(self):
        """fetch_ohlcv with same params should be cached."""
        mock = self._make_exchange()
        cached = CachedExchange(mock)

        cached.fetch_ohlcv("BTC/USDT", "1h", 100)
        cached.fetch_ohlcv("BTC/USDT", "1h", 100)

        mock.fetch_ohlcv.assert_called_once()

    def test_different_params_not_cached(self):
        """fetch_ohlcv with different timeframes should not share cache."""
        mock = self._make_exchange()
        cached = CachedExchange(mock)

        cached.fetch_ohlcv("BTC/USDT", "1h", 100)
        cached.fetch_ohlcv("BTC/USDT", "4h", 200)

        assert mock.fetch_ohlcv.call_count == 2

    def test_balance_cached(self):
        """Balance should be fetched once and cached."""
        mock = self._make_exchange()
        cached = CachedExchange(mock)

        b1 = cached.get_free_balance("USDT")
        b2 = cached.get_free_balance("BTC")

        assert b1 == 500.0
        assert b2 == 0.001
        mock._client.fetch_balance.assert_called_once()

    def test_write_invalidates_cache(self):
        """place_maker_buy should invalidate balance cache."""
        mock = self._make_exchange()
        mock.place_maker_buy.return_value = {"id": "123"}
        cached = CachedExchange(mock)

        # Populate cache
        cached.get_free_balance("USDT")
        assert mock._client.fetch_balance.call_count == 1

        # Write operation
        cached.place_maker_buy("BTC/USDT", 44000.0, 0.001)

        # Next balance call should re-fetch
        cached.get_free_balance("USDT")
        assert mock._client.fetch_balance.call_count == 2

    def test_getattr_delegates(self):
        """Unknown methods should delegate to real exchange."""
        mock = self._make_exchange()
        mock.symbol_exists.return_value = True
        cached = CachedExchange(mock)

        assert cached.symbol_exists("BTC/USDT") is True
        mock.symbol_exists.assert_called_once_with("BTC/USDT")
