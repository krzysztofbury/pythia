"""Cycle-scoped caching proxy for the exchange (thread-safe)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class CycleCache:
    """Thread-safe dict-based cache, created fresh each cycle."""

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._store: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get_or_compute(
        self,
        key: str,
        factory: Callable[[], Any],
    ) -> Any:  # noqa: ANN401
        """Return cached value or compute via factory (thread-safe)."""
        with self._lock:
            if key in self._store:
                return self._store[key]

        # Compute outside lock to avoid holding it during API calls
        value = factory()

        with self._lock:
            # Double-check: another thread may have populated it
            if key not in self._store:
                self._store[key] = value
            return self._store[key]

    def get(self, key: str) -> Any | None:  # noqa: ANN401
        """Return cached value or ``None``."""
        with self._lock:
            return self._store.get(key)

    def set(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Store a value."""
        with self._lock:
            self._store[key] = value

    def invalidate(self, *prefixes: str) -> None:
        """Remove all keys starting with any of the given prefixes."""
        with self._lock:
            keys_to_remove = [k for k in self._store if any(k.startswith(p) for p in prefixes)]
            for k in keys_to_remove:
                del self._store[k]


class CachedExchange:
    """Thread-safe caching proxy around KrakenExchange for one cycle.

    Read operations are cached; write operations pass through and
    invalidate relevant caches.
    """

    def __init__(self, exchange: Any) -> None:  # noqa: ANN401
        """Wrap exchange with per-cycle cache."""
        self._exchange = exchange
        self._cache = CycleCache()

    # -- Cached read operations -----------------------------------------------

    def fetch_price(self, symbol: str) -> float:
        """Fetch price (cached per symbol)."""
        return self._cache.get_or_compute(
            f"price:{symbol}",
            lambda: self._exchange.fetch_price(symbol),
        )

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> list[list]:
        """Fetch OHLCV candles (cached per symbol+timeframe+limit)."""
        return self._cache.get_or_compute(
            f"ohlcv:{symbol}:{timeframe}:{limit}",
            lambda: self._exchange.fetch_ohlcv(symbol, timeframe, limit),
        )

    def fetch_balance(self) -> dict:
        """Fetch full balance (cached)."""
        return self._cache.get_or_compute(
            "balance",
            lambda: self._exchange._client.fetch_balance(),  # noqa: SLF001
        )

    def get_free_balance(self, currency: str) -> float:
        """Free balance from cached fetch_balance."""
        balance = self.fetch_balance()
        return float(balance.get("free", {}).get(currency, 0.0))

    def get_total_balance(self, currency: str) -> float:
        """Total balance from cached fetch_balance."""
        balance = self.fetch_balance()
        return float(balance.get("total", {}).get(currency, 0.0))

    def has_open_position(self, symbol: str) -> bool:
        """Check position using cached balance + price."""
        assert "/" in symbol, f"Symbol must be BASE/QUOTE format, got {symbol!r}"  # noqa: S101

        base_currency = symbol.split("/")[0]
        balance = self.get_total_balance(base_currency)
        if balance <= 0:
            return False

        price = self.fetch_price(symbol)
        assert price >= 0, f"Price must be non-negative, got {price}"  # noqa: S101

        usd_value = balance * price
        return usd_value >= 1.0

    def get_24h_stats(self, symbol: str) -> dict[str, float]:
        """Fetch 24h stats (cached per symbol)."""
        return self._cache.get_or_compute(
            f"stats:{symbol}",
            lambda: self._exchange.get_24h_stats(symbol),
        )

    def get_market_structure(self, symbol: str) -> dict[str, float]:
        """Fetch market structure (cached per symbol)."""
        return self._cache.get_or_compute(
            f"structure:{symbol}",
            lambda: self._exchange.get_market_structure(symbol),
        )

    def get_open_orders(self, symbol: str) -> list[dict]:
        """Fetch open orders (cached per symbol)."""
        return self._cache.get_or_compute(
            f"orders:{symbol}",
            lambda: self._exchange.get_open_orders(symbol),
        )

    # -- Write operations (pass through + invalidate) -------------------------

    def place_maker_buy(self, symbol: str, price: float, amount: float) -> dict:
        """Place buy order and invalidate balance/orders cache."""
        result = self._exchange.place_maker_buy(symbol, price, amount)
        self._cache.invalidate("balance", f"orders:{symbol}")
        return result

    def place_maker_sell(self, symbol: str, price: float, amount: float) -> dict:
        """Place sell order and invalidate balance/orders cache."""
        result = self._exchange.place_maker_sell(symbol, price, amount)
        self._cache.invalidate("balance", f"orders:{symbol}")
        return result

    def cancel_all_orders(self, symbol: str) -> dict:
        """Cancel all orders and invalidate orders cache."""
        result = self._exchange.cancel_all_orders(symbol)
        self._cache.invalidate(f"orders:{symbol}")
        return result

    # -- Delegate everything else ---------------------------------------------

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Delegate unknown attributes to the real exchange."""
        return getattr(self._exchange, name)
